/*---------------------------------------------------------------------------*\
  fgmFluid — thermoGPU offload of the per-cell property refresh

  thermo_.correct()가 도는 RhoFluidThermo::calculate() 루프(셀·경계면마다
  he→T 뉴턴 역산 + Cp/Cv/psi/rho/mu/kappa 체인, 셀당 SRK 3차방정식 ~10회)를
  단일 CUDA 배치 평가로 대체한다.

  fgmFluid에서 T는 매니폴드에서 직접 읽혀 updateManifold()가 이미 세팅했고
  he는 he(p,T)로 재시드된 진단량이므로, CPU 경로의 he→T 역산은 T_table을
  뉴턴 허용오차 내로 되돌려줄 뿐이다. GPU 경로는 역산을 생략하고 T를 그대로
  둔 채 (p, T, Y)에서 물성만 일괄 재계산한다 — 매니폴드와 더 정확히 일치.

  경계면도 같은 배치에 이어붙여 한 번의 커널 launch로 처리한다.
\*---------------------------------------------------------------------------*/

#include "fgmFluid.H"
#include "tabulatedRealGasMixture.H"
#include "thermodynamicConstants.H"
#include "rhoThermo.H"

#include "gpu/rgpKernelTypes.H"
#include "gpu/rgpFgmTypes.H"
#include <cstring>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::armGpuThermo()
{
    const tabulatedRealGasMixture* hook =
        dynamic_cast<const tabulatedRealGasMixture*>(&thermo_);

    scalarList W, BM, CM, jH, jL;
    List<scalarList> pair;
    scalar TlowJ = 0, ThighJ = 0, Tcommon = 0;
    bool stableRoot = false;

    if
    (
        hook == nullptr
     || !hook->gpuThermoTables
        (
            W, BM, CM, jH, jL, pair, TlowJ, ThighJ, Tcommon, stableRoot
        )
    )
    {
        FatalErrorInFunction
            << "'gpuThermo on;' requires a mixture with GPU-kernelised "
            << "thermophysics (SRKchungTakaMixture). The active mixture "
            << "does not provide gpuThermoTables()."
            << exit(FatalError);
    }

    const int nDev = rgpGpuDeviceCount();
    if (nDev == 0)
    {
        FatalErrorInFunction
            << "'gpuThermo on;' but no CUDA device is visible "
            << "(container: --nv; WSL2: also --bind /usr/lib/wsl and "
            << "LD_LIBRARY_PATH=/usr/lib/wsl/lib)."
            << exit(FatalError);
    }

    // 랭크→디바이스 매핑 (단일 GPU면 전 랭크가 device 0을 공유)
    const int dev = Pstream::parRun() ? (Pstream::myProcNo() % nDev) : 0;
    int err = rgpGpuInit(dev);
    if (err)
    {
        FatalErrorInFunction
            << "rgpGpuInit(" << dev << "): " << rgpGpuLastError()
            << exit(FatalError);
    }

    rgpGpuTables t;
    t.nSpecies   = W.size();
    t.stableRoot = stableRoot ? 1 : 0;
    t.RR         = constant::thermodynamic::RR;
    t.TlowJ      = TlowJ;
    t.ThighJ     = ThighJ;
    t.Tcommon    = Tcommon;
    t.W = W.begin();  t.BM = BM.begin();  t.CM = CM.begin();
    t.janafHigh = jH.begin();  t.janafLow = jL.begin();
    t.COEF1 = pair[0].begin();  t.COEF2 = pair[1].begin();
    t.COEF3 = pair[2].begin();  t.SIGMA3M = pair[3].begin();
    t.EPSILONKM0 = pair[4].begin();  t.OMEGAM0 = pair[5].begin();
    t.MM0 = pair[6].begin();  t.MIUIM0 = pair[7].begin();
    t.KAPPAIM = pair[8].begin();

    err = rgpGpuUpload(&t);
    if (err)
    {
        FatalErrorInFunction
            << "rgpGpuUpload: " << rgpGpuLastError()
            << exit(FatalError);
    }

    Info<< "fgmFluid: thermoGPU ARMED -- " << W.size()
        << "-species SRK+Chung tables on CUDA device " << dev
        << " (of " << nDev << "); thermo_.correct() replaced by the "
        << "batched GPU property refresh" << nl << endl;

    gpuArmed_ = true;
}


void Foam::solvers::fgmFluid::gpuThermoCorrect()
{
    if (!gpuArmed_)
    {
        armGpuThermo();
    }

    const label n = Y_.size();
    const label nInt = mesh.nCells();

    const volScalarField& pf = thermo_.p();
    const volScalarField& Tf = thermo_.T();

    // 배치 크기: 내부 셀 + 전체 패치 면
    label N = nInt;
    forAll(Tf.boundaryField(), patchi)
    {
        N += Tf.boundaryField()[patchi].size();
    }

    const label nB = N - nInt;

    if (gpuP_.size() < N)
    {
        gpuP_.setSize(N);      gpuT_.setSize(N);      gpuY_.setSize(N*n);
    }
    if (gpuRho_.size() < N)   // gpuHeReseed는 입력 버퍼만 사이징한다
    {
        gpuRho_.setSize(N);    gpuMu_.setSize(N);     gpuKappa_.setSize(N);
        gpuCp_.setSize(N);     gpuCv_.setSize(N);     gpuPsi_.setSize(N);
    }

    // ── fgm 디바이스 체인: 직전 updateManifold의 (T, Y_tab) SoA가 아직
    // 디바이스에 있으므로 내부 셀의 T/Y gather + H2D 재업로드를 제거하고
    // p만 올린다. 비테이블 종(보통 0-1개)만 호스트 SoA로 보충.
    bool chained = false;
    if (gpuManifoldArmed_ && rgpFgmDevLastN() == nInt)
    {
        List<int> yMap(n, -1);
        forAll(tabSpecieIDs_, k) { yMap[tabSpecieIDs_[k]] = 2 + k; }
        label nH = 0;
        forAll(yMap, s) { if (yMap[s] < 0) { yMap[s] = -1 - nH; nH++; } }

        label h = 0;
        forAll(yMap, s)
        {
            if (yMap[s] < 0)
            {
                std::memcpy
                (
                    &gpuY_[h*nInt],
                    Y_[s].primitiveField().begin(),
                    nInt*sizeof(double)
                );
                h++;
            }
        }

        std::memcpy
        (
            gpuP_.begin(), pf.primitiveField().begin(), nInt*sizeof(double)
        );

        chained =
            rgpGpuEvaluateFromFgm
            (
                nInt,
                gpuP_.begin(), yMap.begin(), nH, gpuY_.begin(),
                gpuRho_.begin(), gpuMu_.begin(), gpuKappa_.begin(),
                gpuCp_.begin(), gpuCv_.begin(), gpuPsi_.begin()
            ) == 0;

        if (!chained)
        {
            WarningInFunction
                << "fgm device chain failed (" << rgpGpuLastError()
                << ") -- falling back to the full-batch path" << endl;
        }
    }

    // ── gather: 내부 셀 (비체인 폴백만) ──────────────────────────────
    if (!chained)
    {
        const scalarField& pc = pf.primitiveField();
        const scalarField& Tc = Tf.primitiveField();
        for (label i = 0; i < nInt; i++)
        {
            gpuP_[i] = pc[i];
            gpuT_[i] = Tc[i];
        }
        for (label s = 0; s < n; s++)
        {
            const scalarField& Ys = Y_[s].primitiveField();
            for (label i = 0; i < nInt; i++)
            {
                gpuY_[i*n + s] = Ys[i];
            }
        }
    }

    // ── gather: 패치 면 (양 경로 공통 — 경계는 항상 호스트에서) ──────
    label off = nInt;
    forAll(Tf.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pf.boundaryField()[patchi];
        const fvPatchScalarField& Tp = Tf.boundaryField()[patchi];
        const label np = Tp.size();
        for (label fi = 0; fi < np; fi++)
        {
            gpuP_[off + fi] = pp[fi];
            gpuT_[off + fi] = Tp[fi];
        }
        for (label s = 0; s < n; s++)
        {
            const fvPatchScalarField& Yp = Y_[s].boundaryField()[patchi];
            for (label fi = 0; fi < np; fi++)
            {
                gpuY_[(off + fi)*n + s] = Yp[fi];
            }
        }
        off += np;
    }

    // ── GPU 일괄 평가: 비체인=전체 배치 / 체인=경계면 소배치 ─────────
    if (!chained)
    {
        const int err = rgpGpuEvaluate
        (
            N,
            gpuP_.begin(), gpuT_.begin(), gpuY_.begin(),
            gpuRho_.begin(), gpuMu_.begin(), gpuKappa_.begin(),
            gpuCp_.begin(), gpuCv_.begin(), gpuPsi_.begin()
        );
        if (err)
        {
            FatalErrorInFunction
                << "rgpGpuEvaluate (" << N << " states): "
                << rgpGpuLastError()
                << exit(FatalError);
        }
    }
    else if (nB > 0)
    {
        const int err = rgpGpuEvaluate
        (
            nB,
            gpuP_.begin() + nInt, gpuT_.begin() + nInt,
            gpuY_.begin() + nInt*n,
            gpuRho_.begin() + nInt, gpuMu_.begin() + nInt,
            gpuKappa_.begin() + nInt, gpuCp_.begin() + nInt,
            gpuCv_.begin() + nInt, gpuPsi_.begin() + nInt
        );
        if (err)
        {
            FatalErrorInFunction
                << "rgpGpuEvaluate (boundary, " << nB << " states): "
                << rgpGpuLastError()
                << exit(FatalError);
        }
    }

    // ── scatter: thermo 필드 (RhoFluidThermo::calculate()가 채우는 세트).
    // rho는 rhoThermo의 non-const 접근자로, 나머지는 basicThermo가 const
    // 접근자만 노출하므로 const_cast로 쓴다 -- thermo_.correct()가 내부에서
    // 하는 갱신을 대신하는 것.
    volScalarField& rhoT   = dynamic_cast<rhoThermo&>(thermo_).rho();
    volScalarField& psiT   = const_cast<volScalarField&>(thermo_.psi());
    volScalarField& muT    = const_cast<volScalarField&>(thermo_.mu());
    volScalarField& kappaT = const_cast<volScalarField&>(thermo_.kappa());
    volScalarField& CpT    = const_cast<volScalarField&>(thermo_.Cp());
    volScalarField& CvT    = const_cast<volScalarField&>(thermo_.Cv());

    {
        scalarField& rhoc   = rhoT.primitiveFieldRef();
        scalarField& psic   = psiT.primitiveFieldRef();
        scalarField& muc    = muT.primitiveFieldRef();
        scalarField& kappac = kappaT.primitiveFieldRef();
        scalarField& Cpc    = CpT.primitiveFieldRef();
        scalarField& Cvc    = CvT.primitiveFieldRef();
        for (label i = 0; i < nInt; i++)
        {
            rhoc[i]   = gpuRho_[i];
            psic[i]   = gpuPsi_[i];
            muc[i]    = gpuMu_[i];
            kappac[i] = gpuKappa_[i];
            Cpc[i]    = gpuCp_[i];
            Cvc[i]    = gpuCv_[i];
        }
    }

    off = nInt;
    forAll(Tf.boundaryField(), patchi)
    {
        fvPatchScalarField& rhop   = rhoT.boundaryFieldRef()[patchi];
        fvPatchScalarField& psip   = psiT.boundaryFieldRef()[patchi];
        fvPatchScalarField& mup    = muT.boundaryFieldRef()[patchi];
        fvPatchScalarField& kappap = kappaT.boundaryFieldRef()[patchi];
        fvPatchScalarField& Cpp    = CpT.boundaryFieldRef()[patchi];
        fvPatchScalarField& Cvp    = CvT.boundaryFieldRef()[patchi];
        const label np = rhop.size();
        for (label fi = 0; fi < np; fi++)
        {
            rhop[fi]   = gpuRho_[off + fi];
            psip[fi]   = gpuPsi_[off + fi];
            mup[fi]    = gpuMu_[off + fi];
            kappap[fi] = gpuKappa_[off + fi];
            Cpp[fi]    = gpuCp_[off + fi];
            Cvp[fi]    = gpuCv_[off + fi];
        }
        off += np;
    }
}


void Foam::solvers::fgmFluid::gpuHeReseed()
{
    if (!gpuArmed_)
    {
        armGpuThermo();
    }

    const label n = Y_.size();
    const label nInt = mesh.nCells();

    const volScalarField& pf = thermo_.p();
    const volScalarField& Tf = thermo_.T();

    // 배치 크기: 내부 셀 + 전체 패치 면 (thermo_.he(p,T) 대입과 동일 범위)
    label N = nInt;
    forAll(Tf.boundaryField(), patchi)
    {
        N += Tf.boundaryField()[patchi].size();
    }

    const label nB = N - nInt;

    if (gpuP_.size() < N)
    {
        gpuP_.setSize(N);  gpuT_.setSize(N);  gpuY_.setSize(N*n);
    }
    if (gpuCp_.size() < N)
    {
        gpuCp_.setSize(N);   // ha 출력 버퍼로 재사용
    }

    // ── fgm 디바이스 체인: (T, Y_tab)는 직전 fgm evaluate의 디바이스
    // SoA에서 직접 — 내부 셀은 p 업로드만으로 ha 평가.
    bool chained = false;
    if (gpuManifoldArmed_ && rgpFgmDevLastN() == nInt)
    {
        List<int> yMap(n, -1);
        forAll(tabSpecieIDs_, k) { yMap[tabSpecieIDs_[k]] = 2 + k; }
        label nH = 0;
        forAll(yMap, s) { if (yMap[s] < 0) { yMap[s] = -1 - nH; nH++; } }

        label h = 0;
        forAll(yMap, s)
        {
            if (yMap[s] < 0)
            {
                std::memcpy
                (
                    &gpuY_[h*nInt],
                    Y_[s].primitiveField().begin(),
                    nInt*sizeof(double)
                );
                h++;
            }
        }

        std::memcpy
        (
            gpuP_.begin(), pf.primitiveField().begin(), nInt*sizeof(double)
        );

        chained =
            rgpGpuEvaluateHaFromFgm
            (
                nInt,
                gpuP_.begin(), yMap.begin(), nH, gpuY_.begin(),
                gpuCp_.begin()
            ) == 0;

        if (!chained)
        {
            WarningInFunction
                << "fgm device chain failed (" << rgpGpuLastError()
                << ") -- falling back to the full-batch path" << endl;
        }
    }

    // ── gather: 내부 셀 (비체인 폴백만) ──────────────────────────────
    if (!chained)
    {
        const scalarField& pc = pf.primitiveField();
        const scalarField& Tc = Tf.primitiveField();
        for (label i = 0; i < nInt; i++)
        {
            gpuP_[i] = pc[i];
            gpuT_[i] = Tc[i];
        }
        for (label s = 0; s < n; s++)
        {
            const scalarField& Ys = Y_[s].primitiveField();
            for (label i = 0; i < nInt; i++)
            {
                gpuY_[i*n + s] = Ys[i];
            }
        }
    }

    // ── gather: 패치 면 (양 경로 공통) ───────────────────────────────
    label off = nInt;
    forAll(Tf.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = pf.boundaryField()[patchi];
        const fvPatchScalarField& Tp = Tf.boundaryField()[patchi];
        const label np = Tp.size();
        for (label fi = 0; fi < np; fi++)
        {
            gpuP_[off + fi] = pp[fi];
            gpuT_[off + fi] = Tp[fi];
        }
        for (label s = 0; s < n; s++)
        {
            const fvPatchScalarField& Yp = Y_[s].boundaryField()[patchi];
            for (label fi = 0; fi < np; fi++)
            {
                gpuY_[(off + fi)*n + s] = Yp[fi];
            }
        }
        off += np;
    }

    // ── GPU 일괄 ha 평가: 비체인=전체 / 체인=경계면 소배치 ───────────
    if (!chained)
    {
        const int err = rgpGpuEvaluateHa
        (
            N,
            gpuP_.begin(), gpuT_.begin(), gpuY_.begin(), gpuCp_.begin()
        );
        if (err)
        {
            FatalErrorInFunction
                << "rgpGpuEvaluateHa (" << N << " states): "
                << rgpGpuLastError()
                << exit(FatalError);
        }
    }
    else if (nB > 0)
    {
        const int err = rgpGpuEvaluateHa
        (
            nB,
            gpuP_.begin() + nInt, gpuT_.begin() + nInt,
            gpuY_.begin() + nInt*n, gpuCp_.begin() + nInt
        );
        if (err)
        {
            FatalErrorInFunction
                << "rgpGpuEvaluateHa (boundary, " << nB << " states): "
                << rgpGpuLastError()
                << exit(FatalError);
        }
    }

    // ── scatter: he (내부 + 경계, thermo_.he(p,T) 대입과 동일 커버리지) ─
    volScalarField& he = thermo_.he();
    {
        scalarField& hec = he.primitiveFieldRef();
        for (label i = 0; i < nInt; i++)
        {
            hec[i] = gpuCp_[i];
        }
    }
    off = nInt;
    forAll(he.boundaryField(), patchi)
    {
        fvPatchScalarField& hep = he.boundaryFieldRef()[patchi];
        const label np = hep.size();
        for (label fi = 0; fi < np; fi++)
        {
            hep[fi] = gpuCp_[off + fi];
        }
        off += np;
    }
}


// ************************************************************************* //
