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
#include <cstdlib>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

int Foam::solvers::fgmFluid::gpuSelectDevice() const
{
    const int nDev = rgpGpuDeviceCount();
    if (nDev == 0)
    {
        return -1;
    }

    // 랭크→디바이스 매핑 (단일 GPU면 전 랭크가 device 0을 공유).
    // cudaSetDevice는 idempotent — thermo/manifold 아밍 어느 쪽이 먼저
    // 호출해도 같은 디바이스에 할당이 이뤄진다.
    const int dev = Pstream::parRun() ? (Pstream::myProcNo() % nDev) : 0;
    if (rgpGpuInit(dev))
    {
        return -2;
    }
    return dev;
}


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

    const int dev = gpuSelectDevice();
    if (dev == -1)
    {
        FatalErrorInFunction
            << "'gpuThermo on;' but no CUDA device is visible "
            << "(container: --nv; WSL2: also --bind /usr/lib/wsl and "
            << "LD_LIBRARY_PATH=/usr/lib/wsl/lib)."
            << exit(FatalError);
    }
    if (dev < 0)
    {
        FatalErrorInFunction
            << "rgpGpuInit: " << rgpGpuLastError()
            << exit(FatalError);
    }

    int err = 0;

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

    const int um = rgpGpuUnifiedMode();
    Info<< "fgmFluid: thermoGPU ARMED -- " << W.size()
        << "-species SRK+Chung tables on CUDA device " << dev
        << " (of " << rgpGpuDeviceCount()
        << "); thermo_.correct() replaced by the "
        << "batched GPU property refresh" << nl
        << "    memory mode: "
        << (um == 2 ? "unified-native (coherent, zero-copy)"
          : um == 1 ? "unified-mapped (pinned zero-copy, validation)"
          : "explicit-copy (discrete GPU)")
        << " [coherent HW: " << (rgpGpuCoherentHW() ? "yes" : "no")
        << "]" << nl << endl;

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

        // Tier-2: 매니폴드 SoA의 RG_* 13필드 인덱스 — 커널이 계수를
        // 직접 소비해 셀당 O(n^2) 혼합 제거 (CPU lookup 경로와 1:1)
        List<int> cMap;
        if (tabRealGasCoeffs_)
        {
            cMap.setSize(13);
            forAll(cMap, k)
            {
                cMap[k] = 2 + tabSpecieIDs_.size() + k;
            }
        }

        chained =
            rgpGpuEvaluateFromFgm
            (
                nInt,
                gpuP_.begin(), yMap.begin(), nH, gpuY_.begin(),
                cMap.size() ? cMap.begin() : nullptr,
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
        // Y-다이어트 중이면 호스트 내부 Y가 stale — SoA에서 먼저 복원
        syncHostTabY();

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

    // gpuManifold 없이 gpuThermo만 켠 경우의 pin 훅 (updateManifold의
    // 트리거가 닿지 않음). 켜진 스위치의 grow-only 버퍼가 전부 최종
    // 크기에 도달한 뒤 1회.
    if
    (
        !gpuPinned_ && !gpuManifold_
     && (!gpuPEqn_ || gpuPEqnFlux_.size() > 0)
     && (!gpuUEqn_ || gpuUBuf_.size() > 0)
    )
    {
        pinGpuHostBuffers();
    }
}


void Foam::solvers::fgmFluid::gpuHeReseed()
{
    // gpuManifold만 켠 경우: 믹스처가 GPU thermo 테이블을 제공하지 않으면
    // (예: elyHanley) he 재시드만 CPU에 남기고 매니폴드 보간은 GPU 유지.
    // gpuThermo가 켜져 있으면 armGpuThermo가 Fatal로 명시 처리한다.
    if (gpuHeMode_ < 0)
    {
        if (gpuArmed_ || gpuThermo_)
        {
            gpuHeMode_ = 1;
        }
        else
        {
            const tabulatedRealGasMixture* hook =
                dynamic_cast<const tabulatedRealGasMixture*>(&thermo_);

            scalarList W, BM, CM, jH, jL;
            List<scalarList> pair;
            scalar TlowJ = 0, ThighJ = 0, Tcommon = 0;
            bool stableRoot = false;

            gpuHeMode_ =
            (
                hook
             && hook->gpuThermoTables
                (
                    W, BM, CM, jH, jL, pair, TlowJ, ThighJ, Tcommon,
                    stableRoot
                )
            ) ? 1 : 0;

            if (!gpuHeMode_)
            {
                WarningInFunction
                    << "gpuManifold: the active mixture does not provide "
                    << "gpuThermoTables() -- the he re-seed stays on the "
                    << "CPU (manifold interpolation remains on the GPU)"
                    << endl;
            }
        }
    }
    if (!gpuHeMode_)
    {
        thermo_.he() = thermo_.he(thermo_.p(), thermo_.T());
        return;
    }

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

        List<int> cMap;
        if (tabRealGasCoeffs_)
        {
            cMap.setSize(13);
            forAll(cMap, k)
            {
                cMap[k] = 2 + tabSpecieIDs_.size() + k;
            }
        }

        chained =
            rgpGpuEvaluateHaFromFgm
            (
                nInt,
                gpuP_.begin(), yMap.begin(), nH, gpuY_.begin(),
                cMap.size() ? cMap.begin() : nullptr,
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
        // Y-다이어트 중이면 호스트 내부 Y가 stale — SoA에서 먼저 복원
        syncHostTabY();

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


void Foam::solvers::fgmFluid::pinGpuHostBuffers()
{
    if (gpuPinned_) return;

    // 디버그/격리용: env RGP_NO_PIN=1 이면 page-lock 등록을 끈다
    if (getenv("RGP_NO_PIN"))
    {
        Info<< "fgmFluid: RGP_NO_PIN set -- host buffer pinning disabled"
            << nl << endl;
        gpuPinned_ = true;
        return;
    }

    // 동적 메시(AMR/이동): 필드/버퍼 재할당 시 page-lock 등록이 해제
    // 없이 무효화(dangling registration)될 수 있어 pinning을 끈다 —
    // 전송은 pageable로 동작(정합 무관, 성능 ~2%)
    if (mesh.dynamic())
    {
        Info<< "fgmFluid: dynamic mesh -- host buffer pinning disabled"
            << nl << endl;
        gpuPinned_ = true;   // 재시도 방지
        return;
    }

    // 병렬: fvMeshDistributor(부하 재분배)는 mesh.dynamic()에 잡히지
    // 않으면서 필드 스토리지를 재할당한다 — dangling registration을
    // 원천 차단 (병렬 GPU는 thermo/manifold 폴백 경로뿐이라 손실 미미)
    if (Pstream::parRun())
    {
        Info<< "fgmFluid: parallel run -- host buffer pinning disabled"
            << nl << endl;
        gpuPinned_ = true;
        return;
    }

    // grow-only 스테이징 버퍼 (첫 스텝 후 최종 크기)
    size_t pinnedBytes = 0;
    auto pinList = [&](List<double>& l)
    {
        if (l.size() > 0)
        {
            if (rgpPinHost(l.begin(), l.size()*sizeof(double)) == 0)
            {
                pinnedBytes += l.size()*sizeof(double);
            }
        }
    };
    pinList(gpuFgmOut_);
    pinList(gpuP_); pinList(gpuT_); pinList(gpuY_);
    pinList(gpuRho_); pinList(gpuMu_); pinList(gpuKappa_);
    pinList(gpuCp_); pinList(gpuCv_); pinList(gpuPsi_);
    pinList(gpuUBuf_);
    pinList(gpuPEqnFlux_);
    pinList(gpuZCBuf_);
    pinList(gpuYDietBuf_);

    // ⚠️ OF 필드(primitiveField)는 등록하지 않는다: rho_=thermo.rho(),
    // Z_=max(min(Z_,1),0) 류의 tmp-대입이 스토리지를 transfer(교체)해
    // 등록이 즉시 dangling이 되고, 이후 glibc가 stale 등록 범위 안쪽을
    // 재활용해 새 할당을 내주는 순간 그 포인터의 H2D가 invalid
    // argument로 깨진다 — AmgX 병용 시 결정적으로 재현됐던 크래시의
    // 근본 원인(2026-07-11 규명; AmgX는 자체 호스트 할당으로 malloc
    // 재활용 배치를 바꿔 충돌을 표면화했을 뿐, pcg에서도 잠복했다).
    // 위의 List<double> 스테이징 버퍼들은 transfer 대상이 아니라 안전.

    gpuPinned_ = true;
    Info<< "fgmFluid: GPU host buffers page-locked (pinned) -- "
        << label(pinnedBytes/(1024*1024)) << " MiB total" << nl << endl;
}


// ************************************************************************* //
