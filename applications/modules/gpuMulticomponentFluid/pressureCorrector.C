/*---------------------------------------------------------------------------*\
  gpuMulticomponentFluid — GPU pressure corrector: stock isothermalFluid::
  correctPressure의 비-transonic/비-consistent 경로를 rgpPEqnSolve로.
      pEqn: fvc::ddt(rho) + psi*correction(fvm::ddt(p))
          + fvc::div(phiHbyA) - fvm::laplacian(rhorAUf, p)
  매핑: psis←psi, rAUf←rhorAUf(선형 보간), phiInt←phiHbyA(질량 플럭스),
  srcCellExtra ← -(fvc::ddt(rho) - psi*fvc::ddt(p)) [명시 잔여],
  플럭스 재구성: phi = phiHbyA + pEqn.flux().
  rAU/HbyA는 rgpUEqnAH(디바이스 상주 UEqn 행렬)에서 회수.
  MRF: makeRelative(phiHbyA) + constrainPressure. 부력: p_rgh 정식화
  (correctBuoyantPressure 1:1) — phig 항 + netForce/p 재구성 후처리.
  v1: 정적 메시·transient·비-transonic·비-consistent·fvModels(p)
  미지원(Fatal).
\*---------------------------------------------------------------------------*/

#include "gpuMulticomponentFluid.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvcFlux.H"
#include "fvcGrad.H"
#include "fvcSnGrad.H"
#include "snGradScheme.H"
#include "fvcReconstruct.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "solutionControl.H"
#include "localEulerDdtScheme.H"
#include "processorFvPatchFields.H"
#include "gpu/rgpPEqnTypes.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::gpuMulticomponentFluid::pressureCorrector()
{
    if (!gpuPEqn_)
    {
        multicomponentFluid::pressureCorrector();
        return;
    }

    // stock isothermalFluid::pressureCorrector와 동일한 PISO 보정자
    // 루프 — pimple.correct()가 보정자 진행과 finalIteration 플래그를
    // 관리한다 (누락 시 보정자 1회 + 솔버딕이 pFinal로 고정되는 버그)
    while (pimple.correct())
    {
        correctPressureGpu();
    }
}


void Foam::solvers::gpuMulticomponentFluid::correctPressureGpu()
{
    volScalarField& rho(rho_);
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    if (pimple.transonic() || pimple.consistent())
    {
        FatalErrorInFunction
            << "gpuPEqn (v1) does not support transonic/consistent"
            << exit(FatalError);
    }
    if (mesh.schemes().steady() || mesh.moving())
    {
        FatalErrorInFunction
            << "gpuPEqn (v1) supports static-mesh transient runs only"
            << exit(FatalError);
    }
    // 부력: p_rgh 정식화 (stock correctBuoyantPressure의 비-transonic/
    // 비-consistent/transient 경로) — 솔브 변수만 p_rgh로 바뀌고 조립
    // 매핑은 동일, phig 항과 netForce/p 재구성 후처리가 추가된다
    const bool buoyant = buoyancy.valid();
    volScalarField& solveP = buoyant ? p_rgh_ : p_;

    checkGpuGuards(solveP);

    const volScalarField& psi = thermo.psi();
    rho = thermo.rho();
    rho.relax();

    const volScalarField psip0(psi*p);
    const surfaceScalarField rhof(fvc::interpolate(rho));

    // ── rAU/HbyA: 디바이스 상주 UEqn 행렬에서 회수 ──────────────────
    const label nc = mesh.nCells();
    double* U3 = gpuUBuf_.begin();
    double* rAUh = U3 + 6*nc;
    double* H3 = U3 + 9*nc;
    {
        const vectorField& Uc = U.primitiveField();
        for (label i = 0; i < nc; i++)
        {
            for (label k = 0; k < 3; k++)
            {
                U3[k*nc + i] = Uc[i][k];
            }
        }
    }
    // 병렬: fvMatrix::H()의 coupled addBoundarySource(H += bC*U_nbr)
    List<double> UNbr3;
    if (Pstream::parRun())
    {
        label nPar = 0;
        forAll(U.boundaryField(), patchi)
        {
            if (U.boundaryField()[patchi].coupled())
            {
                nPar += U.boundaryField()[patchi].size();
            }
        }
        if (nPar > 0)
        {
            UNbr3.setSize(3*nPar);
            label offp = 0;
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp =
                    U.boundaryField()[patchi];
                if (!pp.coupled()) continue;
                forAll(pp, f)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        UNbr3[k*nPar + offp + f] = pp[f][k];
                    }
                }
                offp += pp.size();
            }
        }
    }

    if (rgpUEqnAH
        (
            U3, UNbr3.size() ? UNbr3.begin() : nullptr, rAUh, H3
        ))
    {
        FatalErrorInFunction
            << "rgpUEqnAH: " << rgpPEqnLastError() << exit(FatalError);
    }

    tmp<volScalarField> trAU = volScalarField::New
    (
        "rAU", mesh,
        dimensionedScalar(dimVolume*dimTime/dimMass, 0),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    {
        scalarField& r = trAU.ref().primitiveFieldRef();
        for (label i = 0; i < nc; i++) { r[i] = rAUh[i]; }
    }
    trAU.ref().correctBoundaryConditions();
    const volScalarField& rAU = trAU();

    tmp<volVectorField> tH = volVectorField::New
    (
        "H(U)", mesh,
        dimensionedVector(dimDensity*dimVelocity/dimTime, Zero),
        extrapolatedCalculatedFvPatchScalarField::typeName
    );
    {
        vectorField& h = tH.ref().primitiveFieldRef();
        for (label i = 0; i < nc; i++)
        {
            for (label k = 0; k < 3; k++)
            {
                h[i][k] =
                    (mesh.solutionD()[k] == 1) ? H3[k*nc + i] : 0.0;
            }
        }
    }
    tH.ref().correctBoundaryConditions();

    volVectorField HbyA(constrainHbyA(rAU*tH, U, solveP));

    const surfaceScalarField rhorAUf
    (
        "rhorAUf", fvc::interpolate(rho*rAU)
    );

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        rhof*fvc::flux(HbyA)
      + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
    );

    MRF.makeRelative(rhof, phiHbyA);

    // 부력: phig 항 — ghGradRhof는 netForce 재구성에도 쓰인다
    tmp<surfaceScalarField> tGhGradRhof;
    if (buoyant)
    {
        tGhGradRhof =
            (-buoyancy->ghf*fvc::snGrad(rho)*mesh.magSf()).ptr();
        phiHbyA += rhorAUf*tGhGradRhof();
    }

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(solveP, rho, U, phiHbyA, rhorAUf, MRF);

    // CPU 규약: fvMatrix 생성자의 BC updateCoeffs() 상응
    solveP.boundaryFieldRef().updateCoeffs();

    // ── pEqn 조립+솔브 (rgpPEqnSolve — 매핑은 파일 헤더 참조) ────────
    label nbf = 0;
    forAll(solveP.boundaryField(), patchi)
    {
        nbf += solveP.boundaryField()[patchi].size();
    }

    gpuPBuf_.setSize(3*nbf);
    double* bDiagA = gpuPBuf_.begin();
    double* bSrcA = bDiagA + nbf;
    double* phiBA = bSrcA + nbf;

    // 비직교: coupled 경계 pdc·명시 보정 소스·faceFluxCorrection은
    // solveP의 snGrad 스킴으로 (fgmFluid 패턴 1:1; 직교 스킴이면
    // deltaCoeffs와 동일 — 비트-불변)
    tmp<fv::snGradScheme<scalar>> tsnPE
    (
        fv::snGradScheme<scalar>::New
        (
            mesh, mesh.schemes().snGrad(solveP.name())
        )
    );
    const surfaceScalarField dcsPEqn(tsnPE().deltaCoeffs(solveP));
    {
        label off = 0;
        label offPar = 0;
        forAll(solveP.boundaryField(), patchi)
        {
            const fvPatchScalarField& pp =
                solveP.boundaryField()[patchi];
            const label np = pp.size();
            if (np == 0) continue;

            const scalarField pGamma
            (
                rhorAUf.boundaryField()[patchi]
               *mesh.magSf().boundaryField()[patchi]
            );
            const scalarField& phb = phiHbyA.boundaryField()[patchi];

            if (pp.coupled())
            {
                // processor: rhorAUf/phiHbyA coupled boundaryField는
                // fvc 보간이 이미 면값. iC → diag, bC → 인터페이스.
                const scalarField pdc
                (
                    dcsPEqn.boundaryField()[patchi]
                );
                const scalarField gic(pp.gradientInternalCoeffs(pdc));
                const scalarField gbc(pp.gradientBoundaryCoeffs(pdc));

                for (label k = 0; k < np; k++)
                {
                    bDiagA[off + k] = -pGamma[k]*gic[k];
                    bSrcA[off + k] = 0;
                    phiBA[off + k] = phb[k];
                    gpuParB_[offPar + k] = pGamma[k]*gbc[k];
                }
                offPar += np;
            }
            else
            {
                const scalarField gic(pp.gradientInternalCoeffs());
                const scalarField gbc(pp.gradientBoundaryCoeffs());

                for (label k = 0; k < np; k++)
                {
                    bDiagA[off + k] = -pGamma[k]*gic[k];
                    bSrcA[off + k] = pGamma[k]*gbc[k];
                    phiBA[off + k] = phb[k];
                }
            }
            off += np;
        }

        if (offPar > 0 && rgpPEqnParCoeffs(gpuParB_.begin()))
        {
            FatalErrorInFunction
                << "rgpPEqnParCoeffs: " << rgpPEqnLastError()
                << exit(FatalError);
        }
    }

    const volScalarField ddtRho(fvc::ddt(rho));

    // 명시 잔여: fvc::ddt(rho) - psi*fvc::ddt(solveP) → b -= (·)*V.
    // stock pDDtEqn 1:1 — 보정자 루프 밖에서 1회 동결:
    // correction(fvm::ddt(p))의 명시부 fvc::ddt(p)는 사전-솔브 p
    // 기준이며 패스마다 재계산하지 않는다 (재계산 시 방정식이 패스마다
    // 이동해 수렴 파괴 — 왜곡 메시 gmc 벤치 실측: 패스2 초기잔차
    // 0.26 vs CPU 0.01, 2스텝 T 9.6K 이탈)
    const volScalarField ddtP(fvc::ddt(solveP));
    scalarField srcExtra0(nc);
    {
        const scalarField& psic = psi.primitiveField();
        const scalarField& dr = ddtRho.primitiveField();
        const scalarField& dp = ddtP.primitiveField();
        forAll(srcExtra0, i)
        {
            srcExtra0[i] = -(dr[i] - psic[i]*dp[i]);
        }
    }

    // 검증 모드용 동결 pDDt 행렬 (stock pDDtEqn 상응)
    autoPtr<fvScalarMatrix> pDDtChk;
    if (gpuCheck_)
    {
        pDDtChk.set
        (
            new fvScalarMatrix
            (
                fvc::ddt(rho)
              + psi*correction(fvm::ddt(solveP))
              + fvc::div(phiHbyA)
            )
        );
    }

    const bool needRef =
        solveP.needReference() && pressureReference.refCell() >= 0;
    const label refCell = needRef ? pressureReference.refCell() : 0;
    const scalar refValue = needRef ? pressureReference.refValue() : 0;

    const bool LTS = fv::localEulerDdt::enabled(mesh);
    const dictionary& sd = mesh.solution().solverDict
    (
        solveP.select
        (
            !mesh.schemes().steady()
         && solutionControl::finalIteration(mesh)
        )
    );
    const scalar tol = sd.lookup<scalar>("tolerance");
    const scalar rtol = sd.lookupOrDefault<scalar>("relTol", 0);
    const label maxIter = sd.lookupOrDefault<label>("maxIter", 1000);

    gpuPFlux_.setSize(mesh.owner().size());

    double res0 = 0, resF = 0;
    int nIter = 0;

    // ── 검증 모드: CPU pEqn fvMatrix 계수와 대조 (fgmFluid 1:1;
    //    serial-only는 생성자 가드). 보정자 루프 밖(사전)과 루프 내
    //    각 패스에서 호출 — 패스별 재조립 검증 ──
    auto pEqnCheck = [&]()
    {
        // stock 1:1: 동결 pDDt + 패스별 laplacian 재조립
        fvScalarMatrix pEqnChk
        (
            pDDtChk() - fvm::laplacian(rhorAUf, solveP)
        );
        if (needRef)
        {
            pEqnChk.setReference
            (
                pressureReference.refCell(),
                pressureReference.refValue()
            );
        }

        scalarField diagC(pEqnChk.diag());
        scalarField srcC(pEqnChk.source());
        forAll(solveP.boundaryField(), patchi)
        {
            const labelUList& fc = mesh.boundary()[patchi].faceCells();
            const scalarField& iC = pEqnChk.internalCoeffs()[patchi];
            const scalarField& bC = pEqnChk.boundaryCoeffs()[patchi];
            forAll(fc, k)
            {
                diagC[fc[k]] += iC[k];
                srcC[fc[k]] += bC[k];
            }
        }

        scalarField srcChk(srcExtra0);
        if (gpuNonOrtho_)
        {
            srcChk += fvc::div
            (
                rhorAUf*mesh.magSf()*tsnPE().correction(solveP)
            )().primitiveField();
        }

        List<double> dG(nc), uG(mesh.owner().size()), bG(nc);
        if (rgpPEqnAssembleDump
            (
                LTS ? 1.0 : 1.0/runTime.deltaTValue(),
                LTS
              ? fv::localEulerDdt::localRDeltaT(mesh)
                   .primitiveField().begin()
              : nullptr,
                srcChk.begin(),
                rhorAUf.primitiveField().begin(),
                psi.primitiveField().begin(),
                solveP.oldTime().primitiveField().begin(),
                phiHbyA.primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                dG.begin(), uG.begin(), bG.begin()
            ))
        {
            FatalErrorInFunction
                << "rgpPEqnAssembleDump: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        scalar dMax = 0, uMax = 0, bMax = 0;
        forAll(diagC, i)
        {
            dMax = max(dMax, mag(dG[i] - diagC[i])
                /max(mag(diagC[i]), small));
            bMax = max(bMax, mag(bG[i] - srcC[i])
                /max(mag(srcC[i]), small));
        }
        const scalarField& uC = pEqnChk.upper();
        forAll(uC, f)
        {
            uMax = max(uMax, mag(uG[f] - uC[f])
                /max(mag(uC[f]), small));
        }
        Info<< "gpuPEqnCheck: maxRel diag = " << dMax
            << ", upper = " << uMax << ", source = " << bMax << endl;
    };

    // ── 비직교 보정자 루프 (CPU while(correctNonOrthogonal()) 1:1):
    //    매 패스 최신 solveP로 명시 잔여(ddtP)·보정 소스·
    //    faceFluxCorrection을 재계산해 재조립·재솔브. 직교(0회)
    //    케이스는 1회 실행 = 기존과 동일 ──
    tmp<surfaceScalarField> tPCorrFace;
    while (pimple.correctNonOrthogonal())
    {
    if (gpuCheck_) { pEqnCheck(); }
    scalarField srcExtra(srcExtra0);

    // 비직교 명시 보정 (gaussLaplacianScheme 1:1): M = pDDt − L 에서
    // L.source = −V·div(ΓmagSf·corr) → M 소스 += div(·) (per-vol).
    // fluxRequired라 faceFluxCorrection = ΓmagSf·corr도 보존
    // (플럭스 재구성에서 −).
    if (gpuNonOrtho_)
    {
        tPCorrFace = surfaceScalarField::New
        (
            "pFaceFluxCorr",
            rhorAUf*mesh.magSf()*tsnPE().correction(solveP)
        );
        srcExtra += fvc::div(tPCorrFace())().primitiveField();
    }

    if (rgpPEqnSolve
        (
            LTS ? 1.0 : 1.0/runTime.deltaTValue(),
            LTS
          ? fv::localEulerDdt::localRDeltaT(mesh)
               .primitiveField().begin()
          : nullptr,
            srcExtra.begin(),
            rhorAUf.primitiveField().begin(),
            psi.primitiveField().begin(),
            solveP.oldTime().primitiveField().begin(),
            solveP.primitiveField().begin(),
            phiHbyA.primitiveField().begin(),
            phiBA, bDiagA, bSrcA,
            needRef, refCell, refValue,
            tol, rtol, maxIter,
            solveP.primitiveFieldRef().begin(),
            gpuPFlux_.begin(),
            &res0, &resF, &nIter
        ))
    {
        FatalErrorInFunction
            << "rgpPEqnSolve: " << rgpPEqnLastError() << exit(FatalError);
    }

    Info<< "rgpPCG:  Solving for " << solveP.name()
        << ", Initial residual = " << res0
        << ", Final residual = " << resF
        << ", No Iterations " << nIter << endl;

    solveP.correctBoundaryConditions();
    }   // while correctNonOrthogonal

    // ── 질량 플럭스 재구성: phi = phiHbyA + pEqn.flux() ─────────────
    //    비직교: pEqn.flux()에 faceFluxCorrection(−ΓmagSf·corr)이
    //    포함되므로(fvMatrix::flux 1:1) 내부·경계 모두 차감
    {
        scalarField& phic = phi.primitiveFieldRef();
        const scalarField& ph = phiHbyA.primitiveField();
        if (tPCorrFace.valid())
        {
            const scalarField& cf = tPCorrFace().primitiveField();
            forAll(phic, f)
            {
                phic[f] = ph[f] + gpuPFlux_[f] - cf[f];
            }
        }
        else
        {
            forAll(phic, f) { phic[f] = ph[f] + gpuPFlux_[f]; }
        }

        label off = 0;
        label offPar = 0;
        forAll(solveP.boundaryField(), patchi)
        {
            const fvPatchScalarField& pp =
                solveP.boundaryField()[patchi];
            const label np = pp.size();
            if (np == 0) continue;

            const scalarField pi(pp.patchInternalField());
            const scalarField& phb = phiHbyA.boundaryField()[patchi];
            scalarField& phibf = phi.boundaryFieldRef()[patchi];

            const scalarField corrB
            (
                tPCorrFace.valid()
              ? scalarField(tPCorrFace().boundaryField()[patchi])
              : scalarField(np, 0.0)
            );

            if (pp.coupled())
            {
                for (label k = 0; k < np; k++)
                {
                    const scalar fluxB =
                        bDiagA[off + k]*pi[k]
                      - gpuParB_[offPar + k]*pp[k]
                      - corrB[k];
                    phibf[k] = phb[k] + fluxB;
                }
                offPar += np;
            }
            else
            {
                for (label k = 0; k < np; k++)
                {
                    const scalar fluxB =
                        bDiagA[off + k]*pi[k] - bSrcA[off + k]
                      - corrB[k];
                    phibf[k] = phb[k] + fluxB;
                }
            }
            off += np;
        }
    }

    // ── stock 후처리 1:1 ────────────────────────────────────────────
    if (buoyant)
    {
        // 순 압력-부력 힘 재구성/완화 → U 보정 (correctBuoyantPressure
        // 1:1; pEqn.flux() = phi - phiHbyA, rhorAAtUf = rhorAUf)
        const surfaceScalarField pEqnFlux(phi - phiHbyA);
        netForce.ref().relax
        (
            fvc::reconstruct(tGhGradRhof() + pEqnFlux/rhorAUf),
            solveP.relaxationFactor()
        );

        U = HbyA + rAU*netForce();
        U.correctBoundaryConditions();
        fvConstraints().constrain(U);
        K = 0.5*magSqr(U);

        p = solveP + rho*buoyancy->gh + buoyancy->pRef;
        const bool constrained = fvConstraints().constrain(p);

        thermo_.correctRho(psi*p - psip0);
        if (constrained)
        {
            rho = thermo.rho();
        }
        // isothermalFluid::correctDensity (private) 인라인 — rho 수송
        {
            fvScalarMatrix rhoEqn
            (
                fvm::ddt(rho) + fvc::div(phi)
             ==
                fvModels().source(rho)
            );
            fvConstraints().constrain(rhoEqn);
            rhoEqn.solve();
            fvConstraints().constrain(rho);
        }

        fluidSolver::continuityErrors(rho, thermo.rho(), phi);

        // rho 수송 후 p 재구성 (upstream 순서)
        p = solveP + rho*buoyancy->gh + buoyancy->pRef;
        p.relax();
    }
    else
    {
    const bool constrained = fvConstraints().constrain(p);

    thermo_.correctRho(psi*p - psip0);
    if (constrained)
    {
        rho = thermo.rho();
    }
    // isothermalFluid::correctDensity (private) 인라인 — rho 수송
    {
        fvScalarMatrix rhoEqn
        (
            fvm::ddt(rho) + fvc::div(phi)
         ==
            fvModels().source(rho)
        );
        fvConstraints().constrain(rhoEqn);
        rhoEqn.solve();
        fvConstraints().constrain(rho);
    }

    // isothermalFluid::continuityErrors (private) — fluidSolver판 직접
    fluidSolver::continuityErrors(rho, thermo.rho(), phi);

    p.relax();

    U = HbyA - rAU*fvc::grad(p);
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);
    K = 0.5*magSqr(U);
    }

    if (pimple.simpleRho())
    {
        rho = thermo.rho();
        rho.relax();
    }

    if (thermo.dpdt())
    {
        dpdt = fvc::ddt(p);
    }
}


// ************************************************************************* //
