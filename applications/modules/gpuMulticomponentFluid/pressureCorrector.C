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
#include "fvcFlux.H"
#include "fvcGrad.H"
#include "fvcSnGrad.H"
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
                    mesh.deltaCoeffs().boundaryField()[patchi]
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

    // 명시 잔여: fvc::ddt(rho) - psi*fvc::ddt(solveP) → b -= (·)*V
    const volScalarField ddtRho(fvc::ddt(rho));
    const volScalarField ddtP(fvc::ddt(solveP));
    scalarField srcExtra(nc);
    {
        const scalarField& psic = psi.primitiveField();
        const scalarField& dr = ddtRho.primitiveField();
        const scalarField& dp = ddtP.primitiveField();
        forAll(srcExtra, i)
        {
            srcExtra[i] = -(dr[i] - psic[i]*dp[i]);
        }
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

    // ── 질량 플럭스 재구성: phi = phiHbyA + pEqn.flux() ─────────────
    {
        scalarField& phic = phi.primitiveFieldRef();
        const scalarField& ph = phiHbyA.primitiveField();
        forAll(phic, f) { phic[f] = ph[f] + gpuPFlux_[f]; }

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

            if (pp.coupled())
            {
                for (label k = 0; k < np; k++)
                {
                    const scalar fluxB =
                        bDiagA[off + k]*pi[k]
                      - gpuParB_[offPar + k]*pp[k];
                    phibf[k] = phb[k] + fluxB;
                }
                offPar += np;
            }
            else
            {
                for (label k = 0; k < np; k++)
                {
                    const scalar fluxB =
                        bDiagA[off + k]*pi[k] - bSrcA[off + k];
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
