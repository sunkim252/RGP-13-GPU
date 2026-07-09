/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Description
    fgmFluid pressure corrector -- overrides isothermalFluid::pressureCorrector
    so the pressure equation can be replaced by a pressure-equilibrium-
    preserving (pressure-evolution) formulation. STEP 1 (this file, first
    build): correctPressurePEP() is a FAITHFUL COPY of the base
    isothermalFluid::correctPressure (non-buoyant, SIMPLErho path), so the 1-D
    advecting-interface benchmark must reproduce the baseline spurious pressure
    spike unchanged -- confirming the override mechanism is isolated before the
    PEP modification is introduced. Reference for the PEP target:
    Terashima & Koshi, J. Comput. Phys. 231 (2012) 6907; Kai/Kurose PEQSI,
    Phys. Fluids 36 (2024) 116104.

\*---------------------------------------------------------------------------*/

#include "fgmFluid.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include "fvcMeshPhi.H"
#include "fvcFlux.H"
#include "fvcDdt.H"
#include "fvcGrad.H"
#include "fvcSnGrad.H"
#include "fvcReconstruct.H"
#include "fvcVolumeIntegrate.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvcLaplacian.H"
#include "fvcAverage.H"
#include "zeroGradientFvPatchFields.H"
#include "solutionControl.H"
#include "gpu/rgpPEqnTypes.H"

#include <chrono>
#include <dlfcn.h>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::armGpuPEqnMesh()
{
    if (gpuPEqnArmed_) return;

    const label nc = mesh.nCells();
    const label nif = mesh.owner().size();

    label nbf = 0;
    forAll(p_.boundaryField(), patchi)
    {
        if (p_.boundaryField()[patchi].coupled())
        {
            FatalErrorInFunction
                << "gpuPEqn/gpuZC (v1) do not support coupled patches"
                << exit(FatalError);
        }
        nbf += p_.boundaryField()[patchi].size();
    }

    List<int> own(nif), nei(nif);
    forAll(mesh.owner(), f)
    {
        own[f] = mesh.owner()[f];
        nei[f] = mesh.neighbour()[f];
    }
    List<double> gg(nif);
    const scalarField& magSf = mesh.magSf().primitiveField();
    const scalarField& dc = mesh.deltaCoeffs().primitiveField();
    forAll(gg, f) { gg[f] = magSf[f]*dc[f]; }

    List<int> bfc(nbf);
    label off = 0;
    forAll(p_.boundaryField(), patchi)
    {
        const labelUList& fc = mesh.boundary()[patchi].faceCells();
        forAll(fc, k) { bfc[off + k] = fc[k]; }
        off += fc.size();
    }

    const scalarField Vc(mesh.V());
    const int rc = rgpPEqnMeshUpload
    (
        nc, nif, own.begin(), nei.begin(), gg.begin(),
        Vc.begin(), nbf, bfc.begin()
    );
    if (rc)
    {
        FatalErrorInFunction
            << "rgpPEqnMeshUpload: " << rgpPEqnLastError()
            << exit(FatalError);
    }
    if (gpuPEqnSolver_ == "amgx")
    {
        int nnz = 0;
        if (rgpPEqnCsrPrepare(&nnz))
        {
            FatalErrorInFunction
                << "rgpPEqnCsrPrepare: " << rgpPEqnLastError()
                << exit(FatalError);
        }
        gpuPEqnNnz_ = nnz;
    }

    gpuPEqnArmed_ = true;
    Info<< "fgmFluid: GPU pEqn mesh armed -- " << nc << " cells, "
        << nif << " internal faces, " << nbf
        << " boundary faces (pEqn solver: " << gpuPEqnSolver_ << ")"
        << nl << endl;
}


void Foam::solvers::fgmFluid::correctPressurePEP()
{
    volScalarField& rho(rho_);
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    // --- RANK 1: refresh the real-fluid rho, psi at the CURRENT pressure each
    // pressure corrector (not once per outer). A stale SRK compressibility
    // psi = (drho/dp)_T at the stiff cold-LOX injector cells lets the pressure
    // corrector run against a near-singular diagonal and generate the injector
    // pressure spike -- the pressure-velocity / EOS-stiffness ill-conditioning
    // that neither PEP nor LAD addresses (Ma, Lv & Ihme, J. Comput. Phys. 340
    // (2017) 330). Property-update-per-corrector is the realFluidFoam recipe
    // (Nguyen & Yoo, Comput. Phys. Commun. 312 (2025) 109600). thermo_.correct()
    // inverts the manifold-seeded he to T and refreshes rho/psi/mu at (p,T,Y).
    // (b) he->T-drift-free variant: re-seed he from the manifold at the CURRENT
    // pressure (updateManifold) BEFORE correct(), so the he->T inversion returns
    // T_table exactly (no drift) while rho/psi still refresh at the new p. This
    // fixes the RANK-1 side effect where bare thermo_.correct() inverted a stale
    // he against the new p and drifted T, moving the spike into the chamber.
    std::chrono::steady_clock::time_point tPh, tTot;
    if (thermoTimings_) { tPh = tTot = std::chrono::steady_clock::now(); }

    updateManifold();

    if (thermoTimings_)
    {
        Info<< "pCorr manifold = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tPh).count()
            << " s" << endl;
        tPh = std::chrono::steady_clock::now();
    }

    // 예측자(thermophysicalPredictor)와 동일한 GPU 물성 refresh 대체.
    // updateManifold가 직전에 he를 T_table로 재시드하므로 CPU 경로의
    // he->T 역산은 T_table을 되돌려줄 뿐 — GPU 경로(역산 생략, (p,T,Y)
    // 일괄 물성)와 의미가 같다. 보정자당 1회 x nCorr회 호출되는 CPU
    // calculate() 루프가 제거된다.
    if (gpuThermo_)
    {
        gpuThermoCorrect();
    }
    else
    {
        thermo_.correct();
    }

    if (thermoTimings_)
    {
        Info<< "pCorr thermo correct = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tPh).count()
            << " s" << endl;
        tPh = std::chrono::steady_clock::now();
    }

    // Per-corrector transported-density re-sync (modified-PIMPLE, cf.
    // realFluidFoam/Jarczyk-Pfitzner). The transported rho_ is otherwise
    // advanced only by the correctRho(psi*dp) increments; at a flame-zone
    // pressure spike those increments are huge and (with the pMinPa clamp)
    // inconsistent, so rho_ drifts from the EOS state and accumulates error
    // -- observed: rho_ down to -1175 kg/m3 at the ox tangential holes while
    // the EOS density stayed positive. Snapping rho_ to the just-corrected
    // thermo state each corrector removes the drift (rho.oldTime() is
    // untouched, so the ddt history stays consistent).
    rho_ = thermo.rho();
    rho_.correctBoundaryConditions();

    const volScalarField& psi = thermo.psi();
    rho = thermo.rho();
    rho.relax();

    fvVectorMatrix& UEqn = tUEqn.ref();

    // Thermodynamic density needs to be updated by psi*d(p) after the
    // pressure solution
    const volScalarField psip0(psi*p);

    const surfaceScalarField rhof(fvc::interpolate(rho));

    const volScalarField rAU("rAU", 1.0/UEqn.A());
    const surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho*rAU));

    tmp<volScalarField> rAtU
    (
        pimple.consistent()
      ? volScalarField::New("rAtU", 1.0/(1.0/rAU - UEqn.H1()))
      : tmp<volScalarField>(nullptr)
    );

    tmp<surfaceScalarField> rhorAtUf
    (
        pimple.consistent()
      ? surfaceScalarField::New("rhoRAtUf", fvc::interpolate(rho*rAtU()))
      : tmp<surfaceScalarField>(nullptr)
    );

    const volScalarField& rAAtU = pimple.consistent() ? rAtU() : rAU;
    const surfaceScalarField& rhorAAtUf =
        pimple.consistent() ? rhorAtUf() : rhorAUf;

    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));

    if (pimple.nCorrPiso() <= 1)
    {
        tUEqn.clear();
    }

    // --- PEP (pressure-evolution) pressure equation -------------------------
    // Replace the base continuity pEqn's fvc::ddt(rho) -- whose Eulerian
    // thermal density change radiates the spurious contact pressure spike --
    // with the pressure-evolution form  psis*dp/dt + div(u) = 0, where psis =
    // 1/(rho c^2) is the isentropic compressibility and div(u) is the
    // VOLUMETRIC velocity divergence (NOT the mass-flux divergence). For a
    // uniform-p, uniform-u contact div(u) = 0 and dp/dt = 0 by construction, so
    // the density jump advects without radiating pressure (Terashima & Koshi,
    // J. Comput. Phys. 231 (2012) 6907; Kai/Kurose PEQSI, Phys. Fluids 36
    // (2024) 116104). ATTEMPT 1: ideal-gas isentropic compressibility
    // 1/(gamma p) -- validates the STRUCTURE (spike removal); refined to the
    // real SRK sound speed next. constrainPressure / MRF-mass / consistent
    // branches dropped for this first cyclic-benchmark attempt.
    // C1a: HARMONIC-mean face mobility (Rhie-Chow across the stiff density
    // jump). rAU = 1/A() ~ dt/rho jumps ~60x cell-to-cell at the recess-tip
    // LOx/gas interface; the arithmetic fvc::interpolate(rAU) over-weights the
    // low-density (large rAU) side and over-stiffens the pEqn face coefficient
    // by ~O(30), so a small dp drives a huge face flux -> the spurious pressure
    // spike with |U| pinned at the limiter (co-located max(p)/|U|-cap). The
    // harmonic (series-resistance-correct) mean is the density-jump-robust
    // choice for a face mobility (Ferziger & Peric; Rhie & Chow, AIAA J. 21
    // (1983) 1525). 1/rAU = A() > 0 (momentum diagonal) so no floor needed.
    const surfaceScalarField rAUf("rAUf", 1.0/fvc::interpolate(1.0/rAU));
    // ATTEMPT 2: real SRK isothermal compressibility psis = (drho/dp)_T/rho =
    // kappa_T (thermo.psi() now returns the real (drho/dp)_T -- see SRKGasI.H).
    // This supplies the true dense-fluid stiffness that the ideal-gas 1/(gamma
    // p) lacked (~100x too soft for liquid LOX).
    // C3: pressure-based double-flux analogue -- bound the cell-to-cell
    // CONTRAST of the pEqn diagonal compressibility psis = kappa_T across the
    // stiff-LOX / soft-gas interface. psis jumps enormously there (dense LOX
    // stiff, warm gas soft); a large diagonal-ratio next to the (now harmonic)
    // off-diagonal still lets a residual dp blow up. Floor each cell's psis at
    // (face-averaged psis)/psisCapRatio so the interface diagonal ratio is
    // capped -- the honest pressure-based cousin of freezing a local effective
    // gamma* (Ma, Lv & Ihme, JCP 340 (2017) 330). psisCapRatio default GREAT
    // (off); read each step. Conservative at convergence (psis*ddt(p) -> 0).
    const scalar psisCapRatio
    (
        pimple.dict().lookupOrDefault<scalar>("psisCapRatio", GREAT)
    );
    // psisIsentropic: use the ISENTROPIC compressibility kappa_s = kappa_T/gamma
    // = 1/(rho c^2) instead of the isothermal kappa_T = thermo.psi()/rho. For a
    // pressure-EVOLUTION (PEP) equation the acoustically-correct diagonal is the
    // isentropic one; the isothermal psi() over-stiffens the dense cold LOX
    // (Terashima-Koshi; design-agent refinement). Default off.
    const Switch psisIsentropic
    (
        pimple.dict().lookupOrDefault<Switch>("psisIsentropic", false)
    );
    volScalarField psis
    (
        "psis",
        psisIsentropic
      ? volScalarField(thermo.psi()/(rho*thermo.gamma()))
      : volScalarField(thermo.psi()/rho)
    );
    if (psisCapRatio < GREAT)
    {
        const volScalarField psisSm(fvc::average(fvc::interpolate(psis)));
        psis = max(psis, psisSm/psisCapRatio);
        psis.correctBoundaryConditions();
    }

    // Volumetric predicted face flux with the RHO-CONSISTENT transient
    // Rhie-Chow correction (C2 proper). The base compressible form is the MASS
    // flux  rhof*fvc::flux(HbyA) + rhorAUf*fvc::ddtCorr(rho,U,phi,rhoUf)
    // (isothermalFluid::correctPressure); the volumetric PEP is its /rhof. The
    // rho-aware ddtCorr(rho,U,phi,rhoUf) (rhoUf is null on a static mesh, where
    // the overload falls back to the rho-weighted form) keeps the transient RC
    // CONSISTENT with the momentum flux fvm::div(phi,U) across the ~60x rho jump
    // -- unlike the earlier ddtCorr(U, phi/rhof), whose arithmetic rhof in the
    // denominator injected a spurious face velocity (the recess-tip/injector
    // spike). This replaces the rcDdtScale 0 workaround, which suppressed the
    // spurious RC flux but left the injector fine cells to CHECKERBOARD (a new
    // spike re-formed there). rcDdtScale (default 1) still gates the term for
    // A/B testing. Read each step (runTimeModifiable).
    const scalar rcDdtScale
    (
        pimple.dict().lookupOrDefault<scalar>("rcDdtScale", scalar(1))
    );
    surfaceScalarField phiHbyAv
    (
        "phiHbyAv",
        fvc::flux(HbyA)
      + rcDdtScale*rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)/rhof  // rho-consistent RC
    );
    MRF.makeRelative(phiHbyAv);

    // Update the pressure BCs for flux consistency (3D: waveTransmissive
    // outlet, fixedFluxPressure walls). Volumetric-flux form: pass the
    // volumetric predicted flux and the rAUf (velocity-level) coefficient,
    // matching the pEqn's laplacian(rAUf, p).
    constrainPressure(p, rho, U, phiHbyAv, rAUf, MRF);

    // --- RANK 4: Artificial Mass Diffusivity (AMD) on DENSITY -----------------
    // Kawai, Terashima & Negishi, J. Comput. Phys. 300 (2015) 116: at a large
    // density ratio, diffusing the SCALARS/temperature (as the manifold LAD does
    // on Z,C,h) drives spurious p/u oscillations THROUGH the nonlinear EOS,
    // whereas diffusing MASS/DENSITY is the consistent choice. Smooth the steep
    // transcritical density interface at source by adding a density-gradient-
    // sensed diffusion of rho to the (volumetric) continuity, WITHOUT touching
    // the manifold scalars. Kinematic AMD coefficient [m^2/s], sized to the
    // local cell and active only where rho varies:
    //   Dr = LADrhoCoeff * V^(1/3) * |U| * s,  s = min(|grad rho| V^(1/3)/rho, 1)
    // where s is a DIMENSIONLESS density-gradient sensor CAPPED at 1: the raw
    // |grad rho| blows up at the spurious spike itself, so an uncapped Dr
    // violates the diffusive CFL (Dr ~ 164 m^2/s -> dt collapse). The cap ties
    // Dr_max = LADrhoCoeff*V^(1/3)*|U| to the convective scale, keeping the
    // diffusive step >= the convective one (Olson & Lele, JCP 246 (2013) 207).
    // The continuity then gains -(1/rho) div(Dr grad rho) [1/s], matching the
    // volumetric pressure-evolution form. LADrhoCoeff read each step (default 0).
    const scalar LADrhoCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADrhoCoeff", scalar(0))
    );
    volScalarField Dr
    (
        IOobject
        (
            "Dr_amd",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimArea/dimTime, 0),
        zeroGradientFvPatchScalarField::typeName
    );
    if (LADrhoCoeff > 0)
    {
        const scalarField V13(pow(scalarField(mesh.V()), 1.0/3.0));
        // Dimensionless density-gradient sensor in [0,1]: the relative rho
        // change across a cell, CAPPED at 1 so the spurious spike itself (huge
        // |grad rho|) cannot drive the coefficient past the diffusive-CFL limit.
        const scalarField sensor
        (
            min
            (
                mag(fvc::grad(rho))().primitiveField()*V13/rho.primitiveField(),
                scalar(1)
            )
        );
        Dr.primitiveFieldRef() =
            LADrhoCoeff*V13*mag(U)().primitiveField()*sensor;
        Dr.correctBoundaryConditions();
        Info<< "LAD-rho: Dr_amd max = " << gMax(Dr.primitiveField())
            << " m^2/s" << endl;
    }

    if (gpuPEqn_)
    {
        // --- pEqnGPU: fvMatrix 우회 — 디바이스 조립 + Jacobi-PCG + 플럭스.
        // 'Gauss linear orthogonal' laplacian + Euler ddt + 비커플드 패치
        // 전용(v1). 경계 기여만 호스트가 fvPatchField API로 정확 계산.
        if (LTS || LADrhoCoeff > 0)
        {
            FatalErrorInFunction
                << "gpuPEqn (v1) does not support LTS or LADrhoCoeff > 0"
                << exit(FatalError);
        }

        const label nc = mesh.nCells();
        const label nif = mesh.owner().size();

        label nbf = 0;
        forAll(p.boundaryField(), patchi)
        {
            if (p.boundaryField()[patchi].coupled())
            {
                FatalErrorInFunction
                    << "gpuPEqn (v1) does not support coupled patches"
                    << exit(FatalError);
            }
            nbf += p.boundaryField()[patchi].size();
        }

        // ── 1회 아밍: 정적 메시 (owner/neigh, magSf*deltaCoeffs, V) ──
        armGpuPEqnMesh();

        // ── 경계 기여: pEqn = -laplacian(rAUf, p)의 경계 계수 ─────────
        //    diag += -pGamma*gic, source += +pGamma*gbc (fvm 부호 규약)
        gpuPEqnBuf_.setSize(3*nbf);
        double* bDiagA = gpuPEqnBuf_.begin();
        double* bSrcA  = bDiagA + nbf;
        double* phiBA  = bSrcA + nbf;
        {
            label off = 0;
            forAll(p.boundaryField(), patchi)
            {
                const fvPatchScalarField& pp = p.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const scalarField pGamma
                (
                    rAUf.boundaryField()[patchi]
                   *mesh.magSf().boundaryField()[patchi]
                );
                const scalarField gic(pp.gradientInternalCoeffs());
                const scalarField gbc(pp.gradientBoundaryCoeffs());
                const scalarField& phb =
                    phiHbyAv.boundaryField()[patchi];

                for (label k = 0; k < np; k++)
                {
                    bDiagA[off + k] = -pGamma[k]*gic[k];
                    bSrcA[off + k]  =  pGamma[k]*gbc[k];
                    phiBA[off + k]  =  phb[k];
                }
                off += np;
            }
        }

        const bool needRef = p.needReference();
        const label refCell = needRef ? pressureReference.refCell() : 0;
        const scalar refValue = needRef ? pressureReference.refValue() : 0;
        const scalar dtInv = 1.0/runTime.deltaTValue();

        // ── 검증 모드: CPU fvMatrix 계수와 대조 ──────────────────────
        if (gpuPEqnCheck_)
        {
            fvScalarMatrix pDDtEqnChk
            (
                psis*fvm::ddt(p)
              + fvc::div(phiHbyAv)
            );
            fvScalarMatrix pEqnChk(pDDtEqnChk - fvm::laplacian(rAUf, p));
            pEqnChk.setReference
            (
                pressureReference.refCell(), pressureReference.refValue()
            );

            scalarField diagC(pEqnChk.diag());
            scalarField srcC(pEqnChk.source());
            forAll(p.boundaryField(), patchi)
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

            List<double> dG(nc), uG(nif), bG(nc);
            const int rc = rgpPEqnAssembleDump
            (
                dtInv, rAUf.primitiveField().begin(),
                psis.primitiveField().begin(),
                p.oldTime().primitiveField().begin(),
                phiHbyAv.primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                dG.begin(), uG.begin(), bG.begin()
            );
            if (rc)
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
        }

        // ── 솔버 컨트롤 (fvSolution p/pFinal 자동 선택) + 솔브 ────────
        // fvMatrix::solver()와 동일 규약: 최종 보정자에서 "pFinal" 선택
        const dictionary& sd = mesh.solution().solverDict
        (
            p.select
            (
                !mesh.schemes().steady()
             && solutionControl::finalIteration(mesh)
            )
        );
        const scalar tol = sd.lookup<scalar>("tolerance");
        const scalar rtol = sd.lookupOrDefault<scalar>("relTol", 0);
        const label maxIter = sd.lookupOrDefault<label>("maxIter", 1000);

        gpuPEqnFlux_.setSize(nif);
        double res0 = 0, resF = 0;
        int nIter = 0;

        std::chrono::steady_clock::time_point tS;
        if (thermoTimings_) { tS = std::chrono::steady_clock::now(); }

        if (gpuPEqnSolver_ == "amgx")
        {
            // ── AmgX 직결: 디바이스 CSR 조립 → AmgX PCG+AMG(계층 동결)
            //    → OF 규약 잔차로 수렴 확인(미달 시 톨 조여 재솔브) ──
            typedef int (*AmgxFn)
            (
                int, int, const int*, const int*,
                void*, void*, void*, double, int, int*
            );
            typedef const char* (*AmgxErrFn)(void);
            static AmgxFn amgxFn = nullptr;
            static AmgxErrFn amgxErrFn = nullptr;
            if (!amgxFn)
            {
                amgxFn = reinterpret_cast<AmgxFn>
                    (dlsym(RTLD_DEFAULT, "rgpPEqnAmgxSolve"));
                amgxErrFn = reinterpret_cast<AmgxErrFn>
                    (dlsym(RTLD_DEFAULT, "rgpPEqnAmgxErr"));
                if (!amgxFn || !amgxErrFn)
                {
                    FatalErrorInFunction
                        << "gpuPEqnSolver amgx requires libRGP13amgx.so "
                        << "in the controlDict libs list"
                        << exit(FatalError);
                }
            }

            double normFactor = 0;
            int rc = rgpPEqnAssembleCsr
            (
                dtInv,
                rAUf.primitiveField().begin(),
                psis.primitiveField().begin(),
                p.oldTime().primitiveField().begin(),
                p.primitiveField().begin(),
                phiHbyAv.primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                &normFactor, &res0
            );
            if (rc)
            {
                FatalErrorInFunction
                    << "rgpPEqnAssembleCsr: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            const label setupInterval =
                fgmTable_.lookupOrDefault<label>
                (
                    "gpuPEqnSetupInterval", 25
                );

            const scalar target = max(tol, rtol*res0);
            resF = res0;
            // AmgX per-solve 톨 주입은 라이브 솔버에 반영되지 않으므로
            // (config_add_parameters 잠복 이슈) 고정 4-iter 블록으로
            // 돌리고 OF 규약 잔차로 외부에서 수렴 판정한다. x는 블록 간
            // 연속이라 정확히 이어서 수렴한다.
            for (int round = 0; round < 25 && resF >= target; round++)
            {
                int it = 0;
                rc = amgxFn
                (
                    nc, gpuPEqnNnz_,
                    rgpPEqnCsrRowPtr(), rgpPEqnCsrColInd(),
                    rgpPEqnDevValues(), rgpPEqnDevB(), rgpPEqnDevX(),
                    0.0, setupInterval, &it
                );
                if (rc)
                {
                    FatalErrorInFunction
                        << "rgpPEqnAmgxSolve: " << amgxErrFn()
                        << exit(FatalError);
                }
                nIter += it;
                if (rgpPEqnResidual(normFactor, &resF))
                {
                    FatalErrorInFunction
                        << "rgpPEqnResidual: " << rgpPEqnLastError()
                        << exit(FatalError);
                }
            }

            if (rgpPEqnFinish
                (
                    p.primitiveFieldRef().begin(), gpuPEqnFlux_.begin()
                ))
            {
                FatalErrorInFunction
                    << "rgpPEqnFinish: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            Info<< "rgpAmgX: Solving for p, Initial residual = " << res0
                << ", Final residual = " << resF
                << ", No Iterations " << nIter << endl;
        }
        else
        {
            const int rc = rgpPEqnSolve
            (
                dtInv,
                rAUf.primitiveField().begin(),
                psis.primitiveField().begin(),
                p.oldTime().primitiveField().begin(),
                p.primitiveField().begin(),
                phiHbyAv.primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                tol, rtol, maxIter,
                p.primitiveFieldRef().begin(), gpuPEqnFlux_.begin(),
                &res0, &resF, &nIter
            );
            if (rc)
            {
                FatalErrorInFunction
                    << "rgpPEqnSolve: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            Info<< "rgpPCG:  Solving for p, Initial residual = " << res0
                << ", Final residual = " << resF
                << ", No Iterations " << nIter << endl;
        }

        if (thermoTimings_)
        {
            Info<< "pCorr pEqn.solve = "
                << std::chrono::duration<double>
                   (std::chrono::steady_clock::now() - tS).count()
                << " s" << endl;
        }

        p.correctBoundaryConditions();

        // ── 질량 플럭스 재구성: phi = rhof*(phiHbyAv + pEqn.flux()) ──
        {
            scalarField& phic = phi.primitiveFieldRef();
            const scalarField& rf = rhof.primitiveField();
            const scalarField& ph = phiHbyAv.primitiveField();
            forAll(phic, f)
            {
                phic[f] = rf[f]*(ph[f] + gpuPEqnFlux_[f]);
            }

            // 경계: fvMatrix::flux() 규약 — iC*p_internal - bC
            label off = 0;
            forAll(p.boundaryField(), patchi)
            {
                const fvPatchScalarField& pp = p.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const scalarField pi(pp.patchInternalField());
                const scalarField& rfb = rhof.boundaryField()[patchi];
                const scalarField& phb =
                    phiHbyAv.boundaryField()[patchi];
                scalarField& phibf = phi.boundaryFieldRef()[patchi];

                for (label k = 0; k < np; k++)
                {
                    const scalar fluxB =
                        bDiagA[off + k]*pi[k] - bSrcA[off + k];
                    phibf[k] = rfb[k]*(phb[k] + fluxB);
                }
                off += np;
            }
        }
    }
    else
    {
        fvScalarMatrix pDDtEqn
        (
            psis*fvm::ddt(p)
          + fvc::div(phiHbyAv)
        );
        if (LADrhoCoeff > 0)
        {
            // AMD 항도 계수가 켜졌을 때만 조립 (0 기여, 생략은 비트-동일)
            pDDtEqn -= fvc::laplacian(Dr, rho)/rho;
        }

        while (pimple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rAUf, p));

            pEqn.setReference
            (
                pressureReference.refCell(),
                pressureReference.refValue()
            );

            fvConstraints().constrain(pEqn);

            {
                std::chrono::steady_clock::time_point tS;
                if (thermoTimings_)
                {
                    tS = std::chrono::steady_clock::now();
                }
                pEqn.solve();
                if (thermoTimings_)
                {
                    Info<< "pCorr pEqn.solve = "
                        << std::chrono::duration<double>
                           (std::chrono::steady_clock::now() - tS).count()
                        << " s" << endl;
                }
            }

            if (pimple.finalNonOrthogonalIter())
            {
                // Reconstruct the mass flux from the corrected volumetric
                // flux
                phi = rhof*(phiHbyAv + pEqn.flux());
            }
        }
    }

    // --- POSITIVITY GUARD: floor the SOLVED pressure to a physical minimum ---
    // ROOT ISSUE (2026-07-02): the swirl low-pressure core + stiff cold-LOX SRK
    // drive the pEqn to undershoot the SOLVED pressure NEGATIVE (min p ~ -1e7
    // observed even after the EOS-internal density floor). Absolute p<0 is
    // thermodynamically impossible; via SRK rho = p/(Z R T) it makes rho<0 and,
    // through -grad(p), pumps |U| overshoots that collapse dt. The EOS-internal
    // floor only protects rho -- the solved p field itself must also be bounded.
    // Floor p here (right after the solve, before correctRho/relax/U) so every
    // downstream step sees a positive pressure. pMinPa read each step (default 0
    // = off). Positivity guard confined to sub-physical cells, paired with the
    // EOS floor; a physically-exact version would use p_sat(T) instead.
    const scalar pMinPa
    (
        pimple.dict().lookupOrDefault<scalar>("pMinPa", scalar(0))
    );
    if (pMinPa > 0)
    {
        p = max(p, dimensionedScalar("pMinPa", p.dimensions(), pMinPa));
        p.correctBoundaryConditions();
    }

    // Upper positivity-guard twin of pMinPa: cap the solved pressure. Used as
    // a TEMPORARY surgical bound while the ignition-phase flame-zone spike
    // dissolves (fvConstraints limitPressure proved inert here even after
    // constrain(p) -- root cause not chased; this knob is on the proven
    // pMinPa path). pMaxPa read each step (default 0 = off).
    const scalar pMaxPa
    (
        pimple.dict().lookupOrDefault<scalar>("pMaxPa", scalar(0))
    );
    if (pMaxPa > 0)
    {
        p = min(p, dimensionedScalar("pMaxPa", p.dimensions(), pMaxPa));
        p.correctBoundaryConditions();
    }

    // Thermodynamic density update: the stock SIMPLErho increment
    // correctRho(psi*dp) is DISABLED in the PEP path. At a flame-zone
    // pressure spike the increment is huge (psi_gas ~1e-5 x dp ~ -4e8 ->
    // drho ~ -4000) and drove the THERMO density to -4869 kg/m3 in written
    // states -- and the per-corrector updateManifold()+thermo_.correct()+
    // rho_ re-sync at the top of this function already provides the full
    // EOS-consistent density update (one-corrector lag, nOuter >= 3).

    // Continuity diagnostics (base isothermalFluid::continuityErrors wraps
    // this grandparent call)
    fluidSolver::continuityErrors(rho, thermo.rho(), phi);

    // Explicitly relax pressure for momentum corrector
    p.relax();

    // Apply the field-level fvConstraints (limitPressure min/max). The stock
    // correctPressure applies them after the solve; the PEP override had
    // omitted this call, leaving constant/fvConstraints limitP INERT (both
    // its min and the surgical max) -- only the pMinPa floor was active.
    fvConstraints().constrain(p);

    U = HbyA - rAAtU*fvc::grad(p);
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);
    K = 0.5*magSqr(U);

    // SIMPLErho: density from the equation of state
    if (pimple.simpleRho())
    {
        rho = thermo.rho();
        rho.relax();
    }

    // Correct rhoUf if the mesh is moving
    fvc::correctRhoUf(rhoUf, rho, U, phi, MRF);

    if (thermo.dpdt())
    {
        dpdt = fvc::ddt(p);
    }

    if (thermoTimings_)
    {
        Info<< "pCorr total = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tTot).count()
            << " s" << endl;
    }
}


void Foam::solvers::fgmFluid::pressureCorrector()
{
    if (buoyancy.valid())
    {
        FatalErrorInFunction
            << "fgmFluid PEP pressure corrector does not support buoyant "
            << "(p_rgh) cases." << exit(FatalError);
    }

    while (pimple.correct())
    {
        correctPressurePEP();
    }

    tUEqn.clear();
}


// ************************************************************************* //
