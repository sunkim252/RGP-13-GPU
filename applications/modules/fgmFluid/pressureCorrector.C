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

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

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
    updateManifold();
    thermo_.correct();

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

    fvScalarMatrix pDDtEqn
    (
        psis*fvm::ddt(p)
      + fvc::div(phiHbyAv)
      - fvc::laplacian(Dr, rho)/rho
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rAUf, p));

        pEqn.setReference
        (
            pressureReference.refCell(),
            pressureReference.refValue()
        );

        fvConstraints().constrain(pEqn);

        pEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            // Reconstruct the mass flux from the corrected volumetric flux
            phi = rhof*(phiHbyAv + pEqn.flux());
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
