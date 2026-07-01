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

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::correctPressurePEP()
{
    volScalarField& rho(rho_);
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

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
    const surfaceScalarField rAUf("rAUf", fvc::interpolate(rAU));
    // ATTEMPT 2: real SRK isothermal compressibility psis = (drho/dp)_T/rho =
    // kappa_T (thermo.psi() now returns the real (drho/dp)_T -- see SRKGasI.H).
    // This supplies the true dense-fluid stiffness that the ideal-gas 1/(gamma
    // p) lacked (~100x too soft for liquid LOX).
    const volScalarField psis("psis", thermo.psi()/rho);

    // Volumetric predicted face flux (no rho weighting) with transient
    // Rhie-Chow (ddtCorr) on the volumetric flux phi/rhof.
    surfaceScalarField phiHbyAv
    (
        "phiHbyAv",
        fvc::flux(HbyA)
      + rhorAUf*fvc::ddtCorr(U, phi/rhof)   // volumetric transient Rhie-Chow
    );
    MRF.makeRelative(phiHbyAv);

    // Update the pressure BCs for flux consistency (3D: waveTransmissive
    // outlet, fixedFluxPressure walls). Volumetric-flux form: pass the
    // volumetric predicted flux and the rAUf (velocity-level) coefficient,
    // matching the pEqn's laplacian(rAUf, p).
    constrainPressure(p, rho, U, phiHbyAv, rAUf, MRF);

    fvScalarMatrix pDDtEqn
    (
        psis*fvm::ddt(p)
      + fvc::div(phiHbyAv)
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

    // Thermodynamic density update (non-steady, SIMPLErho path)
    thermo_.correctRho(psi*p - psip0);

    // Continuity diagnostics (base isothermalFluid::continuityErrors wraps
    // this grandparent call)
    fluidSolver::continuityErrors(rho, thermo.rho(), phi);

    // Explicitly relax pressure for momentum corrector
    p.relax();

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
