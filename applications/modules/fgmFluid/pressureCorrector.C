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

    surfaceScalarField phiHbyA
    (
        "phiHbyA",
        rhof*fvc::flux(HbyA)
      + rhorAUf*fvc::ddtCorr(rho, U, phi, rhoUf)
    );

    MRF.makeRelative(rhof, phiHbyA);

    if (pimple.consistent())
    {
        phiHbyA += (rhorAAtUf - rhorAUf)*fvc::snGrad(p)*mesh.magSf();
        HbyA += (rAAtU - rAU)*fvc::grad(p);
    }

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p, rho, U, phiHbyA, rhorAAtUf, MRF);

    fvc::makeRelative(phiHbyA, rho, U);

    fvScalarMatrix pDDtEqn
    (
        fvc::ddt(rho) + psi*correction(fvm::ddt(p))
      + fvc::div(phiHbyA)
     ==
        fvModels().sourceProxy(rho, p)
    );

    while (pimple.correctNonOrthogonal())
    {
        fvScalarMatrix pEqn(pDDtEqn - fvm::laplacian(rhorAAtUf, p));

        pEqn.setReference
        (
            pressureReference.refCell(),
            pressureReference.refValue()
        );

        fvConstraints().constrain(pEqn);

        pEqn.solve();

        if (pimple.finalNonOrthogonalIter())
        {
            phi = phiHbyA + pEqn.flux();
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
