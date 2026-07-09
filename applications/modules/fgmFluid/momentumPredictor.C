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

\*---------------------------------------------------------------------------*/

#include "fgmFluid.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvcGrad.H"
#include "fvcDiv.H"
#include "zeroGradientFvPatchFields.H"

#include <chrono>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::momentumPredictor()
{
    std::chrono::steady_clock::time_point tTot;
    if (thermoTimings_) { tTot = std::chrono::steady_clock::now(); }

    volVectorField& U(U_);

    // --- Localized Artificial (shear) Viscosity on the momentum -------------
    // Companion to the scalar mass-diffusivity LAD: damps at source the
    // spurious VELOCITY overshoot at the transcritical density interface
    // (diagnosed recess-tip |U| spike, limitMag-capped at 500, that throttles
    // the time step). A density-gradient-sensed artificial dynamic viscosity
    //   muArt = LADUCoeff * V^(2/3) * |U| * |grad(rho)|   [kg/(m s)]
    // is added to the viscous stress (-fvm::laplacian(muArt, U)); cell-sized so
    // its own viscous time step stays of order the convective one and smooth
    // regions (|grad(rho)| ~ 0) are untouched -- the LES subgrid stress is
    // unaffected away from the interface. Cook & Cabot, J. Comput. Phys. 195
    // (2004); Kawai & Lele, J. Comput. Phys. 227 (2008); Kawai, Terashima &
    // Negishi, J. Comput. Phys. 300 (2015). LADUCoeff read from the PIMPLE
    // dict each step (runTimeModifiable); default 0 = off.
    const scalar LADUCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADUCoeff", scalar(0))
    );
    volScalarField muArt
    (
        IOobject
        (
            "muArt",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1, -1, -1, 0, 0, 0, 0), 0),
        zeroGradientFvPatchScalarField::typeName
    );
    if (LADUCoeff > 0)
    {
        const scalarField V23(pow(scalarField(mesh.V()), 2.0/3.0));
        muArt.primitiveFieldRef() =
            LADUCoeff*V23
           *mag(U)().primitiveField()
           *mag(fvc::grad(rho))().primitiveField();
        muArt.correctBoundaryConditions();
        Info<< "LAD-U: muArt max = " << gMax(muArt.primitiveField())
            << " kg/(m s)" << endl;
    }

    // Artificial BULK viscosity (Cook-Cabot) -- damps the DILATATIONAL /
    // compressive part directly, targeting the injector pressure oscillation
    // (p +/- spike and |U| overshoot at the fine tangential-hole cells) that
    // the shear muArt and the Rhie-Chow fixes do not reach. Dilatation-sensed
    //   betaArt = LADbulkCoeff * rho * V^(2/3) * |div U|   [kg/(m s)]
    // added as -grad(betaArt*div(U)) to the momentum. Cook & Cabot, J. Comput.
    // Phys. 195 (2004) 594; Kawai, Terashima & Negishi, J. Comput. Phys. 300
    // (2015) 116. Read each step (runTimeModifiable); default 0 = off.
    const scalar LADbulkCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADbulkCoeff", scalar(0))
    );
    const volScalarField divU(fvc::div(U));
    volScalarField betaArt
    (
        IOobject
        (
            "betaArt",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1, -1, -1, 0, 0, 0, 0), 0),
        zeroGradientFvPatchScalarField::typeName
    );
    if (LADbulkCoeff > 0)
    {
        const scalarField V23(pow(scalarField(mesh.V()), 2.0/3.0));
        betaArt.primitiveFieldRef() =
            LADbulkCoeff*rho.primitiveField()*V23
           *mag(divU.primitiveField());
        betaArt.correctBoundaryConditions();
        Info<< "LAD-bulk: betaArt max = " << gMax(betaArt.primitiveField())
            << " kg/(m s)" << endl;
    }

    tUEqn =
    (
        fvm::ddt(rho, U) + fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + momentumTransport->divDevTau(U)
      - fvm::laplacian(muArt, U)
      - fvc::grad(betaArt*divU)
     ==
        fvModels().source(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    UEqn.relax();

    fvConstraints().constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        if (buoyancy.valid())
        {
            solve
            (
                UEqn
             ==
                netForce()
            );
        }
        else
        {
            solve(UEqn == -fvc::grad(p));
        }

        fvConstraints().constrain(U);
        K = 0.5*magSqr(U);
    }

    if (thermoTimings_)
    {
        Info<< "momentum predictor total = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tTot).count()
            << " s" << endl;
    }
}


// ************************************************************************* //
