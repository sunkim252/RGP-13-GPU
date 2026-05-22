/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2024 OpenFOAM Foundation
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
#include "fvcDdt.H"
#include "fvmLaplacian.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::thermophysicalPredictor()
{
    tmp<fv::convectionScheme<scalar>> mvConvection
    (
        fv::convectionScheme<scalar>::New
        (
            mesh,
            fields,
            phi,
            mesh.schemes().div("div(phi,Yi_h)")
        )
    );

    // --- FGM manifold update: gZ, composition Y_k, PV source (from Z, C) ---
    updateManifold();
    thermo_.normaliseY();

    // --- Mixture-fraction transport (conserved scalar, no source) ---
    {
        const volScalarField DZ("DZ", Deff("Z"));
        fvScalarMatrix ZEqn
        (
            fvm::ddt(rho, Z_)
          + mvConvection->fvmDiv(phi, Z_)
          - fvm::laplacian(DZ, Z_)
         ==
            fvModels().source(rho, Z_)
        );

        ZEqn.relax();
        fvConstraints().constrain(ZEqn);
        ZEqn.solve("Yi");
        fvConstraints().constrain(Z_);

        Z_ = max(min(Z_, scalar(1)), scalar(0));
    }

    // --- Progress-variable transport (source = rho*omega_C from table) ---
    {
        const volScalarField DC("DC", Deff("C"));
        fvScalarMatrix CEqn
        (
            fvm::ddt(rho, C_)
          + mvConvection->fvmDiv(phi, C_)
          - fvm::laplacian(DC, C_)
         ==
            sourcePV_
          + fvModels().source(rho, C_)
        );

        CEqn.relax();
        fvConstraints().constrain(CEqn);
        CEqn.solve("Yi");
        fvConstraints().constrain(C_);

        C_ = max(C_, scalar(0));
    }


    volScalarField& he = thermo_.he();

    fvScalarMatrix EEqn
    (
        fvm::ddt(rho, he) + mvConvection->fvmDiv(phi, he)
      + fvc::ddt(rho, K) + fvc::div(phi, K)
      + pressureWork
        (
            he.name() == "e"
          ? mvConvection->fvcDiv(phi, p/rho)()
          : -dpdt
        )
      + thermophysicalTransport->divq(he)
     ==
        // No explicit Qdot: heat release is implicit in the tabulated
        // composition when the thermo uses absolute enthalpy.
        (
            buoyancy.valid()
          ? fvModels().source(rho, he) + rho*(U & buoyancy->g)
          : fvModels().source(rho, he)
        )
    );

    EEqn.relax();

    fvConstraints().constrain(EEqn);

    EEqn.solve();

    fvConstraints().constrain(he);

    thermo_.correct();
}


// ************************************************************************* //
