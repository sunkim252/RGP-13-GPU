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
    // NOTE: thermo_.normaliseY() REMOVED -- with every specie marked inactive
    // it sets the default specie (N2) = 1 - sum(ACTIVE) = 1 (no active species),
    // spuriously diluting the tabulated composition (which already sums to 1,
    // with N2 = 0 from the table) by ~50% N2 after the mixture renormalises.
    // The FGM manifold already delivers a normalised composition, so the call
    // is unnecessary and corrupts it.

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

        // The transported progress variable is the NORMALIZED c in [0, 1]
        // (table closure: omega = 0 at c = 1, the equilibrium/envelope
        // boundary). The explicit source can step THROUGH the c = 1 zero in
        // a single dt (omega ~ 1e4 1/s), so clamp both ends -- the standard
        // practice layered on top of the equilibrium closure (Pierce 2004's
        // library truncation; cf. solver-side bounding in tabulated codes).
        C_ = max(min(C_, scalar(1)), scalar(0));
    }

    // --- Total-enthalpy transport (non-adiabatic FPV, method b) ---
    // Total (absolute) enthalpy is a CONSERVED scalar (no reaction source:
    // chemical heat release is internal to h), so this is a plain advection-
    // diffusion equation -- NOT the unstable he equation. It carries the cold-
    // inlet / heat-loss enthalpy in from the boundaries; the manifold lookup
    // then reads T(Z,gZ,c,dh) at the local defect dh = h - h_ad(Z). T is taken
    // from the table (not inverted from h), so the he<->T drift that forbids an
    // EEqn does not occur. Solved only when the table carries an enthalpy axis.
    if (fgmTable_.useEnthalpy())
    {
        volScalarField& h = hPtr_();
        const volScalarField Dh("Dh", Deff("h"));
        fvScalarMatrix hEqn
        (
            fvm::ddt(rho, h)
          + mvConvection->fvmDiv(phi, h)
          - fvm::laplacian(Dh, h)
         ==
            fvModels().source(rho, h)
        );

        hEqn.relax();
        fvConstraints().constrain(hEqn);
        hEqn.solve("Yi");
        fvConstraints().constrain(h);
    }


    // --- No transported energy equation (adiabatic FPV/FGM closure) ---
    // The thermochemical state (T, he, Y) is a tabulated function of the
    // manifold coordinates and was set in updateManifold() above: he has been
    // re-seeded to he(p, T_table) on the looked-up composition. We therefore
    // do NOT advance an EEqn for he here. The reason is essential, not an
    // optimisation: under fully-compressible acoustics a transported he is
    // perturbed off the manifold (the pressure-work/dilatation and convective
    // fluxes inject/remove energy that the algebraic composition update does
    // not see), so the he->T inversion drifts and the strained flame
    // eventually collapses. flameletFoam/FPVFoam transport only Z and c and
    // read T/he from the table for exactly this reason.
    //
    // thermo.correct() inverts the manifold he straight back to T_table and
    // refreshes rho, psi, mu, ... at the new (p, T, Y) for the pressure solve.
    thermo_.correct();
}


// ************************************************************************* //
