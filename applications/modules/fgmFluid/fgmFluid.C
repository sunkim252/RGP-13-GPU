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
#include "localEulerDdtScheme.H"
#include "fvcGrad.H"
#include "DynamicList.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(fgmFluid, 0);
    addToRunTimeSelectionTable(solver, fgmFluid, fvMesh);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solvers::fgmFluid::fgmFluid(fvMesh& mesh)
:
    isothermalFluid
    (
        mesh,
        autoPtr<fluidThermo>(fluidMulticomponentThermo::New(mesh).ptr())
    ),

    thermo_(refCast<fluidMulticomponentThermo>(isothermalFluid::thermo_)),

    Y_(thermo_.Y()),

    fgmTable_(mesh),

    Z_
    (
        IOobject("Z", runTime.name(), mesh, IOobject::MUST_READ,
                 IOobject::AUTO_WRITE),
        mesh
    ),

    C_
    (
        IOobject("C", runTime.name(), mesh, IOobject::MUST_READ,
                 IOobject::AUTO_WRITE),
        mesh
    ),

    gZ_
    (
        IOobject("gZ", runTime.name(), mesh, IOobject::NO_READ,
                 IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless, 0)
    ),

    sourcePV_
    (
        IOobject("FGM:sourcePV", runTime.name(), mesh, IOobject::NO_READ,
                 IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimDensity/dimTime, 0)
    ),

    Cv_(fgmTable_.lookupOrDefault<scalar>("Cv", 0.1)),

    Sct_(fgmTable_.lookupOrDefault<scalar>("Sct", 0.7)),

    thermophysicalTransport
    (
        fluidMulticomponentThermophysicalTransportModel::New
        (
            momentumTransport(),
            thermo_
        )
    ),

    thermo(thermo_),
    Y(Y_)
{
    // fgmFluid uses ABSOLUTE enthalpy/energy so combustion heat release is
    // implicit in the tabulated composition's formation enthalpy (no Qdot).
    thermo.validate(type(), "ha", "ea");

    // All thermo species are reconstructed from the FGM manifold, so none are
    // transported: mark every specie inactive (the solver then skips its
    // YEqn) and record which ones the table actually provides.
    DynamicList<label> tabIDs;
    forAll(Y_, i)
    {
        thermo_.setSpecieInactive(i);
        if (fgmTable_.hasSpecie(thermo_.species()[i]))
        {
            tabIDs.append(i);
        }
    }
    tabSpecieIDs_ = tabIDs;

    // Per-variable Lewis numbers. Control variables default to unity; an
    // optional 'Le' sub-dict in fgmProperties overrides any name (control
    // variable or, in the differential-diffusion extension, species).
    Le_.set("Z", 1.0);
    Le_.set("C", 1.0);
    if (fgmTable_.found("Le"))
    {
        const dictionary& leDict = fgmTable_.subDict("Le");
        const wordList keys(leDict.toc());
        forAll(keys, i)
        {
            Le_.set(keys[i], leDict.lookup<scalar>(keys[i]));
        }
    }

    Info<< "fgmFluid: " << tabSpecieIDs_.size() << " of " << Y_.size()
        << " species tabulated; Cv = " << Cv_ << ", Sct = " << Sct_ << nl
        << "    Lewis numbers: " << Le_ << nl
        << "    control variables Z, C transported; thermo from FGM "
        << "composition + real-fluid EOS" << nl << endl;

    // Multivariate convection over the transported scalars (he, Z, C).
    fields.add(thermo.he());
    fields.add(Z_);
    fields.add(C_);

    // Initialise the manifold-derived fields.
    updateManifold();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solvers::fgmFluid::~fgmFluid()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::updateManifold()
{
    // |grad(Z)|^2 for the algebraic LES subgrid-variance closure
    const tmp<volScalarField> tmagSqrGradZ(magSqr(fvc::grad(Z_)));
    const scalarField& magSqrGradZ = tmagSqrGradZ();
    const scalarField& V = mesh.V();
    const scalarField& rhoc = rho;

    scalarField& gZc = gZ_.primitiveFieldRef();
    scalarField& srcc = sourcePV_.primitiveFieldRef();
    const scalarField& Zc = Z_;
    const scalarField& Cc = C_;

    // Hoist references to the tabulated species fields
    List<scalarField*> Yref(tabSpecieIDs_.size());
    List<word> Yname(tabSpecieIDs_.size());
    forAll(tabSpecieIDs_, k)
    {
        const label id = tabSpecieIDs_[k];
        Yref[k] = &Y_[id].primitiveFieldRef();
        Yname[k] = thermo_.species()[id];
    }

    forAll(gZc, celli)
    {
        const scalar Zcl = max(min(Zc[celli], scalar(1)), scalar(0));
        const scalar Ccl = max(Cc[celli], scalar(0));

        // LES algebraic variance + segregation factor
        const scalar delta = cbrt(V[celli]);
        const scalar Zvar = Cv_*sqr(delta)*magSqrGradZ[celli];
        const scalar gz =
            min(max(Zvar/max(Zcl*(scalar(1) - Zcl), SMALL), scalar(0)), scalar(1));
        gZc[celli] = gz;

        // PV source: tabulated mass-fraction rate [1/s] -> volumetric [kg/m^3/s]
        srcc[celli] = rhoc[celli]*fgmTable_.interpolate(Zcl, gz, Ccl);

        // Reconstruct composition from the manifold
        forAll(Yref, k)
        {
            (*Yref[k])[celli] = fgmTable_.interpolateY(Yname[k], Zcl, gz, Ccl);
        }
    }

    gZ_.correctBoundaryConditions();
    forAll(tabSpecieIDs_, k)
    {
        Y_[tabSpecieIDs_[k]].correctBoundaryConditions();
    }
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::fgmFluid::Deff(const word& var)
{
    const scalar Le = Le_.found(var) ? Le_[var] : 1.0;

    // Laminar mass diffusivity rho*D_lam = mu/Le (unity-Lewis base scaled by
    // the variable's Lewis number), plus the turbulent subgrid contribution
    // rho*nut/Sct. All dynamic [kg/m/s]. Per-species Le slots in here for the
    // future differential-diffusion extension.
    return thermo.mu()/Le + rho*momentumTransport().nut()/Sct_;
}


void Foam::solvers::fgmFluid::thermophysicalTransportPredictor()
{
    thermophysicalTransport->predict();
}


void Foam::solvers::fgmFluid::thermophysicalTransportCorrector()
{
    thermophysicalTransport->correct();
}


// ************************************************************************* //
