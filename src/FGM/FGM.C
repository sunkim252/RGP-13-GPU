/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2024 OpenFOAM Foundation
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

#include "FGM.H"
#include "fvmSup.H"
#include "fvcGrad.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{
    defineTypeNameAndDebug(FGM, 0);
    addToRunTimeSelectionTable(combustionModel, FGM, dictionary);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::combustionModels::FGM::FGM
(
    const word& modelType,
    const fluidMulticomponentThermo& thermo,
    const compressibleMomentumTransportModel& turb,
    const word& combustionProperties
)
:
    combustionModel(modelType, thermo, turb),
    table_(this->mesh()),
    zName_
    (
        this->coeffs().lookupOrDefault<word>("mixtureFraction", "Z")
    ),
    cName_
    (
        this->coeffs().lookupOrDefault<word>("progressVariable", "C")
    ),
    zIndex_(-1),
    cIndex_(-1),
    Cv_
    (
        this->coeffs().lookupOrDefault<scalar>("Cv", 0.1)
    ),
    sourcePV_
    (
        IOobject
        (
            "FGM:sourcePV",
            this->mesh().time().name(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimDensity/dimTime, 0)
    ),
    Qdot_
    (
        IOobject
        (
            "FGM:Qdot",
            this->mesh().time().name(),
            this->mesh(),
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh(),
        dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
    )
{
    const speciesTable& species = thermo.species();

    zIndex_ = species[zName_];
    cIndex_ = species[cName_];

    Info<< "FGM (FPV + beta-PDF) combustion model:" << nl
        << "    Mixture fraction:  " << zName_
        << "  (species index " << zIndex_ << ")" << nl
        << "    Progress variable: " << cName_
        << "  (species index " << cIndex_ << ")" << nl
        << "    LES variance coeff Cv = " << Cv_ << nl
        << endl;
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::combustionModels::FGM::~FGM()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::combustionModels::FGM::correct()
{
    const volScalarField& Z = this->thermo().Y(zIndex_);
    const volScalarField& C = this->thermo().Y(cIndex_);

    // Keep the density tmp alive while we read it.
    const tmp<volScalarField> trho(this->thermo().rho());
    const volScalarField& rho = trho();

    // |grad(Z)|^2 for the algebraic subgrid variance closure.
    const tmp<volScalarField> tmagSqrGradZ(magSqr(fvc::grad(Z)));
    const volScalarField& magSqrGradZ = tmagSqrGradZ();

    const scalarField& V = this->mesh().V();

    forAll(sourcePV_, celli)
    {
        const scalar Zc = max(min(Z[celli], scalar(1)), scalar(0));
        const scalar Cc = max(C[celli], scalar(0));

        // LES filter width Delta = V^(1/3); algebraic variance + segregation.
        const scalar delta = cbrt(V[celli]);
        const scalar Zvar = Cv_*sqr(delta)*magSqrGradZ[celli];
        const scalar Zfluc = max(Zc*(scalar(1) - Zc), SMALL);
        const scalar gZ = min(max(Zvar/Zfluc, scalar(0)), scalar(1));

        // Tabulated mass-fraction PV source [1/s] -> volumetric [kg/m^3/s].
        sourcePV_[celli] = rho[celli]*table_.interpolate(Zc, gZ, Cc);
    }

    // Qdot = 0 for now; temperature follows species transport + real-fluid
    // thermodynamic coupling.
    Qdot_ = dimensionedScalar(dimEnergy/dimVolume/dimTime, 0);
}


Foam::tmp<Foam::volScalarField::Internal>
Foam::combustionModels::FGM::R(const label speciei) const
{
    if (speciei == cIndex_)
    {
        return sourcePV_;
    }

    // Mixture fraction Z and all other species carry no chemical source.
    return
        volScalarField::Internal::New
        (
            typedName("R_" + this->thermo().Y()[speciei].name()),
            this->mesh(),
            dimensionedScalar(dimDensity/dimTime, 0)
        );
}


Foam::tmp<Foam::fvScalarMatrix>
Foam::combustionModels::FGM::R(volScalarField& Y) const
{
    tmp<fvScalarMatrix> tSu(new fvScalarMatrix(Y, dimMass/dimTime));

    if (Y.name() == cName_)
    {
        // Explicit source for the progress variable. fvScalarMatrix stores
        // the RHS source with a negative sign, so subtract rate * V.
        tSu.ref().source() -= sourcePV_ * this->mesh().V();
    }

    return tSu;
}


Foam::tmp<Foam::volScalarField>
Foam::combustionModels::FGM::Qdot() const
{
    return volScalarField::New
    (
        this->thermo().phasePropertyName(typedName("Qdot")),
        this->mesh(),
        dimensionedScalar(dimEnergy/dimVolume/dimTime, 0)
    );
}


bool Foam::combustionModels::FGM::read()
{
    if (combustionModel::read())
    {
        return true;
    }
    else
    {
        return false;
    }
}


// ************************************************************************* //
