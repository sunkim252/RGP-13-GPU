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

Class
    Foam::combustionModels::FGM

Description
    Diffusion-flame Flamelet-Generated Manifold (FPV + presumed beta-PDF)
    combustion model.

    The mixture fraction Z and progress variable C are transported as
    regular species Yi in the multicomponentFluid solver (the "piggyback"
    coupling): Z carries no source, C is driven by the tabulated source
    omega_C. The mixture-fraction subgrid variance is closed algebraically
    for LES,
        Zvar = Cv * Delta^2 * |grad(Z~)|^2,
    and the segregation factor gZ = Zvar/(Z~(1-Z~)) together with Z~ and C~
    indexes the 3-D FGM table (see FGMTable) to return omega_C.

    Qdot returns zero; temperature follows from species transport and the
    real-fluid thermodynamic coupling.

SourceFiles
    FGM.C

\*---------------------------------------------------------------------------*/

#ifndef FGM_H
#define FGM_H

#include "combustionModel.H"
#include "FGMTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace combustionModels
{

/*---------------------------------------------------------------------------*\
                             Class FGM Declaration
\*---------------------------------------------------------------------------*/

class FGM
:
    public combustionModel
{
    // Private data

        //- FGM lookup table
        FGMTable table_;

        //- Name of the mixture-fraction species
        word zName_;

        //- Name of the progress-variable species
        word cName_;

        //- Index of the mixture fraction in the species list
        label zIndex_;

        //- Index of the progress variable in the species list
        label cIndex_;

        //- LES mixture-fraction variance coefficient (Cv)
        scalar Cv_;

        //- Cached source term for progress variable [kg/m^3/s]
        volScalarField::Internal sourcePV_;

        //- Cached heat release rate [kg/m/s^3]
        volScalarField::Internal Qdot_;


public:

    //- Runtime type information
    TypeName("FGM");


    // Constructors

        //- Construct from components
        FGM
        (
            const word& modelType,
            const fluidMulticomponentThermo& thermo,
            const compressibleMomentumTransportModel& turb,
            const word& combustionProperties
        );

        //- Disallow default bitwise copy construction
        FGM(const FGM&) = delete;


    //- Destructor
    virtual ~FGM();


    // Member Functions

        //- Correct combustion rate
        virtual void correct();

        //- Specie consumption rate field
        virtual tmp<volScalarField::Internal> R(const label speciei) const;

        //- Specie consumption rate matrix
        virtual tmp<fvScalarMatrix> R(volScalarField& Y) const;

        //- Heat release rate [kg/m/s^3]
        virtual tmp<volScalarField> Qdot() const;

        //- Update properties from given dictionary
        virtual bool read();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const FGM&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace combustionModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
