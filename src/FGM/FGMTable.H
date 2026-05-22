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
    Foam::FGMTable

Description
    Lookup table for the diffusion-flame (FPV + presumed beta-PDF) FGM
    combustion model.

    Reads a 3-D table of the Favre-filtered progress-variable source term
    from constant/fgmProperties, as a function of
        (Z~ , gZ , C~)
    where
        Z~  = Favre mean mixture fraction,
        gZ  = Zvar/(Z~(1-Z~)) = mixture-fraction segregation factor [0,1),
        C~  = Favre mean progress variable.

    The dictionary provides the axis sizes (nZ, nGz, nC), the three axes
    (Z, gZ, C) and a flat sourcePV list in C-order:
        idx = (iZ*nGz + iGz)*nC + iC
    Trilinear interpolation is provided via lookup().

SourceFiles
    FGMTable.C

\*---------------------------------------------------------------------------*/

#ifndef FGMTable_H
#define FGMTable_H

#include "IOdictionary.H"
#include "fvMesh.H"
#include "scalar.H"
#include "List.H"
#include "HashTable.H"
#include "wordList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class FGMTable Declaration
\*---------------------------------------------------------------------------*/

class FGMTable
:
    public IOdictionary
{
    // Private data

        //- Reference to the mesh database
        const fvMesh& mesh_;

        //- Axis sizes
        label nZ_;
        label nGz_;
        label nC_;

        //- Mixture-fraction axis (Z~)
        List<scalar> Z_axis_;

        //- Segregation-factor axis (gZ = Zvar/(Z~(1-Z~)))
        List<scalar> gZ_axis_;

        //- Progress-variable axis (C~)
        List<scalar> C_axis_;

        //- Flat 3-D PV source table, C-order idx = (iZ*nGz + iGz)*nC + iC
        List<scalar> sourcePV_;

        //- Optional tabulated temperature [K] (empty if not present)
        List<scalar> T_table_;

        //- Tabulated species mass fractions, keyed by species name.
        //  Each entry is a flat nZ*nGz*nC list in the same C-order.
        HashTable<List<scalar>> Y_tables_;

        //- Names of the tabulated species (order as in the 'species' entry)
        wordList speciesNames_;


    // Private Member Functions

        //- Trilinear interpolation of an arbitrary flat table at (Z~, gZ, C~)
        scalar interpolateTable(const List<scalar>& table,
                                scalar Z, scalar gZ, scalar C) const;

        //- Locate value v in a sorted axis; returns lower index i and sets
        //  the linear weight w in [0,1] such that
        //  result = (1-w)*axis[i] + w*axis[i+1]. Clamps to the ends.
        void bracket
        (
            const List<scalar>& axis,
            scalar v,
            label& i,
            scalar& w
        ) const;

        //- Flat index into sourcePV_
        inline label flatIndex(label iZ, label iGz, label iC) const
        {
            return (iZ*nGz_ + iGz)*nC_ + iC;
        }


public:

    // Constructors

        //- Construct from mesh
        FGMTable(const fvMesh& mesh);


    //- Destructor
    ~FGMTable();


    // Member Functions

        //- Trilinear interpolation of the PV source term at (Z~, gZ, C~).
        //  Named 'interpolate' (not 'lookup') to avoid hiding the inherited
        //  dictionary::lookup used in the constructor.
        scalar interpolate(scalar Z, scalar gZ, scalar C) const;

        //- Trilinear interpolation of the tabulated temperature [K]
        scalar interpolateT(scalar Z, scalar gZ, scalar C) const;

        //- Trilinear interpolation of a tabulated species mass fraction
        scalar interpolateY(const word& specie,
                            scalar Z, scalar gZ, scalar C) const;

        //- True if the given species is tabulated
        inline bool hasSpecie(const word& specie) const
        {
            return Y_tables_.found(specie);
        }

        //- True if temperature is tabulated
        inline bool hasT() const { return !T_table_.empty(); }

        // Access

            inline const List<scalar>& Zaxis() const { return Z_axis_; }
            inline const List<scalar>& gZaxis() const { return gZ_axis_; }
            inline const List<scalar>& Caxis() const { return C_axis_; }
            inline const List<scalar>& sourcePV() const { return sourcePV_; }
            inline const wordList& speciesNames() const { return speciesNames_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
