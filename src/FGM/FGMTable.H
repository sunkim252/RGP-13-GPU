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

    Reads a 3-D or 4-D table of the Favre-filtered progress-variable source
    term and the tabulated state from constant/fgmProperties.

    Axes:
        Z~   = Favre mean mixture fraction,
        gZ   = Zvar/(Z~(1-Z~)) = mixture-fraction segregation factor [0,1),
        C~   = Favre mean progress variable,
        chi  = scalar dissipation rate evaluated at Z = Z_st [1/s]
               (4-D version only; legacy 3-D tables omit this axis).

    The dictionary provides the axis sizes (nZ, nGz, nC, [nChi]), the axes
    (Z, gZ, C, [chi]) and the flat tables. C-order indexing:

        4-D:  idx = ((iZ*nGz + iGz)*nC + iC)*nChi + iChi
        3-D:  idx =  (iZ*nGz + iGz)*nC + iC        (legacy)

    Interpolation is quadrilinear in 4-D, trilinear in 3-D. The legacy
    interpolate(Z, gZ, C) entry points still work and evaluate the table
    at chi = chi_axis_[0] when the table is 4-D.

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
        label nChi_;            //!< 1 if the table is legacy 3-D

        //- True if a chi axis was provided in fgmProperties
        bool hasChi_;

        //- Mixture-fraction axis (Z~)
        List<scalar> Z_axis_;

        //- Segregation-factor axis (gZ = Zvar/(Z~(1-Z~)))
        List<scalar> gZ_axis_;

        //- Progress-variable axis (C~)
        List<scalar> C_axis_;

        //- Scalar-dissipation axis (chi_st [1/s]); single entry {0} for
        //  legacy 3-D tables.
        List<scalar> chi_axis_;

        //- Flat PV source table.  Length nZ*nGz*nC*nChi.
        List<scalar> sourcePV_;

        //- Optional tabulated temperature [K] (empty if not present).
        List<scalar> T_table_;

        //- Tabulated species mass fractions, keyed by species name.
        //  Each entry is a flat list of length nZ*nGz*nC*nChi.
        HashTable<List<scalar>> Y_tables_;

        //- Names of the tabulated species (order as in the 'species' entry)
        wordList speciesNames_;


    // Private Member Functions

        //- Quadrilinear interpolation of an arbitrary flat 4-D table.
        scalar interpolateTable
        (
            const List<scalar>& table,
            scalar Z, scalar gZ, scalar C, scalar chi
        ) const;

        //- Locate value v in a sorted axis; returns lower index i and sets
        //  the linear weight w in [0,1] such that
        //  result = (1-w)*axis[i] + w*axis[i+1]. Clamps to the ends. For
        //  axes of length 1 the call is degenerate: returns i=0, w=0 so
        //  the multilinear interpolation collapses to that slice.
        void bracket
        (
            const List<scalar>& axis,
            scalar v,
            label& i,
            scalar& w
        ) const;

        //- Flat index into the 4-D table.
        inline label flatIndex
        (
            label iZ, label iGz, label iC, label iChi
        ) const
        {
            return ((iZ*nGz_ + iGz)*nC_ + iC)*nChi_ + iChi;
        }


public:

    // Constructors

        //- Construct from mesh
        FGMTable(const fvMesh& mesh);


    //- Destructor
    ~FGMTable();


    // Member Functions

        // 4-D entry points (preferred for new clients)

            //- Quadrilinear interpolation of the PV source term.
            scalar interpolate
            (
                scalar Z, scalar gZ, scalar C, scalar chi
            ) const;

            //- Quadrilinear interpolation of tabulated temperature [K].
            scalar interpolateT
            (
                scalar Z, scalar gZ, scalar C, scalar chi
            ) const;

            //- Quadrilinear interpolation of a tabulated species mass
            //  fraction.
            scalar interpolateY
            (
                const word& specie,
                scalar Z, scalar gZ, scalar C, scalar chi
            ) const;

        // 3-D legacy entry points
        //  Evaluate at chi = chi_axis_[0]. Required so that the original
        //  FGM combustion model and any existing case that still ships a
        //  3-D fgmProperties keeps working unchanged.

            scalar interpolate (scalar Z, scalar gZ, scalar C) const;
            scalar interpolateT(scalar Z, scalar gZ, scalar C) const;
            scalar interpolateY
            (
                const word& specie, scalar Z, scalar gZ, scalar C
            ) const;

        // Predicates

            inline bool hasSpecie(const word& specie) const
            { return Y_tables_.found(specie); }

            inline bool hasT() const { return !T_table_.empty(); }

            //- True if the table carries a chi_st axis (nChi > 1).
            inline bool hasChi() const { return hasChi_; }

        // Access

            inline const List<scalar>& Zaxis()   const { return Z_axis_;   }
            inline const List<scalar>& gZaxis()  const { return gZ_axis_;  }
            inline const List<scalar>& Caxis()   const { return C_axis_;   }
            inline const List<scalar>& chiAxis() const { return chi_axis_; }
            inline const List<scalar>& sourcePV()const { return sourcePV_; }
            inline const wordList& speciesNames() const
            { return speciesNames_; }

            inline label nZ()   const { return nZ_;   }
            inline label nGz()  const { return nGz_;  }
            inline label nC()   const { return nC_;   }
            inline label nChi() const { return nChi_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
