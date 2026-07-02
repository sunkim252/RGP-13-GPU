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

        //- True if the 4th axis is an enthalpy-defect axis (non-adiabatic FPV,
        //  method b) rather than scalar dissipation. When set, the 4th
        //  coordinate passed to interpolate() is the enthalpy defect
        //  dh = h_total - ((1-Z) hOx + Z hFuel) and the same quadrilinear
        //  machinery is reused (chi_axis_ then holds the dh axis [J/kg]).
        bool useEnthalpy_;

        //- Pure-stream total enthalpies at the flamelet boundary T [J/kg],
        //  used to form the adiabatic reference mixing line h_ad(Z).
        scalar hOx_;
        scalar hFuel_;

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

        //- Tier-2: optional pre-tabulated real-gas mixture coefficients, in
        //  tabulatedRealGasMixture::coeffNames() order (table key
        //  "RG_"+name). Each entry is a flat list of length nZ*nGz*nC*nChi.
        //  Empty when the table does not carry the RG_* fields.
        List<List<scalar>> RGcoeff_;

        //- Tier-4: optional differential-diffusion Lewis-number fields, keyed
        //  by control-variable name ("Z", "C"); dictionary key "Le_"+name.
        //  Each entry is a flat list of length nZ*nGz*nC*nChi. When present the
        //  solver applies a per-cell Le(Z,gZ,c[,chi]) to that variable's
        //  laminar diffusivity (rho*D = mu/Le), generalising the constant 'Le'
        //  sub-dict to the standard differential-diffusion FGM closure.
        HashTable<List<scalar>> Le_tables_;


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

    //- Shared interpolation stencil: the 4-axis bracket + 16 corner flat
    //  indices of ONE query point (Z,gZ,C,chi), reusable across every
    //  tabulated field. updateManifold evaluates ~120+ fields (106 Y + 13
    //  RG + Le + T + source) at the SAME point per cell; sharing the
    //  bracket search + index computation removes that per-field
    //  duplication. Corner order matches interpolateTable's c0000..c1111,
    //  and interpolate(table, stencil) reproduces its nested-blend
    //  arithmetic verbatim -- results are bit-identical.
    struct FGMStencil
    {
        label idx[16];
        scalar wZ, wG, wC, wK;
    };


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

            //- Quadrilinear interpolation of the Tier-2 real-gas mixture
            //  coefficients. Fills coeffs (resized to the number of RG_*
            //  fields) in coeffNames() order. No-op if hasRealGasCoeffs() is
            //  false.
            void interpolateRealGasCoeffs
            (
                scalar Z, scalar gZ, scalar C, scalar chi,
                List<scalar>& coeffs
            ) const;

            //- Quadrilinear interpolation of the Tier-4 tabulated Lewis number
            //  Le_<var> for a control variable. Caller must check
            //  hasLeField(var) first.
            scalar interpolateLe
            (
                const word& var,
                scalar Z, scalar gZ, scalar C, scalar chi
            ) const;

        // Shared-stencil entry points (fast path for many-field queries)

            //- Build the interpolation stencil for one query point.
            void makeStencil
            (
                scalar Z, scalar gZ, scalar C, scalar chi,
                FGMStencil& st
            ) const;

            //- Evaluate an arbitrary flat table on a prebuilt stencil
            //  (bit-identical to interpolateTable at the same point).
            scalar interpolate
            (
                const List<scalar>& table,
                const FGMStencil& st
            ) const;

            //- Direct table access for hoisting out of cell loops.
            inline const List<scalar>& sourcePVTable() const
            { return sourcePV_; }

            inline const List<scalar>& Ttable() const
            { return T_table_; }

            inline const List<scalar>& Ytable(const word& sp) const
            { return Y_tables_[sp]; }

            inline label nRGcoeffs() const { return RGcoeff_.size(); }

            inline const List<scalar>& RGtable(const label k) const
            { return RGcoeff_[k]; }

            inline const List<scalar>& LeTable(const word& var) const
            { return Le_tables_[var]; }

            //- The 4th-axis coordinate a legacy 3-D query collapses to.
            inline scalar chi0() const { return chi_axis_[0]; }

            //- Opt-1 (base-blend node interpolation): return the 16 surrounding
            //  manifold node flat indices and their quadrilinear weights for
            //  (Z,gZ,C,chi). nodes[m] and weights[m], m=0..15; sum(weights)=1.
            //  For a 3-D table the chi axis collapses (the chi-high corners
            //  duplicate the chi-low ones with weight 0). Lets a client blend
            //  pre-tabulated node quantities instead of summing over species.
            void interpStencil
            (
                scalar Z, scalar gZ, scalar C, scalar chi,
                label nodes[16], scalar weights[16]
            ) const;

            //- Opt-1: total number of manifold nodes (nZ*nGz*nC*nChi).
            inline label nTot() const
            {
                return nZ_*nGz_*nC_*nChi_;
            }

            //- Opt-1: the flat tabulated mass-fraction node list of a species
            //  (length nTot), used to pre-blend the per-node base thermo.
            inline const List<scalar>& Ynodes(const word& specie) const
            {
                return Y_tables_[specie];
            }

            //- Opt-1: true if the table tabulates this species' mass fraction.
            inline bool hasY(const word& specie) const
            {
                return Y_tables_.found(specie);
            }

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

            //- True if the table carries the Tier-2 real-gas coefficient
            //  fields (RG_*). The solver then enables the per-cell coefficient
            //  lookup instead of the live O(n^2) mixing.
            inline bool hasRealGasCoeffs() const
            { return !RGcoeff_.empty(); }

            //- True if the table carries a tabulated Lewis number Le_<var>
            //  (Tier-4 differential diffusion). The solver then uses a per-cell
            //  Le(Z,gZ,c[,chi]) for that variable instead of a constant.
            inline bool hasLeField(const word& var) const
            { return Le_tables_.found(var); }

            //- True if the table carries a chi_st axis (nChi > 1).
            inline bool hasChi() const { return hasChi_ && !useEnthalpy_; }

            //- True if the 4th axis is an enthalpy-defect axis (non-adiabatic
            //  FPV). The solver then transports total enthalpy and forms the
            //  defect coordinate; the table interpolation is unchanged.
            inline bool useEnthalpy() const { return useEnthalpy_; }

            //- Oxidizer-stream total enthalpy at the flamelet boundary [J/kg].
            inline scalar hOx() const { return hOx_; }

            //- Fuel-stream total enthalpy at the flamelet boundary [J/kg].
            inline scalar hFuel() const { return hFuel_; }

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
