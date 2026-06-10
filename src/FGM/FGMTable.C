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

#include "FGMTable.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::FGMTable::FGMTable
(
    const fvMesh& mesh
)
:
    IOdictionary
    (
        IOobject
        (
            "fgmProperties",
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ_IF_MODIFIED,
            IOobject::NO_WRITE
        )
    ),
    mesh_(mesh),
    nZ_(readLabel(lookup("nZ"))),
    nGz_(readLabel(lookup("nGz"))),
    nC_(readLabel(lookup("nC"))),
    nChi_(1),
    hasChi_(false),
    Z_axis_(lookup("Z")),
    gZ_axis_(lookup("gZ")),
    C_axis_(lookup("C")),
    chi_axis_(1, scalar(0))
{
    Info<< "\nFGM (FPV + beta-PDF) table initialisation" << endl;

    // -------- optional chi (scalar-dissipation) axis --------
    if (found("nChi"))
    {
        nChi_ = readLabel(lookup("nChi"));
        if (nChi_ < 1)
        {
            FatalErrorInFunction
                << "nChi must be >= 1 (read " << nChi_ << ")"
                << exit(FatalError);
        }
        if (nChi_ >= 2 || found("chi"))
        {
            chi_axis_ = List<scalar>(lookup("chi"));
            if (chi_axis_.size() != nChi_)
            {
                FatalErrorInFunction
                    << "chi axis size " << chi_axis_.size()
                    << " != nChi = " << nChi_
                    << exit(FatalError);
            }
            hasChi_ = true;
        }
    }

    Info<< "    axes: nZ=" << nZ_
        << "  nGz=" << nGz_
        << "  nC="  << nC_
        << "  nChi=" << nChi_
        << (hasChi_ ? "  (4-D table)" : "  (3-D legacy table)")
        << endl;

    if (nZ_ < 2 || nGz_ < 2 || nC_ < 2)
    {
        FatalErrorInFunction
            << "Each of Z/gZ/C axes must have at least 2 entries: "
            << "nZ=" << nZ_ << " nGz=" << nGz_ << " nC=" << nC_
            << exit(FatalError);
    }

    if
    (
        Z_axis_.size()   != nZ_
     || gZ_axis_.size()  != nGz_
     || C_axis_.size()   != nC_
    )
    {
        FatalErrorInFunction
            << "FGM axis lengths do not match declared sizes." << nl
            << "Z: "  << Z_axis_.size()  << " (nZ=" << nZ_ << ")" << nl
            << "gZ: " << gZ_axis_.size() << " (nGz=" << nGz_ << ")" << nl
            << "C: "  << C_axis_.size()  << " (nC=" << nC_ << ")"
            << exit(FatalError);
    }

    // -------- main PV source table --------
    sourcePV_ = List<scalar>(lookup("sourcePV"));
    const label nTot = nZ_*nGz_*nC_*nChi_;
    Info<< "    sourcePV entries: " << sourcePV_.size()
        << " (expected " << nTot << ")" << nl << endl;
    if (sourcePV_.size() != nTot)
    {
        FatalErrorInFunction
            << "sourcePV size " << sourcePV_.size()
            << " != nZ*nGz*nC*nChi = " << nTot
            << exit(FatalError);
    }

    // -------- optional temperature table --------
    if (found("T"))
    {
        T_table_ = List<scalar>(lookup("T"));
        if (T_table_.size() != nTot)
        {
            FatalErrorInFunction
                << "T table size " << T_table_.size() << " != " << nTot
                << exit(FatalError);
        }
        Info<< "    tabulated: T" << endl;
    }

    // -------- optional species composition tables --------
    if (found("species"))
    {
        speciesNames_ = wordList(lookup("species"));
        forAll(speciesNames_, i)
        {
            const word key("Y_" + speciesNames_[i]);
            List<scalar> tbl(lookup(key));
            if (tbl.size() != nTot)
            {
                FatalErrorInFunction
                    << key << " size " << tbl.size() << " != " << nTot
                    << exit(FatalError);
            }
            Y_tables_.insert(speciesNames_[i], tbl);
        }
        Info<< "    tabulated species: " << speciesNames_ << nl << endl;
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::FGMTable::~FGMTable()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::FGMTable::bracket
(
    const List<scalar>& axis,
    scalar v,
    label& i,
    scalar& w
) const
{
    const label n = axis.size();

    // Degenerate axis: collapse to that slice (used when nChi=1).
    if (n <= 1)
    {
        i = 0;
        w = 0;
        return;
    }

    if (v <= axis[0])
    {
        i = 0;
        w = 0;
        return;
    }
    if (v >= axis[n - 1])
    {
        i = n - 2;
        w = 1;
        return;
    }

    // Linear scan (axes are small). Find j with axis[j] >= v.
    label j = 1;
    while (j < n - 1 && axis[j] < v)
    {
        j++;
    }

    i = j - 1;
    const scalar d = axis[j] - axis[i];
    w = (d > VSMALL) ? (v - axis[i])/d : 0;
}


Foam::scalar Foam::FGMTable::interpolateTable
(
    const List<scalar>& table,
    scalar Z,
    scalar gZ,
    scalar C,
    scalar chi
) const
{
    label iZ, iG, iC, iK;
    scalar wZ, wG, wC, wK;

    bracket(Z_axis_,   Z,   iZ, wZ);
    bracket(gZ_axis_,  gZ,  iG, wG);
    bracket(C_axis_,   C,   iC, wC);
    bracket(chi_axis_, chi, iK, wK);

    // If chi axis has length 1 the bracket above sets iK=0, wK=0, and we
    // need the iK+1 neighbour to fold to the same slice rather than walk
    // past the end of the array.
    const label iKp = (chi_axis_.size() >= 2) ? (iK + 1) : iK;

    // Sample the 16 surrounding nodes.
    #define _N(dZ, dG, dC, dK) \
        table[flatIndex(iZ + (dZ), iG + (dG), iC + (dC), \
                        (dK) ? iKp : iK)]

    const scalar c0000 = _N(0,0,0,0);
    const scalar c1000 = _N(1,0,0,0);
    const scalar c0100 = _N(0,1,0,0);
    const scalar c1100 = _N(1,1,0,0);
    const scalar c0010 = _N(0,0,1,0);
    const scalar c1010 = _N(1,0,1,0);
    const scalar c0110 = _N(0,1,1,0);
    const scalar c1110 = _N(1,1,1,0);
    const scalar c0001 = _N(0,0,0,1);
    const scalar c1001 = _N(1,0,0,1);
    const scalar c0101 = _N(0,1,0,1);
    const scalar c1101 = _N(1,1,0,1);
    const scalar c0011 = _N(0,0,1,1);
    const scalar c1011 = _N(1,0,1,1);
    const scalar c0111 = _N(0,1,1,1);
    const scalar c1111 = _N(1,1,1,1);

    #undef _N

    // Trilinear in (Z, gZ, C) at chi-low slice
    const scalar a00 = c0000*(1 - wZ) + c1000*wZ;
    const scalar a10 = c0100*(1 - wZ) + c1100*wZ;
    const scalar a01 = c0010*(1 - wZ) + c1010*wZ;
    const scalar a11 = c0110*(1 - wZ) + c1110*wZ;
    const scalar a0  = a00*(1 - wG) + a10*wG;
    const scalar a1  = a01*(1 - wG) + a11*wG;
    const scalar A   = a0 *(1 - wC) + a1 *wC;

    // Trilinear in (Z, gZ, C) at chi-high slice
    const scalar b00 = c0001*(1 - wZ) + c1001*wZ;
    const scalar b10 = c0101*(1 - wZ) + c1101*wZ;
    const scalar b01 = c0011*(1 - wZ) + c1011*wZ;
    const scalar b11 = c0111*(1 - wZ) + c1111*wZ;
    const scalar b0  = b00*(1 - wG) + b10*wG;
    const scalar b1  = b01*(1 - wG) + b11*wG;
    const scalar B   = b0 *(1 - wC) + b1 *wC;

    // Linear in chi
    return A*(1 - wK) + B*wK;
}


// -------- 4-D entry points --------

Foam::scalar Foam::FGMTable::interpolate
(
    scalar Z, scalar gZ, scalar C, scalar chi
) const
{
    return interpolateTable(sourcePV_, Z, gZ, C, chi);
}


Foam::scalar Foam::FGMTable::interpolateT
(
    scalar Z, scalar gZ, scalar C, scalar chi
) const
{
    if (T_table_.empty())
    {
        FatalErrorInFunction
            << "Temperature is not tabulated in fgmProperties."
            << exit(FatalError);
    }
    return interpolateTable(T_table_, Z, gZ, C, chi);
}


Foam::scalar Foam::FGMTable::interpolateY
(
    const word& specie,
    scalar Z, scalar gZ, scalar C, scalar chi
) const
{
    if (!Y_tables_.found(specie))
    {
        FatalErrorInFunction
            << "Species '" << specie
            << "' is not tabulated in fgmProperties." << nl
            << "Available: " << speciesNames_
            << exit(FatalError);
    }
    return interpolateTable(Y_tables_[specie], Z, gZ, C, chi);
}


// -------- 3-D legacy entry points (evaluate at chi_axis_[0]) --------

Foam::scalar Foam::FGMTable::interpolate
(
    scalar Z, scalar gZ, scalar C
) const
{
    return interpolateTable(sourcePV_, Z, gZ, C, chi_axis_[0]);
}


Foam::scalar Foam::FGMTable::interpolateT
(
    scalar Z, scalar gZ, scalar C
) const
{
    if (T_table_.empty())
    {
        FatalErrorInFunction
            << "Temperature is not tabulated in fgmProperties."
            << exit(FatalError);
    }
    return interpolateTable(T_table_, Z, gZ, C, chi_axis_[0]);
}


Foam::scalar Foam::FGMTable::interpolateY
(
    const word& specie,
    scalar Z, scalar gZ, scalar C
) const
{
    if (!Y_tables_.found(specie))
    {
        FatalErrorInFunction
            << "Species '" << specie
            << "' is not tabulated in fgmProperties." << nl
            << "Available: " << speciesNames_
            << exit(FatalError);
    }
    return interpolateTable(Y_tables_[specie], Z, gZ, C, chi_axis_[0]);
}


// ************************************************************************* //
