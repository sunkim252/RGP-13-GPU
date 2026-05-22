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
    Z_axis_(lookup("Z")),
    gZ_axis_(lookup("gZ")),
    C_axis_(lookup("C")),
    sourcePV_(lookup("sourcePV"))
{
    Info<< "\nFGM (FPV + beta-PDF) table initialisation" << endl;
    Info<< "    axes: nZ=" << nZ_
        << "  nGz=" << nGz_
        << "  nC=" << nC_ << endl;
    Info<< "    sourcePV entries: " << sourcePV_.size()
        << " (expected " << nZ_*nGz_*nC_ << ")" << nl << endl;

    if (nZ_ < 2 || nGz_ < 2 || nC_ < 2)
    {
        FatalErrorInFunction
            << "Each FGM axis must have at least 2 entries: "
            << "nZ=" << nZ_ << " nGz=" << nGz_ << " nC=" << nC_
            << exit(FatalError);
    }

    if
    (
        Z_axis_.size() != nZ_
     || gZ_axis_.size() != nGz_
     || C_axis_.size() != nC_
    )
    {
        FatalErrorInFunction
            << "FGM axis lengths do not match declared sizes." << nl
            << "Z: " << Z_axis_.size() << " (nZ=" << nZ_ << ")" << nl
            << "gZ: " << gZ_axis_.size() << " (nGz=" << nGz_ << ")" << nl
            << "C: " << C_axis_.size() << " (nC=" << nC_ << ")"
            << exit(FatalError);
    }

    const label nTot = nZ_*nGz_*nC_;
    if (sourcePV_.size() != nTot)
    {
        FatalErrorInFunction
            << "sourcePV size " << sourcePV_.size()
            << " != nZ*nGz*nC = " << nTot
            << exit(FatalError);
    }

    // ---- Optional temperature table ----
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

    // ---- Optional species composition tables (for R2 real-fluid coupling) ----
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
    scalar C
) const
{
    label iZ, iG, iC;
    scalar wZ, wG, wC;

    bracket(Z_axis_, Z, iZ, wZ);
    bracket(gZ_axis_, gZ, iG, wG);
    bracket(C_axis_, C, iC, wC);

    // Trilinear interpolation over the 8 surrounding nodes.
    const scalar c000 = table[flatIndex(iZ,   iG,   iC  )];
    const scalar c100 = table[flatIndex(iZ+1, iG,   iC  )];
    const scalar c010 = table[flatIndex(iZ,   iG+1, iC  )];
    const scalar c110 = table[flatIndex(iZ+1, iG+1, iC  )];
    const scalar c001 = table[flatIndex(iZ,   iG,   iC+1)];
    const scalar c101 = table[flatIndex(iZ+1, iG,   iC+1)];
    const scalar c011 = table[flatIndex(iZ,   iG+1, iC+1)];
    const scalar c111 = table[flatIndex(iZ+1, iG+1, iC+1)];

    const scalar c00 = c000*(1 - wZ) + c100*wZ;
    const scalar c10 = c010*(1 - wZ) + c110*wZ;
    const scalar c01 = c001*(1 - wZ) + c101*wZ;
    const scalar c11 = c011*(1 - wZ) + c111*wZ;

    const scalar c0 = c00*(1 - wG) + c10*wG;
    const scalar c1 = c01*(1 - wG) + c11*wG;

    return c0*(1 - wC) + c1*wC;
}


Foam::scalar Foam::FGMTable::interpolate(scalar Z, scalar gZ, scalar C) const
{
    return interpolateTable(sourcePV_, Z, gZ, C);
}


Foam::scalar Foam::FGMTable::interpolateT(scalar Z, scalar gZ, scalar C) const
{
    if (T_table_.empty())
    {
        FatalErrorInFunction
            << "Temperature is not tabulated in fgmProperties."
            << exit(FatalError);
    }
    return interpolateTable(T_table_, Z, gZ, C);
}


Foam::scalar Foam::FGMTable::interpolateY
(
    const word& specie,
    scalar Z,
    scalar gZ,
    scalar C
) const
{
    if (!Y_tables_.found(specie))
    {
        FatalErrorInFunction
            << "Species '" << specie << "' is not tabulated in fgmProperties."
            << "  Available: " << speciesNames_
            << exit(FatalError);
    }
    return interpolateTable(Y_tables_[specie], Z, gZ, C);
}


// ************************************************************************* //
