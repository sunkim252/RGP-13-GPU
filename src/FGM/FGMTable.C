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
#include "tabulatedRealGasMixture.H"

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
    useEnthalpy_(false),
    useDilution_(false),
    hOx_(0),
    hFuel_(0),
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

    // -------- optional enthalpy-defect axis (non-adiabatic FPV, method b) --------
    // The 4th axis machinery is generic; here it carries the enthalpy defect
    // dh [J/kg] instead of chi. Detected by 'fourthAxis enthalpy;'. We reuse the
    // chi_axis_/nChi_ slots so interpolateTable() is unchanged, but flag
    // useEnthalpy_ so the solver forms dh = h - ((1-Z)hOx + Z hFuel) per cell.
    if (found("fourthAxis") && word(lookup("fourthAxis")) == "enthalpy")
    {
        nChi_ = readLabel(lookup("nH"));
        chi_axis_ = List<scalar>(lookup("enthalpy"));
        if (chi_axis_.size() != nChi_)
        {
            FatalErrorInFunction
                << "enthalpy axis size " << chi_axis_.size()
                << " != nH = " << nChi_ << exit(FatalError);
        }
        hOx_   = readScalar(lookup("hOx"));
        hFuel_ = readScalar(lookup("hFuel"));
        hasChi_ = true;          // use the 4-D interpolation path
        useEnthalpy_ = true;
        Info<< "    NON-ADIABATIC FPV: 4th axis = enthalpy defect, nH=" << nChi_
            << "  dh=[" << chi_axis_[0] << "," << chi_axis_[nChi_-1] << "] J/kg"
            << nl << "    hOx=" << hOx_ << " hFuel=" << hFuel_ << " J/kg" << endl;
    }

    // -------- optional steam-dilution axis (H2/O2/H2O power-generation FPV) --
    // 4th axis W = steam mole fraction in the oxidiser stream. Reuses the same
    // chi_axis_/nChi_ 4-D machinery, but the coordinate is a TRANSPORTED
    // conserved dilution scalar W passed directly (no defect subtraction),
    // flagged by 'fourthAxis dilution;' with the axis under 'nW'/'W'.
    if (found("fourthAxis") && word(lookup("fourthAxis")) == "dilution")
    {
        nChi_ = readLabel(lookup("nW"));
        chi_axis_ = List<scalar>(lookup("W"));
        if (chi_axis_.size() != nChi_)
        {
            FatalErrorInFunction
                << "dilution axis size " << chi_axis_.size()
                << " != nW = " << nChi_ << exit(FatalError);
        }
        hasChi_ = true;          // use the 4-D interpolation path
        useDilution_ = true;
        Info<< "    STEAM-DILUTED FPV: 4th axis = oxidiser steam fraction W, "
            << "nW=" << nChi_
            << "  W=[" << chi_axis_[0] << "," << chi_axis_[nChi_-1] << "]"
            << endl;
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

    // -------- optional Tier-2 real-gas mixture coefficient tables --------
    // Pre-tabulated output of SRKchungTakaMixture::calculateRealGas, keyed
    // RG_<name> in tabulatedRealGasMixture::coeffNames() order. All 13 must be
    // present (and full length) for the solver to enable the lookup; otherwise
    // the table is treated as a legacy (no-RG) table and live mixing is used.
    {
        const wordList& cn = tabulatedRealGasMixture::coeffNames();
        bool hasAll = true;
        forAll(cn, k)
        {
            if (!found("RG_" + cn[k])) { hasAll = false; break; }
        }

        if (hasAll)
        {
            RGcoeff_.setSize(cn.size());
            forAll(cn, k)
            {
                const word key("RG_" + cn[k]);
                List<scalar> tbl(lookup(key));
                if (tbl.size() != nTot)
                {
                    FatalErrorInFunction
                        << key << " size " << tbl.size() << " != " << nTot
                        << exit(FatalError);
                }
                RGcoeff_[k] = tbl;
            }
            Info<< "    tabulated real-gas coefficients (Tier-2): "
                << cn << nl << endl;
        }
    }

    // -------- optional Tier-4 differential-diffusion Lewis-number tables --------
    // Le_<var> flat fields (length nTot) for the control variables. When
    // present the solver applies a per-cell Le(Z,gZ,c[,chi]) to that variable's
    // laminar diffusivity (rho*D = mu/Le), the standard differential-diffusion
    // FGM closure, generalising the constant 'Le' sub-dict. Each is independent:
    // a table may carry Le_Z, Le_C, both, or neither.
    {
        const wordList leVars({word("Z"), word("C")});
        forAll(leVars, k)
        {
            const word key("Le_" + leVars[k]);
            if (found(key))
            {
                List<scalar> tbl(lookup(key));
                if (tbl.size() != nTot)
                {
                    FatalErrorInFunction
                        << key << " size " << tbl.size() << " != " << nTot
                        << exit(FatalError);
                }
                Le_tables_.insert(leVars[k], tbl);
            }
        }
        if (!Le_tables_.empty())
        {
            Info<< "    tabulated differential-diffusion Lewis numbers "
                << "(Tier-4): " << Le_tables_.sortedToc() << nl << endl;
        }
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

    // Find j with axis[j-1] < v <= axis[j]. The manifold axes are uniform
    // (Z/gZ/C/dh are linspace), so an initial guess assuming constant
    // spacing lands on the exact cell (or a neighbour), and the two guarded
    // walks below run 0-1 times -> O(1). For a non-uniform axis the guess is
    // just a starting point and the walks recover the correct cell, so this
    // is a drop-in, bit-identical replacement for the old O(n) linear scan.
    const scalar span = axis[n - 1] - axis[0];
    label j = 1;
    if (span > VSMALL)
    {
        j = label((v - axis[0])/span*(n - 1)) + 1;
        j = max(label(1), min(j, n - 1));
    }
    while (j < n - 1 && axis[j] < v)
    {
        j++;
    }
    while (j > 1 && axis[j - 1] >= v)
    {
        j--;
    }

    i = j - 1;
    const scalar d = axis[j] - axis[i];
    w = (d > VSMALL) ? (v - axis[i])/d : 0;
}


void Foam::FGMTable::makeStencil
(
    scalar Z,
    scalar gZ,
    scalar C,
    scalar chi,
    FGMStencil& st
) const
{
    label iZ, iG, iC, iK;

    bracket(Z_axis_,   Z,   iZ, st.wZ);
    bracket(gZ_axis_,  gZ,  iG, st.wG);
    bracket(C_axis_,   C,   iC, st.wC);
    bracket(chi_axis_, chi, iK, st.wK);

    // If chi axis has length 1 the bracket above sets iK=0, wK=0, and we
    // need the iK+1 neighbour to fold to the same slice rather than walk
    // past the end of the array.
    const label iKp = (chi_axis_.size() >= 2) ? (iK + 1) : iK;

    // Corner order = interpolateTable's c0000..c1111 (Z fastest, then gZ,
    // then C; chi-low slice first).
    label n = 0;
    for (label dK = 0; dK < 2; dK++)
    {
        const label k = dK ? iKp : iK;
        for (label dC = 0; dC < 2; dC++)
        {
            for (label dG = 0; dG < 2; dG++)
            {
                for (label dZ = 0; dZ < 2; dZ++)
                {
                    st.idx[n++] = flatIndex(iZ + dZ, iG + dG, iC + dC, k);
                }
            }
        }
    }
}


Foam::scalar Foam::FGMTable::interpolate
(
    const List<scalar>& table,
    const FGMStencil& st
) const
{
    const scalar wZ = st.wZ, wG = st.wG, wC = st.wC, wK = st.wK;

    // Gather the 16 corners (same order as interpolateTable's c0000..c1111).
    const scalar c0000 = table[st.idx[0]];
    const scalar c1000 = table[st.idx[1]];
    const scalar c0100 = table[st.idx[2]];
    const scalar c1100 = table[st.idx[3]];
    const scalar c0010 = table[st.idx[4]];
    const scalar c1010 = table[st.idx[5]];
    const scalar c0110 = table[st.idx[6]];
    const scalar c1110 = table[st.idx[7]];
    const scalar c0001 = table[st.idx[8]];
    const scalar c1001 = table[st.idx[9]];
    const scalar c0101 = table[st.idx[10]];
    const scalar c1101 = table[st.idx[11]];
    const scalar c0011 = table[st.idx[12]];
    const scalar c1011 = table[st.idx[13]];
    const scalar c0111 = table[st.idx[14]];
    const scalar c1111 = table[st.idx[15]];

    // Nested blend VERBATIM from the original interpolateTable -- keeps the
    // floating-point evaluation order, hence bit-identical results.

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


Foam::scalar Foam::FGMTable::interpolateTable
(
    const List<scalar>& table,
    scalar Z,
    scalar gZ,
    scalar C,
    scalar chi
) const
{
    // Single-field query: build the stencil and evaluate (the shared-stencil
    // fast path is makeStencil once + interpolate per field).
    FGMStencil st;
    makeStencil(Z, gZ, C, chi, st);
    return interpolate(table, st);
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


// -------- Tier-2 real-gas coefficient interpolation --------

void Foam::FGMTable::interpolateRealGasCoeffs
(
    scalar Z, scalar gZ, scalar C, scalar chi,
    List<scalar>& coeffs
) const
{
    if (RGcoeff_.empty())
    {
        return;
    }

    coeffs.setSize(RGcoeff_.size());
    forAll(RGcoeff_, k)
    {
        coeffs[k] = interpolateTable(RGcoeff_[k], Z, gZ, C, chi);
    }
}


// -------- Opt-1 base-blend node interpolation stencil --------

void Foam::FGMTable::interpStencil
(
    scalar Z, scalar gZ, scalar C, scalar chi,
    label nodes[16], scalar weights[16]
) const
{
    label iZ, iG, iC, iK;
    scalar wZ, wG, wC, wK;

    bracket(Z_axis_,   Z,   iZ, wZ);
    bracket(gZ_axis_,  gZ,  iG, wG);
    bracket(C_axis_,   C,   iC, wC);
    bracket(chi_axis_, chi, iK, wK);

    // Fold the chi-high neighbour to the same slice for a 3-D table (mirrors
    // interpolateTable), so the chi-high corners are valid indices with the
    // weight (1-wK)/wK distribution -- wK=0 here, so they contribute nothing.
    const label iKp = (chi_axis_.size() >= 2) ? (iK + 1) : iK;

    label m = 0;
    for (label dZ = 0; dZ <= 1; dZ++)
    {
        const scalar fZ = dZ ? wZ : (scalar(1) - wZ);
        for (label dG = 0; dG <= 1; dG++)
        {
            const scalar fG = dG ? wG : (scalar(1) - wG);
            for (label dC = 0; dC <= 1; dC++)
            {
                const scalar fC = dC ? wC : (scalar(1) - wC);
                for (label dK = 0; dK <= 1; dK++)
                {
                    const scalar fK = dK ? wK : (scalar(1) - wK);
                    nodes[m] =
                        flatIndex(iZ + dZ, iG + dG, iC + dC, dK ? iKp : iK);
                    weights[m] = fZ*fG*fC*fK;
                    m++;
                }
            }
        }
    }
}


// -------- Tier-4 differential-diffusion Lewis-number interpolation --------

Foam::scalar Foam::FGMTable::interpolateLe
(
    const word& var,
    scalar Z, scalar gZ, scalar C, scalar chi
) const
{
    if (!Le_tables_.found(var))
    {
        FatalErrorInFunction
            << "Lewis number Le_" << var
            << " is not tabulated in fgmProperties." << nl
            << "Available: " << Le_tables_.sortedToc()
            << exit(FatalError);
    }
    return interpolateTable(Le_tables_[var], Z, gZ, C, chi);
}


// ************************************************************************* //
