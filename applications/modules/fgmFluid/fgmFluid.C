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

    chi_st_
    (
        IOobject("chi_st", runTime.name(), mesh, IOobject::NO_READ,
                 IOobject::AUTO_WRITE),
        mesh,
        dimensionedScalar(dimless/dimTime, 0)
    ),

    sourcePV_
    (
        IOobject("FGM:sourcePV", runTime.name(), mesh, IOobject::NO_READ,
                 IOobject::NO_WRITE),
        mesh,
        dimensionedScalar(dimDensity/dimTime, 0)
    ),

    Cv_(fgmTable_.lookupOrDefault<scalar>("Cv", 0.1)),

    Z_st_(fgmTable_.lookupOrDefault<scalar>("Z_st", 0.0625)),

    Sct_(fgmTable_.lookupOrDefault<scalar>("Sct", 0.7)),

    sourcePVscale_(fgmTable_.lookupOrDefault<scalar>("sourcePVscale", 1.0)),

    varModel_(varModel::laminar),

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

    // Auto-select the mixture-fraction variance closure from the momentum-
    // transport "simulationType" so the same FGM table serves LES (subgrid
    // variance) and (U)RANS (full-field variance) runs with no code change.
    // Must be set before the first updateManifold() below.
    {
        const word simType
        (
            momentumTransport().lookupOrDefault<word>
            (
                "simulationType",
                word("laminar")
            )
        );

        if (simType == "LES")
        {
            varModel_ = varModel::les;
        }
        else if (simType == "RAS")
        {
            varModel_ = varModel::ras;
        }
        else
        {
            varModel_ = varModel::laminar;
        }

        Info<< "fgmFluid: variance closure for simulationType " << simType
            << " -> mixing length L = "
            << (
                   varModel_ == varModel::les ? "Delta = V^(1/3)"
                 : varModel_ == varModel::ras ? "k^(3/2)/epsilon"
                 :                              "0 (laminar)"
               )
            << nl << endl;
    }

    // Multivariate convection over the transported scalars (he, Z, C).
    fields.add(thermo.he());
    fields.add(Z_);
    fields.add(C_);

    // Initialise the manifold-derived fields.
    updateManifold();

    // Re-seed the energy field from (p, T) on the JUST-SET manifold
    // composition. The base thermo constructed he() from the ON-DISK
    // composition/T, but updateManifold() has now overwritten the composition
    // with the tabulated Y_k. For a fresh (cold, c~0) start the two agree, but
    // for any lit/burning initial field the manifold products carry a very
    // different formation enthalpy than the (defaultSpecie-lumped) disk
    // composition, leaving he() stale -- the first he->T inversion then fails
    // to converge ("maximum number of iterations exceeded"). Recomputing he
    // here makes a burning initial condition (seed c high) consistent, which
    // is the only way to ignite the manifold (omega_C(c=0)=0 -> a cold seed
    // cannot self-ignite, especially the weaker low-pressure source).
    thermo_.he() = thermo_.he(thermo_.p(), thermo_.T());
    thermo_.correct();

    // [DIAG] Audit the realFluid EOS consistency at the initial state. The
    // stock compressible pEqn assumes rho ~ psi*p (exact for ideal/PR gas);
    // a large rho/(psi*p) deviation flags an inconsistent SRK compressibility.
    {
        const volScalarField rhoF(thermo_.rho());
        const volScalarField& psiF = thermo_.psi();
        const volScalarField& pF = thermo_.p();
        const volScalarField& TF = thermo_.T();
        Info<< "[DIAG] init rho[" << min(rhoF).value() << "," << max(rhoF).value()
            << "] psi[" << min(psiF).value() << "," << max(psiF).value()
            << "] p[" << min(pF).value() << "," << max(pF).value()
            << "] T[" << min(TF).value() << "," << max(TF).value() << "]" << nl;
        const volScalarField ratio(rhoF/(psiF*pF));
        Info<< "[DIAG] init rho/(psi*p) [" << min(ratio).value() << ","
            << max(ratio).value() << "]  (stock PR == 1)" << endl;
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solvers::fgmFluid::~fgmFluid()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

Foam::tmp<Foam::scalarField>
Foam::solvers::fgmFluid::varianceLengthSqr()
{
    // Mixing-length^2 L^2 for the algebraic mixture-fraction variance closure
    //   Zvar = Cv * L^2 * |grad Z~|^2,
    // auto-selected from the momentum-transport simulationType so a single FGM
    // table and coefficient Cv serve both resolution regimes:
    //   les     : L = Delta = V^(1/3)   (filter width; Pierce & Moin 2004)
    //   ras     : L = k^(3/2)/epsilon   (turbulence integral length scale, the
    //             production = dissipation equilibrium of the variance eqn,
    //             giving Zvar with the correct full-field magnitude)
    //   laminar : L = 0                 (no unresolved fluctuation; gZ -> 0)

    tmp<scalarField> tLsqr(new scalarField(mesh.nCells(), scalar(0)));
    scalarField& Lsqr = tLsqr.ref();

    switch (varModel_)
    {
        case varModel::les:
        {
            const scalarField& V = mesh.V();
            forAll(Lsqr, celli)
            {
                // (V^(1/3))^2 = V^(2/3)
                Lsqr[celli] = sqr(cbrt(V[celli]));
            }
            break;
        }

        case varModel::ras:
        {
            const tmp<volScalarField> tk(momentumTransport().k());
            const tmp<volScalarField> tepsilon(momentumTransport().epsilon());
            const scalarField& k = tk();
            const scalarField& epsilon = tepsilon();
            forAll(Lsqr, celli)
            {
                // L^2 = (k^(3/2)/epsilon)^2 = k^3 / epsilon^2
                const scalar kc = max(k[celli], scalar(0));
                const scalar e2 = max(sqr(epsilon[celli]), SMALL);
                Lsqr[celli] = pow3(kc)/e2;
            }
            break;
        }

        case varModel::laminar:
        {
            // Lsqr stays zero -> Zvar = 0 -> gZ = 0.
            break;
        }
    }

    return tLsqr;
}


void Foam::solvers::fgmFluid::updateManifold()
{
    // |grad(Z)|^2 for the algebraic variance closure and for the
    // scalar-dissipation rate.
    const tmp<volScalarField> tmagSqrGradZ(magSqr(fvc::grad(Z_)));
    const scalarField& magSqrGradZ = tmagSqrGradZ();
    const scalarField& rhoc = rho;

    // Variance mixing-length^2 L^2, auto-selected by simulationType
    // (LES filter width / RAS integral length / laminar zero).
    const tmp<scalarField> tLsqr(varianceLengthSqr());
    const scalarField& Lsqr = tLsqr();

    // Effective diffusivity for Z [kg/(m s)] (laminar/Le + turbulent/Sct).
    // Divide by rho cell-by-cell to recover the mass-diffusivity [m^2/s]
    // needed in chi_tilde = 2 D_eff |grad Z|^2.
    const tmp<volScalarField> tDeffZ(Deff("Z"));
    const scalarField& DeffZ = tDeffZ();

    scalarField& gZc  = gZ_.primitiveFieldRef();
    scalarField& chic = chi_st_.primitiveFieldRef();
    scalarField& srcc = sourcePV_.primitiveFieldRef();
    const scalarField& Zc = Z_;
    const scalarField& Cc = C_;

    // Pitsch-Steiner 2000 normalisation, polynomial form f(Z) ≈ Z(1-Z)
    // (Peters & Williams 2000 textbook simplification; identical to the
    // exact erfc^-1-based shape up to a constant at Z = Z_st, no special
    // function call inside the inner loop).
    const scalar shape_Zst = max(Z_st_*(scalar(1) - Z_st_), SMALL);
    const bool useChi = fgmTable_.hasChi();

    // Hoist references to the tabulated species fields
    List<scalarField*> Yref(tabSpecieIDs_.size());
    List<word> Yname(tabSpecieIDs_.size());
    forAll(tabSpecieIDs_, k)
    {
        const label id = tabSpecieIDs_[k];
        Yref[k] = &Y_[id].primitiveFieldRef();
        Yname[k] = thermo_.species()[id];
    }

    // Tabulated temperature: the FPV/FGM thermochemical state (T and the
    // composition) is a FUNCTION of the manifold coordinates (Z, gZ, c, chi)
    // -- it is LOOKED UP here, not advanced by a transported energy equation.
    // This is the standard adiabatic flamelet coupling (cf. flameletFoam /
    // FPVFoam, which transport only Z and c and read T/he from the table) and
    // avoids the transported-enthalpy instability: an EEqn for he is, under
    // fully-compressible acoustics, perturbed off the manifold so the he->T
    // inversion drifts and the flame collapses. Reading T from the table keeps
    // (T, Y) permanently consistent.
    scalarField& Tc = thermo_.T().primitiveFieldRef();

    forAll(gZc, celli)
    {
        const scalar Zcl = max(min(Zc[celli], scalar(1)), scalar(0));
        const scalar Ccl = max(Cc[celli], scalar(0));
        const scalar rho_l = max(rhoc[celli], SMALL);

        // Algebraic mixture-fraction variance + segregation factor; the
        // mixing length L (Lsqr = L^2) is LES/RAS/laminar-aware.
        const scalar Zvar = Cv_*Lsqr[celli]*magSqrGradZ[celli];
        const scalar gz =
            min(max(Zvar/max(Zcl*(scalar(1) - Zcl), SMALL), scalar(0)), scalar(1));
        gZc[celli] = gz;

        // Scalar dissipation rate at Z = Z_st (Pitsch-Steiner 2000).
        //   chi_tilde = 2 D |grad Z|^2 with D = D_eff / rho
        //   chi_st    = chi_tilde * f(Z_st) / f(Z~)
        const scalar D = DeffZ[celli]/rho_l;
        const scalar chiTilde = scalar(2)*D*magSqrGradZ[celli];
        const scalar shape_cell = max(Zcl*(scalar(1) - Zcl), SMALL);
        const scalar chi_st = chiTilde*shape_Zst/shape_cell;
        chic[celli] = chi_st;

        // PV source: tabulated mass-fraction rate [1/s] -> volumetric.
        // The 4-D lookup is used only when the table actually carries a
        // chi axis (FGMTable::hasChi()); otherwise we keep the legacy
        // 3-D interpolation so existing cases still run.
        if (useChi)
        {
            srcc[celli] =
                sourcePVscale_*rho_l*fgmTable_.interpolate(Zcl, gz, Ccl, chi_st);
            Tc[celli] = fgmTable_.interpolateT(Zcl, gz, Ccl, chi_st);
            forAll(Yref, k)
            {
                (*Yref[k])[celli] =
                    fgmTable_.interpolateY(Yname[k], Zcl, gz, Ccl, chi_st);
            }
        }
        else
        {
            srcc[celli] =
                sourcePVscale_*rho_l*fgmTable_.interpolate(Zcl, gz, Ccl);
            Tc[celli] = fgmTable_.interpolateT(Zcl, gz, Ccl);
            forAll(Yref, k)
            {
                (*Yref[k])[celli] =
                    fgmTable_.interpolateY(Yname[k], Zcl, gz, Ccl);
            }
        }
    }

    gZ_.correctBoundaryConditions();
    chi_st_.correctBoundaryConditions();
    forAll(tabSpecieIDs_, k)
    {
        Y_[tabSpecieIDs_[k]].correctBoundaryConditions();
    }

    // Fixed-value T patches (the inlets) keep their prescribed temperature;
    // zeroGradient/outflow patches extrapolate the just-set internal field.
    thermo_.T().correctBoundaryConditions();

    // Re-seed the energy field from (p, T_table) on the manifold composition.
    // he is now a DIAGNOSTIC consistent with the looked-up (T, Y), not a
    // transported variable -- thermo.correct() inverts it straight back to
    // T_table, so the energy state can never drift off the manifold.
    thermo_.he() = thermo_.he(thermo_.p(), thermo_.T());
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
