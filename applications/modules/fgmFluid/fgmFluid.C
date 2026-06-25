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
#include "tabulatedRealGasMixture.H"
#include "Switch.H"

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

    chiClampMin_(fgmTable_.lookupOrDefault<scalar>("chiClampMin", 0.0)),
    chiClampMax_(fgmTable_.lookupOrDefault<scalar>("chiClampMax", GREAT)),

    varModel_(varModel::laminar),

    tabRealGasCoeffs_(false),

    tabLewis_(false),

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

    // Non-adiabatic FPV (method b): allocate and read the transported total
    // enthalpy h. Must exist before the first updateManifold() so the defect
    // coordinate dh = h - h_ad(Z) is available. Adiabatic (chi/3-D) tables skip
    // this and leave hPtr_ null.
    if (fgmTable_.useEnthalpy())
    {
        hPtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "h", runTime.name(), mesh,
                    IOobject::MUST_READ, IOobject::AUTO_WRITE
                ),
                mesh
            )
        );
        Info<< "fgmFluid: NON-ADIABATIC FPV -- transporting total enthalpy h"
            << " (conserved scalar; T read from 4-D table at dh)" << nl << endl;
    }

    // Multivariate convection over the transported scalars (he, Z, C [, h]).
    fields.add(thermo.he());
    fields.add(Z_);
    fields.add(C_);
    // Register total enthalpy in the multivariate set so mvConvection->fvmDiv
    // (phi, h) in the non-adiabatic hEqn resolves the shared div(phi,Yi_h)
    // scheme (otherwise the field is not found in the interpolation table).
    if (fgmTable_.useEnthalpy())
    {
        fields.add(hPtr_());
    }

    // Tier-4: allocate the per-cell differential-diffusion Lewis-number fields
    // for whichever control variables the table tabulates (Le_Z / Le_C). They
    // are filled from the manifold in updateManifold() and consumed by Deff().
    // Initialised to 1 (unity Lewis) so the first Deff("Z") -- called at the top
    // of updateManifold() before the fill loop -- is well defined; thereafter
    // each field carries the previous manifold update's Le (a one-iteration lag
    // that is immaterial: Le enters only the weak Deff->chi->source coupling and
    // converges over the outer correctors).
    if (fgmTable_.hasLeField("Z"))
    {
        LeZField_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "FGM:Le_Z", runTime.name(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar(dimless, 1)
            )
        );
    }
    if (fgmTable_.hasLeField("C"))
    {
        LeCField_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "FGM:Le_C", runTime.name(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar(dimless, 1)
            )
        );
    }
    tabLewis_ = LeZField_.valid() || LeCField_.valid();
    if (tabLewis_)
    {
        Info<< "fgmFluid: Tier-4 differential-diffusion ACTIVE -- per-cell "
            << "tabulated Lewis numbers ["
            << (LeZField_.valid() ? "Le_Z " : "")
            << (LeCField_.valid() ? "Le_C" : "")
            << "] from the manifold drive Deff (rho*D = mu/Le)" << nl << endl;
    }

    // Tier-2: enable the tabulated real-gas coefficient lookup (allocates the
    // RG_* fields and arms the mixture). Must precede updateManifold() so the
    // fields are filled before the first cell-mixture evaluation uses them.
    setupRealGasCoeffTabulation();

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

void Foam::solvers::fgmFluid::setupRealGasCoeffTabulation()
{
    // Gate 1: the table must carry the pre-tabulated RG_* coefficients.
    if (!fgmTable_.hasRealGasCoeffs())
    {
        return;
    }

    // Gate 2: the case may opt out ('tabulatedThermoCoeffs off').
    const Switch want
    (
        fgmTable_.lookupOrDefault<Switch>("tabulatedThermoCoeffs", true)
    );
    if (!want)
    {
        Info<< "fgmFluid: table carries RG_* coefficients but "
            << "'tabulatedThermoCoeffs off' -> live O(n^2) mixing retained"
            << nl << endl;
        return;
    }

    // Gate 3: the mixture must implement the tabulatedRealGasMixture hook
    // (SRKchungTaka backend; the elyHanley backend does not yet).
    const tabulatedRealGasMixture* hook =
        dynamic_cast<const tabulatedRealGasMixture*>(&thermo_);
    if (!hook)
    {
        Info<< "fgmFluid: table carries RG_* coefficients but the active "
            << "mixture does not support tabulatedRealGasMixture -> live mixing"
            << nl << endl;
        return;
    }

    const label nCoeff = tabulatedRealGasMixture::nCoeffs_;
    const wordList& cn = tabulatedRealGasMixture::coeffNames();

    // Allocate the 13 per-cell coefficient fields (internal scratch: dimless,
    // never read or written to disk) and collect pointers to their internal
    // fields for the mixture lookup.
    RGcoeffFields_.setSize(nCoeff);
    List<const scalarField*> ptrs(nCoeff);
    forAll(RGcoeffFields_, k)
    {
        RGcoeffFields_.set
        (
            k,
            new volScalarField
            (
                IOobject
                (
                    "FGM:RG_" + cn[k],
                    mesh.time().name(),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar(dimless, 0)
            )
        );
        ptrs[k] = &RGcoeffFields_[k].primitiveField();
    }

    RGcoeffBuf_.setSize(nCoeff);

    // Arm the mixture: species-0 internal field is the reference used to
    // detect internal-cell composition slices.
    hook->enableCoeffTabulation(Y_[0].primitiveField(), ptrs);
    tabRealGasCoeffs_ = true;

    // Opt-2: register the per-patch species-0 mass-fraction field (range key)
    // and the per-patch coefficient boundary fields so the lookup covers
    // boundary faces too -- the boundary values are filled in updateManifold().
    {
        const volScalarField::Boundary& Y0bf = Y_[0].boundaryField();
        const volScalarField::Boundary& Zbf = Z_.boundaryField();
        List<const scalarField*> patchRefs(Y0bf.size());
        List<List<const scalarField*>> patchCoeffs(Y0bf.size());
        forAll(Y0bf, p)
        {
            // Consistency gate: only tabulate patches whose composition IS the
            // extrapolated internal manifold state, i.e. the boundary Y equals
            // the manifold Y(Z_b,gZ_b,C_b) (zeroGradient walls / outflow). Skip
            // Dirichlet-Z patches (inlets): their prescribed pure-stream Y sits
            // off the (Z_b,gZ_b,C_b) manifold point once gZ_b develops, so the
            // tabulated coeffs would not match the live mix -> keep them live.
            if (Zbf[p].fixesValue())
            {
                patchRefs[p] = nullptr;
                continue;
            }
            patchRefs[p] = &Y0bf[p];
            patchCoeffs[p].setSize(nCoeff);
            forAll(patchCoeffs[p], k)
            {
                patchCoeffs[p][k] = &RGcoeffFields_[k].boundaryField()[p];
            }
        }
        hook->enablePatchCoeffTabulation(patchRefs, patchCoeffs);
    }

    // Opt-1: arm the base-thermo node interpolation (internal cells). Replaces
    // the per-cell 106-species base blend with a 16-corner blend of pre-built
    // node mixtures. Gated off for an enthalpy-defect table (whose 4th coord is
    // dh, not the chi_st_ field passed here as the stencil's chi coordinate) and
    // when the table does not tabulate every mixture species (then the per-node
    // blend would be incomplete).
    if (!fgmTable_.useEnthalpy())
    {
        const wordList& sp = thermo_.species();
        bool haveAllY = true;
        forAll(sp, s)
        {
            if (!fgmTable_.hasY(sp[s]))
            {
                haveAllY = false;
                break;
            }
        }
        if (haveAllY)
        {
            List<List<scalar>> nodeY(sp.size());
            forAll(sp, s)
            {
                nodeY[s] = fgmTable_.Ynodes(sp[s]);
            }
            hook->enableBaseBlendTabulation
            (
                nodeY, fgmTable_,
                Z_.primitiveField(), gZ_.primitiveField(),
                C_.primitiveField(), chi_st_.primitiveField()
            );
        }
        else
        {
            Info<< "fgmFluid: Opt-1 base-blend node interpolation skipped "
                << "(table does not tabulate every mixture species)"
                << nl << endl;
        }
    }
    else
    {
        Info<< "fgmFluid: Opt-1 base-blend node interpolation skipped "
            << "(enthalpy-defect table)" << nl << endl;
    }

    Info<< "fgmFluid: Tier-2 real-gas coefficient tabulation ACTIVE -- "
        << nCoeff << " manifold coefficients (RG_*) looked up per cell; "
        << "the O(n^2) calculateRealGas pair sum is skipped on internal cells"
        << nl << endl;
}

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

    // Non-adiabatic FPV (method b): the 4-D table is indexed by the enthalpy
    // defect dh = h - h_ad(Z), h_ad(Z) = (1-Z) hOx + Z hFuel (the conserved-
    // scalar mixing line). Same quadrilinear lookup as chi; only the 4th
    // coordinate differs.
    const bool useH = fgmTable_.useEnthalpy();
    const bool use4D = useChi || useH;
    const scalar hOxb   = useH ? fgmTable_.hOx()   : 0;
    const scalar hFuelb = useH ? fgmTable_.hFuel() : 0;
    const scalarField* hcl = useH ? &hPtr_->primitiveField() : nullptr;

    // Hoist references to the tabulated species fields
    List<scalarField*> Yref(tabSpecieIDs_.size());
    List<word> Yname(tabSpecieIDs_.size());
    forAll(tabSpecieIDs_, k)
    {
        const label id = tabSpecieIDs_[k];
        Yref[k] = &Y_[id].primitiveFieldRef();
        Yname[k] = thermo_.species()[id];
    }

    // Hoist references to the Tier-2 real-gas coefficient fields (filled below
    // from the table so the mixture can look them up instead of rebuilding the
    // O(n^2) pair sum per cell).
    const bool fillRG = tabRealGasCoeffs_;
    List<scalarField*> RGref(fillRG ? RGcoeffFields_.size() : 0);
    forAll(RGref, k)
    {
        RGref[k] = &RGcoeffFields_[k].primitiveFieldRef();
    }

    // Hoist references to the Tier-4 Lewis-number fields (filled below from the
    // manifold so Deff uses a per-cell Le(Z,gZ,c[,chi]) instead of a constant).
    const bool fillLeZ = LeZField_.valid();
    const bool fillLeC = LeCField_.valid();
    scalarField* LeZc = fillLeZ ? &LeZField_->primitiveFieldRef() : nullptr;
    scalarField* LeCc = fillLeC ? &LeCField_->primitiveFieldRef() : nullptr;

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
        // RUFPV limiting: clamp chi_st to the burning-branch window before the
        // lookup, breaking the instantaneous-|grad Z|^2 source feedback (the
        // documented cold-branch collapse of naive UFPV). Defaults pass through.
        const scalar chi_st =
            max(chiClampMin_,
                min(chiClampMax_, chiTilde*shape_Zst/shape_cell));
        chic[celli] = chi_st;

        // 4th manifold coordinate: chi_st (UFPV) or enthalpy defect (non-
        // adiabatic). For an enthalpy table dh<0 in the cold preheat pulls T
        // down off the 800 K adiabatic slice and (via the table's ignition
        // gate) suppresses the source -> the cold dense-fluid zone is inert.
        scalar coord4 = chi_st;
        if (useH)
        {
            const scalar hAd = (scalar(1) - Zcl)*hOxb + Zcl*hFuelb;
            coord4 = (*hcl)[celli] - hAd;
        }

        // PV source: tabulated mass-fraction rate [1/s] -> volumetric.
        // The 4-D lookup is used when the table carries a chi OR enthalpy axis;
        // otherwise we keep the legacy 3-D interpolation so existing cases run.
        if (use4D)
        {
            srcc[celli] =
                sourcePVscale_*rho_l*fgmTable_.interpolate(Zcl, gz, Ccl, coord4);
            Tc[celli] = fgmTable_.interpolateT(Zcl, gz, Ccl, coord4);
            forAll(Yref, k)
            {
                (*Yref[k])[celli] =
                    fgmTable_.interpolateY(Yname[k], Zcl, gz, Ccl, coord4);
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

        // Tier-2: per-cell real-gas mixture coefficients. coord4 is the chi/dh
        // coordinate (ignored for a 3-D table, where the chi axis collapses).
        if (fillRG)
        {
            fgmTable_.interpolateRealGasCoeffs
            (
                Zcl, gz, Ccl, coord4, RGcoeffBuf_
            );
            forAll(RGref, k)
            {
                (*RGref[k])[celli] = RGcoeffBuf_[k];
            }
        }

        // Tier-4: per-cell differential-diffusion Lewis numbers. coord4 is the
        // chi/dh coordinate (collapses for a 3-D table).
        if (fillLeZ)
        {
            (*LeZc)[celli] = fgmTable_.interpolateLe("Z", Zcl, gz, Ccl, coord4);
        }
        if (fillLeC)
        {
            (*LeCc)[celli] = fgmTable_.interpolateLe("C", Zcl, gz, Ccl, coord4);
        }
    }

    gZ_.correctBoundaryConditions();
    chi_st_.correctBoundaryConditions();
    if (fillLeZ) { LeZField_->correctBoundaryConditions(); }
    if (fillLeC) { LeCField_->correctBoundaryConditions(); }
    forAll(tabSpecieIDs_, k)
    {
        Y_[tabSpecieIDs_[k]].correctBoundaryConditions();
    }

    // Opt-2: fill the patch-face real-gas coefficients from the manifold so the
    // mixture looks them up on boundary faces too (previously live O(n^2) on
    // every boundary face). The boundary composition coordinates (Z, gZ, C,
    // coord4) were just corrected above; the manifold Y(Z_b,gZ_b,C_b,coord4_b)
    // matches the boundary Y the live path mixes (walls extrapolate the internal
    // manifold state, inlets are pure streams at Z=0/1).
    if (fillRG)
    {
        const volScalarField::Boundary& Zbf   = Z_.boundaryField();
        const volScalarField::Boundary& gZbf  = gZ_.boundaryField();
        const volScalarField::Boundary& Cbf   = C_.boundaryField();
        const volScalarField::Boundary& chibf = chi_st_.boundaryField();
        forAll(Zbf, patchi)
        {
            const fvPatchScalarField& Zp   = Zbf[patchi];
            const fvPatchScalarField& gZp  = gZbf[patchi];
            const fvPatchScalarField& Cp   = Cbf[patchi];
            const fvPatchScalarField& chip = chibf[patchi];
            const scalarField* hp =
                useH ? &hPtr_->boundaryField()[patchi] : nullptr;
            forAll(Zp, fi)
            {
                const scalar Zcl = max(min(Zp[fi], scalar(1)), scalar(0));
                const scalar gz  = min(max(gZp[fi], scalar(0)), scalar(1));
                const scalar Ccl = max(Cp[fi], scalar(0));
                scalar coord4 = chip[fi];
                if (useH)
                {
                    const scalar hAd = (scalar(1) - Zcl)*hOxb + Zcl*hFuelb;
                    coord4 = (*hp)[fi] - hAd;
                }
                fgmTable_.interpolateRealGasCoeffs
                (
                    Zcl, gz, Ccl, coord4, RGcoeffBuf_
                );
                forAll(RGcoeffFields_, k)
                {
                    RGcoeffFields_[k].boundaryFieldRef()[patchi][fi] =
                        RGcoeffBuf_[k];
                }
            }
        }
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


const Foam::volScalarField*
Foam::solvers::fgmFluid::leField(const word& var) const
{
    if (var == "Z" && LeZField_.valid()) { return &LeZField_(); }
    if (var == "C" && LeCField_.valid()) { return &LeCField_(); }
    return nullptr;
}


Foam::tmp<Foam::volScalarField>
Foam::solvers::fgmFluid::Deff(const word& var)
{
    // Turbulent subgrid contribution rho*nut/Sct [kg/m/s].
    const volScalarField turb(rho*momentumTransport().nut()/Sct_);

    // Laminar mass diffusivity rho*D_lam = mu/Le. Tier-4: when the manifold
    // tabulates Le_<var>, use the per-cell Le(Z,gZ,c[,chi]) (differential
    // diffusion baked in from the real-fluid flamelet transport); otherwise
    // fall back to the constant Le_ scalar (default 1, unity Lewis).
    const volScalarField* Lefld = leField(var);
    if (Lefld != nullptr)
    {
        return thermo.mu()/(*Lefld) + turb;
    }

    const scalar Le = Le_.found(var) ? Le_[var] : 1.0;
    return thermo.mu()/Le + turb;
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
