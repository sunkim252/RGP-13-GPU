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
#include "gpu/rgpFgmTypes.H"
#include <cstring>
#include <chrono>

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

    armedNCells_(-1),

    tabLewis_(false),

    gpuThermo_(fgmTable_.lookupOrDefault<Switch>("gpuThermo", false)),
    gpuArmed_(false),
    gpuPEqn_(fgmTable_.lookupOrDefault<Switch>("gpuPEqn", false)),
    gpuPEqnCheck_(fgmTable_.lookupOrDefault<Switch>("gpuPEqnCheck", false)),
    gpuPEqnSolver_(fgmTable_.lookupOrDefault<word>("gpuPEqnSolver", "pcg")),
    gpuPEqnPrecon_
    (
        fgmTable_.lookupOrDefault<word>("gpuPEqnPrecon", "jacobi")
    ),
    gpuPEqnNnz_(0),
    gpuZC_(fgmTable_.lookupOrDefault<Switch>("gpuZC", false)),
    gpuZCArmed_(false),
    gpuUEqn_(fgmTable_.lookupOrDefault<Switch>("gpuUEqn", false)),
    gpuPinned_(false),
    gpuMeshCells_(-1),
    gpuMeshFaces_(-1),
    gpuMeshStampTime_(-1),
    gpuPEqnArmed_(false),
    gpuManifold_(fgmTable_.lookupOrDefault<Switch>("gpuManifold", false)),
    gpuManifoldArmed_(false),
    gpuHeMode_(-1),
    gpuNFields_(0),
    thermoTimings_(fgmTable_.lookupOrDefault<Switch>("thermoTimings", false)),

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

    // 병렬: gpuPEqn(pcg)·gpuZC·gpuUEqn 모두 processor 인터페이스
    // halo 교환 + 전역 리덕션으로 지원. devChain(디바이스 상주 pCorr
    // 체인)만 직렬 전용(v1) — 병렬은 rgpUEqnAH 호스트 회수 + CPU fvc
    // 준비체인으로 자동 전환된다 (pressureCorrector의 devChain 게이트).
    // 병렬 amgx: 분산 CSR(AMGX_matrix_upload_distributed) 경로 지원 —
    // 이전의 pcg 강제 강등은 제거 (직렬/병렬 모두 amgx 선택 가능)

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

    // Steam-diluted FPV: allocate and read the transported dilution scalar W.
    // Must exist before the first updateManifold() so the 4th coordinate
    // W = local steam-in-oxidiser fraction is available. chi/enthalpy/3-D
    // tables skip this and leave WPtr_ null.
    if (fgmTable_.useDilution())
    {
        WPtr_.reset
        (
            new volScalarField
            (
                IOobject
                (
                    "W", runTime.name(), mesh,
                    IOobject::MUST_READ, IOobject::AUTO_WRITE
                ),
                mesh
            )
        );
        Info<< "fgmFluid: STEAM-DILUTED FPV -- transporting dilution scalar W"
            << " (conserved; 4th manifold axis queried at local W)"
            << nl << endl;
    }

    // Multivariate convection over the transported scalars (he, Z, C [, h/W]).
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
    // Likewise register the dilution scalar for its shared-scheme convection.
    if (fgmTable_.useDilution())
    {
        fields.add(WPtr_());
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

    // CONSISTENT START (2026-07-02): the base solver constructed the
    // transported density rho_ and the face flux phi_ from the ON-DISK thermo
    // state (species reset to O2, file T), but updateManifold() + the he
    // re-seed above have moved the thermo to the manifold-consistent state.
    // Leaving rho_ at the disk state makes the first steps "rebind" mass:
    // the conservative ddt(rho, X) transport rescales every transported
    // scalar by the density-correction ratio (observed: a Z=1, h=-1.75e6
    // cold kerosene core diluted to Z=0.123 within 25 steps -- Z and h both
    // scaled by exactly the same factor -- while rho corrected itself).
    // Align rho_ and phi_ with the seeded thermo so transport starts
    // mass-consistent and initial scalar cores are preserved.
    rho_ = thermo_.rho();
    rho_.correctBoundaryConditions();
    phi_ = linearInterpolate(rho_*U_) & mesh.Sf();

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
    // the per-cell 106-species base blend with a corner blend of pre-built node
    // mixtures. Valid for an enthalpy-defect (4-D) table too: the tabulated
    // composition Y(Z,gZ,c,dh) is FROZEN across the dh axis (add_enthalpy_axis.py
    // replicates Y along dh), so the per-node base thermo is identical along dh
    // and the corner blend is exact regardless of the 4th-coord field passed to
    // the stencil (chi_st_ here just selects an arbitrary dh slice). Skipped only
    // when the table does not tabulate every mixture species (then the per-node
    // blend would be incomplete).
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

    Info<< "fgmFluid: Tier-2 real-gas coefficient tabulation ACTIVE -- "
        << nCoeff << " manifold coefficients (RG_*) looked up per cell; "
        << "the O(n^2) calculateRealGas pair sum is skipped on internal cells"
        << nl << endl;

    // Record the cell count the lookup was armed for. updateManifold() re-arms
    // the cached mixture field pointers whenever mesh.nCells() later differs
    // from this (a runtime redistribution / AMR refinement reallocates the
    // fields, invalidating the address-range cell recovery in calcMixture).
    armedNCells_ = mesh.nCells();
}


void Foam::solvers::fgmFluid::rearmRealGasCoeffTabulation()
{
    // Re-arm the tabulated real-gas coefficient lookup after a mesh-cell-count
    // change (runtime redistribution or AMR refinement). The mixture recovers a
    // cell/face index from the ADDRESS RANGE of the cached field pointers and
    // then reads the per-cell coefficients from them; a redistribution
    // reallocates those fields, so the stale pointers must be refreshed or the
    // recovery yields a wrong in-range index (out-of-bounds read -> SIGSEGV).
    // The manifold node mixtures (nodeMixtures_) are mesh-INDEPENDENT and are
    // deliberately NOT rebuilt here -- only the field pointers are re-bound.
    if (!tabRealGasCoeffs_)
    {
        return;
    }

    const tabulatedRealGasMixture* hook =
        dynamic_cast<const tabulatedRealGasMixture*>(&thermo_);
    if (!hook)
    {
        return;
    }

    const label nCoeff = tabulatedRealGasMixture::nCoeffs_;

    // (a) Ensure the per-cell coefficient fields address the current cell count.
    //     Registered volScalarFields auto-map on a mesh change, so this is
    //     defensive and normally a no-op (values are refilled in updateManifold
    //     immediately after this call).
    forAll(RGcoeffFields_, k)
    {
        if (RGcoeffFields_[k].primitiveField().size() != mesh.nCells())
        {
            RGcoeffFields_[k].primitiveFieldRef().setSize(mesh.nCells());
        }
    }

    // (b) Re-collect the internal coefficient-field pointers and silently refresh
    //     the mixture's internal-cell lookup (refInternalField_ + coeffFields_).
    {
        List<const scalarField*> ptrs(nCoeff);
        forAll(RGcoeffFields_, k)
        {
            ptrs[k] = &RGcoeffFields_[k].primitiveField();
        }
        hook->refreshCoeffFields(Y_[0].primitiveField(), ptrs);
    }

    // (c) Refresh the Opt-1 base-blend stencil field pointers WITHOUT rebuilding
    //     the (mesh-independent) node mixtures.
    hook->refreshBaseBlendFields
    (
        fgmTable_,
        Z_.primitiveField(), gZ_.primitiveField(),
        C_.primitiveField(), chi_st_.primitiveField()
    );

    // (d) Refresh the Opt-2 patch-face pointers, mirroring exactly the initial
    //     arming in setupRealGasCoeffTabulation() (skip Dirichlet-Z inlet patches
    //     -> those keep live mixing; their prescribed pure-stream Y is off the
    //     (Z_b,gZ_b,C_b) manifold point).
    {
        const volScalarField::Boundary& Y0bf = Y_[0].boundaryField();
        const volScalarField::Boundary& Zbf = Z_.boundaryField();
        List<const scalarField*> patchRefs(Y0bf.size());
        List<List<const scalarField*>> patchCoeffs(Y0bf.size());
        forAll(Y0bf, p)
        {
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
        hook->refreshPatchCoeffFields(patchRefs, patchCoeffs);
    }
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
    // Re-arm the Tier-2 tabulated real-gas coefficient lookup if the mesh cell
    // count changed since it was last armed (runtime redistribution or AMR
    // refinement reallocated the fields, invalidating the mixture's cached
    // address-range cell recovery). Must run before any cell mixture is
    // evaluated below. A no-op guard on a mesh that did not change, so the
    // default (non-distributor) path stays bit-identical.
    if (tabRealGasCoeffs_ && armedNCells_ != mesh.nCells())
    {
        rearmRealGasCoeffTabulation();
        armedNCells_ = mesh.nCells();
    }

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
    // Steam-diluted FPV: 4th coordinate = transported dilution scalar W,
    // passed directly (clamped to the tabulated W range), no defect.
    const bool useW = fgmTable_.useDilution();
    const bool use4D = useChi || useH || useW;
    const scalar hOxb   = useH ? fgmTable_.hOx()   : 0;
    const scalar hFuelb = useH ? fgmTable_.hFuel() : 0;
    const scalarField* hcl = useH ? &hPtr_->primitiveField() : nullptr;
    const scalarField* Wcl = useW ? &WPtr_->primitiveField() : nullptr;
    const scalar Wlo = useW ? fgmTable_.chiAxis().first() : 0;
    const scalar Whi = useW ? fgmTable_.chiAxis().last()  : 0;

    // Hoist references to the tabulated species fields AND their flat tables
    // (resolving the by-name HashTable lookup once, not once per cell).
    List<scalarField*> Yref(tabSpecieIDs_.size());
    List<const List<scalar>*> Ytbl(tabSpecieIDs_.size());
    forAll(tabSpecieIDs_, k)
    {
        const label id = tabSpecieIDs_[k];
        Yref[k] = &Y_[id].primitiveFieldRef();
        Ytbl[k] = &fgmTable_.Ytable(thermo_.species()[id]);
    }

    // Hoist the remaining flat tables for the shared-stencil evaluation: all
    // ~120+ fields (source, T, Y_k, RG_*, Le_*) are interpolated at the SAME
    // manifold point per cell, so the 4-axis bracket is built once per cell
    // (makeStencil) and reused -- bit-identical to the per-field lookups.
    const List<scalar>& srcTbl = fgmTable_.sourcePVTable();
    const List<scalar>& Ttbl = fgmTable_.Ttable();
    if (Ttbl.empty())
    {
        FatalErrorInFunction
            << "Temperature is not tabulated in fgmProperties."
            << exit(FatalError);
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

    // Hoist the flat tables for RG_* and Le out of the per-cell loop. RGtable(k)
    // is an array access but was called per-cell-per-k; LeTable("Z")/("C") did a
    // WORD-keyed HashTable lookup PER CELL (word construction + hash) -- lifting
    // both to raw pointers removes that from the millions-of-cells hot loop.
    List<const List<scalar>*> RGtbl(RGref.size());
    forAll(RGtbl, k)
    {
        RGtbl[k] = &fgmTable_.RGtable(k);
    }
    const List<scalar>* LeZtbl = fillLeZ ? &fgmTable_.LeTable("Z") : nullptr;
    const List<scalar>* LeCtbl = fillLeC ? &fgmTable_.LeTable("C") : nullptr;

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

    // fgmGPU: 셀 루프 전체(대수 gZ/chi_st 클로저 + 전 필드 16-코너 보간)를
    // CUDA로 오프로드. 커널(rgpFgmKernels.cu)은 bracket/makeStencil/
    // interpolate의 부동소수 순서까지 1:1 포트이므로 결과가 비트-동일해야
    // 한다. 경계면 처리와 he 재시드는 CPU에 남는다(v1).
    std::chrono::steady_clock::time_point tSec;
    if (thermoTimings_) { tSec = std::chrono::steady_clock::now(); }

    bool gpuDone = false;
    if (gpuManifold_)
    {
        // 스텝 2부터 1회: 모든 grow-only 버퍼가 최종 크기 도달 후 pin —
        // per-corrector D2H 버퍼(gpuRho_.. / gpuPEqnFlux_)는 첫 스텝의
        // pressureCorrector에서 처음 사이징되므로, 켜진 스위치의 버퍼가
        // 전부 채워진 뒤에만 등록 (안 그러면 size-0 스킵이 영구화됨)
        if
        (
            !gpuPinned_ && gpuManifoldArmed_ && gpuFgmOut_.size() > 0
         && (!gpuThermo_ || gpuRho_.size() > 0)
         && (!gpuPEqn_ || gpuPEqnFlux_.size() > 0)
         && (!gpuUEqn_ || gpuUBuf_.size() > 0)
        )
        {
            pinGpuHostBuffers();
        }

        // 1회 아밍: 축 4개 + [sourcePV, T, Y_k.., RG_.., LeZ?, LeC?] 순서로
        // 연접한 테이블을 업로드 (테이블은 런 중 불변; 산포 순서와 동일).
        if (!gpuManifoldArmed_)
        {
            // 멀티-GPU 병렬: 테이블 디바이스 할당 전에 랭크→디바이스
            // 매핑을 먼저 수행 (armGpuThermo와 동일 매핑). 이게 없으면
            // 테이블이 device 0에 올라간 채 다른 랭크의 디바이스에서
            // 참조돼 cudaErrorIllegalAddress가 난다.
            if (gpuSelectDevice() < 0)
            {
                WarningInFunction
                    << "gpuManifold: no usable CUDA device -- "
                    << "falling back to CPU" << endl;
                gpuManifold_ = Switch(false);
            }
        }
        if (gpuManifold_ && !gpuManifoldArmed_)
        {
            DynamicList<const List<scalar>*> tbls;
            tbls.append(&srcTbl);
            tbls.append(&Ttbl);
            forAll(Ytbl, k) { tbls.append(Ytbl[k]); }
            forAll(RGtbl, k) { tbls.append(RGtbl[k]); }
            if (fillLeZ) { tbls.append(LeZtbl); }
            if (fillLeC) { tbls.append(LeCtbl); }

            gpuNFields_ = tbls.size();
            const label nTot = srcTbl.size();
            List<double> flat(gpuNFields_*nTot);
            forAll(tbls, f)
            {
                std::memcpy
                (
                    flat.begin() + f*nTot,
                    tbls[f]->begin(),
                    nTot*sizeof(double)
                );
            }

            const int rc = rgpFgmUpload
            (
                fgmTable_.nZ(), fgmTable_.nGz(),
                fgmTable_.nC(), fgmTable_.nChi(),
                fgmTable_.Zaxis().begin(), fgmTable_.gZaxis().begin(),
                fgmTable_.Caxis().begin(), fgmTable_.chiAxis().begin(),
                gpuNFields_, flat.begin()
            );

            if (rc == 0)
            {
                gpuManifoldArmed_ = true;
                Info<< "fgmFluid: GPU manifold armed -- " << gpuNFields_
                    << " fields x " << nTot << " table entries" << nl << endl;
            }
            else
            {
                WarningInFunction
                    << "GPU manifold table upload failed ("
                    << rgpFgmLastError() << ") -- falling back to CPU"
                    << endl;
                gpuManifold_ = Switch(false);
            }
        }

        if (gpuManifoldArmed_)
        {
            const label nc = gZc.size();
            gpuFgmOut_.setSize(gpuNFields_*nc);

            const int mode4 = !use4D ? 0 : (useH ? 2 : (useW ? 3 : 1));
            const double* hw =
                useH ? hcl->begin() : (useW ? Wcl->begin() : nullptr);

            // D2H 다이어트: gpuThermo 체인이 켜져 있으면 RG_* 내부
            // 필드는 디바이스 SoA에서 직접 소비되고 호스트 내부값은
            // 아무도 읽지 않는다(경계는 interpolateRealGasCoeffs의
            // 별도 경로, IO는 NO_WRITE) — 해당 슬라이스 D2H 생략.
            const bool skipRG = fillRG && gpuThermo_;
            rgpFgmHostCopySkip
            (
                skipRG ? int(2 + Yref.size()) : 0,
                skipRG ? int(RGref.size()) : 0
            );

            const int rc = rgpFgmEvaluate
            (
                nc, mode4,
                Cv_, shape_Zst, chiClampMin_, chiClampMax_,
                sourcePVscale_, fgmTable_.chi0(), hOxb, hFuelb,
                Wlo, Whi,
                Zc.begin(), Cc.begin(), rhoc.begin(),
                Lsqr.begin(), magSqrGradZ.begin(), DeffZ.begin(),
                hw,
                gZc.begin(), chic.begin(), gpuFgmOut_.begin()
            );

            if (rc != 0)
            {
                FatalErrorInFunction
                    << "rgpFgmEvaluate failed: " << rgpFgmLastError()
                    << exit(FatalError);
            }

            // SoA 슬라이스 산포 (필드 순서는 아밍 때의 연접 순서와 동일)
            const double* out = gpuFgmOut_.begin();
            const size_t bytes = nc*sizeof(double);
            label f = 0;
            std::memcpy(srcc.begin(), out + (f++)*nc, bytes);
            std::memcpy(Tc.begin(),   out + (f++)*nc, bytes);
            forAll(Yref, k)
            {
                std::memcpy(Yref[k]->begin(), out + (f++)*nc, bytes);
            }
            forAll(RGref, k)
            {
                // skipRG면 슬라이스가 D2H되지 않았음 — 산포 생략
                if (!skipRG)
                {
                    std::memcpy(RGref[k]->begin(), out + f*nc, bytes);
                }
                f++;
            }
            if (fillLeZ)
            {
                std::memcpy(LeZc->begin(), out + (f++)*nc, bytes);
            }
            if (fillLeC)
            {
                std::memcpy(LeCc->begin(), out + (f++)*nc, bytes);
            }

            gpuDone = true;
        }
    }

    if (!gpuDone)
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
        else if (useW)
        {
            coord4 = max(Wlo, min(Whi, (*Wcl)[celli]));
        }

        // Shared stencil: ALL tabulated fields (source, T, Y_k, RG_*, Le_*)
        // are interpolated at this one manifold point, so the 4-axis bracket
        // and 16 corner indices are built ONCE and reused. A legacy 3-D query
        // collapses the 4th axis exactly as before (coordinate chi_axis_[0]).
        FGMTable::FGMStencil st;
        fgmTable_.makeStencil
        (
            Zcl, gz, Ccl,
            use4D ? coord4 : fgmTable_.chi0(),
            st
        );

        // PV source: tabulated mass-fraction rate [1/s] -> volumetric.
        srcc[celli] =
            sourcePVscale_*rho_l*fgmTable_.interpolate(srcTbl, st);
        Tc[celli] = fgmTable_.interpolate(Ttbl, st);
        forAll(Yref, k)
        {
            (*Yref[k])[celli] = fgmTable_.interpolate(*Ytbl[k], st);
        }

        // Tier-2: per-cell real-gas mixture coefficients.
        if (fillRG)
        {
            forAll(RGref, k)
            {
                (*RGref[k])[celli] =
                    fgmTable_.interpolate(*RGtbl[k], st);
            }
        }

        // Tier-4: per-cell differential-diffusion Lewis numbers.
        if (fillLeZ)
        {
            (*LeZc)[celli] = fgmTable_.interpolate(*LeZtbl, st);
        }
        if (fillLeC)
        {
            (*LeCc)[celli] = fgmTable_.interpolate(*LeCtbl, st);
        }
    }

    if (thermoTimings_)
    {
        Info<< "manifold interp (" << (gpuDone ? "GPU" : "CPU") << ") = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tSec).count()
            << " s" << endl;
        tSec = std::chrono::steady_clock::now();
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
            const scalarField* Wp =
                useW ? &WPtr_->boundaryField()[patchi] : nullptr;
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
                else if (useW)
                {
                    coord4 = max(Wlo, min(Whi, (*Wp)[fi]));
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

    if (thermoTimings_)
    {
        Info<< "manifold boundary = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tSec).count()
            << " s" << endl;
        tSec = std::chrono::steady_clock::now();
    }

    // Re-seed the energy field from (p, T_table) on the manifold composition.
    // he is now a DIAGNOSTIC consistent with the looked-up (T, Y), not a
    // transported variable -- thermo.correct() inverts it straight back to
    // T_table, so the energy state can never drift off the manifold.
    if (gpuManifold_)
    {
        // GPU 배치 ha(p, T_table, Y_table) — CPU he(p,T) 대입의 1:1 대체
        gpuHeReseed();
    }
    else
    {
        thermo_.he() = thermo_.he(thermo_.p(), thermo_.T());
    }

    if (thermoTimings_)
    {
        Info<< "manifold he re-seed = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tSec).count()
            << " s" << endl;
    }
}


void Foam::solvers::fgmFluid::checkGpuConstraints
(
    const word& fieldName,
    const char* gpuSwitch
) const
{
    const Foam::fvConstraints& cs = fvConstraints();
    forAll(cs, i)
    {
        if (!cs[i].constrainsField(fieldName)) continue;

        const word& t = cs[i].type();
        if
        (
            t == "bound" || t == "limitPressure"
         || t == "limitTemperature" || t == "limitMag"
        )
        {
            // 필드-레벨 전용 제약: 행렬을 건드리지 않고 솔브 후
            // constrain(field)로 적용됨 — CPU 경로와 동일하게 동작
            continue;
        }

        FatalErrorInFunction
            << gpuSwitch << " (v1) does not support the matrix-level "
            << "fvConstraint '" << cs[i].name() << "' (type " << t
            << ") on " << fieldName << " -- the GPU-assembled matrix "
            << "would silently solve a different equation"
            << exit(FatalError);
    }
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
