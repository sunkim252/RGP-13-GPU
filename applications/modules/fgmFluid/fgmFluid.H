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

Class
    Foam::solvers::fgmFluid

Description
    Solver module for steady or transient turbulent flow of compressible
    multicomponent fluids with optional mesh motion and change.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient and steady simulations.

    Optional fvModels and fvConstraints are provided to enhance the simulation
    in many ways including adding various sources, chemical reactions,
    combustion, Lagrangian particles, radiation, surface film etc. and
    constraining or limiting the solution.

    Reference:
    \verbatim
        Greenshields, C. J., & Weller, H. G. (2022).
        Notes on Computational Fluid Dynamics: General Principles.
        CFD Direct Ltd.: Reading, UK.
    \endverbatim

SourceFiles
    fgmFluid.C

See also
    Foam::solvers::fluidSolver
    Foam::solvers::isothermalFluid

\*---------------------------------------------------------------------------*/

#ifndef fgmFluid_H
#define fgmFluid_H

#include "isothermalFluid.H"
#include "fluidMulticomponentThermo.H"
#include "fluidMulticomponentThermophysicalTransportModel.H"
#include "multivariateScheme.H"
#include "FGMTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                          Class fgmFluid Declaration
\*---------------------------------------------------------------------------*/

class fgmFluid
:
    public isothermalFluid
{

protected:

    // Thermophysical properties

        fluidMulticomponentThermo& thermo_;


    // Composition

        PtrList<volScalarField>& Y_;


    // FGM lookup table and control variables

        //- FGM (FPV) lookup table: (Z~, gZ, C~) -> omega_C, T, Y_k
        FGMTable fgmTable_;

        //- Mixture fraction Z~ (transported, no source)
        volScalarField Z_;

        //- Progress variable C~ (transported, source = rho*omega_C)
        volScalarField C_;

        //- Mixture-fraction segregation factor gZ = Zvar/(Z~(1-Z~)) (derived)
        volScalarField gZ_;

        //- Scalar dissipation rate at Z = Z_st [1/s] (derived per cell from
        //  Pitsch-Steiner 2000: chi_st = chi_tilde * f(Z_st)/f(Z~), with
        //  chi_tilde = 2 D_eff |grad Z|^2 and the textbook polynomial
        //  shape f(Z) = Z(1-Z) used in place of exp(-2[erfc^-1(2Z)]^2)).
        volScalarField chi_st_;

        //- Cached volumetric PV source rho*omega_C [kg/m^3/s]
        volScalarField sourcePV_;

        //- Total (absolute) enthalpy h [J/kg], transported ONLY for the
        //  non-adiabatic FPV table (method b). It is a CONSERVED scalar (no
        //  reaction source: total enthalpy is invariant under reaction); the
        //  manifold is then queried at the enthalpy defect
        //  dh = h - ((1-Z) hOx + Z hFuel). Null for adiabatic (chi/3-D) tables.
        //  Crucially T is still READ FROM THE TABLE (T(Z,gZ,c,dh)) -- h is a
        //  coordinate, not inverted to T -- so the he<->T drift instability that
        //  forbids a transported energy equation does NOT apply here.
        autoPtr<volScalarField> hPtr_;

        //- Steam-dilution coordinate W [-], transported ONLY for a
        //  steam-diluted (4-D dilution) table. Conserved scalar (no reaction
        //  source: injected steam is inert as a stream tracer); the manifold's
        //  4th axis is queried directly at W = local steam-in-oxidiser
        //  fraction. Case supplies the field + BCs (steam inlet W=W_ox, O2 and
        //  fuel inlets W=0). Null for chi/enthalpy/3-D tables.
        //  v1 model: W is a single conserved mixing scalar; the mass/mole
        //  basis and the exclusion of combustion-produced H2O are documented
        //  approximations to be refined with a two-tracer reconstruction.
        autoPtr<volScalarField> WPtr_;

        //- LES mixture-fraction variance coefficient (Cv)
        scalar Cv_;

        //- Bilger stoichiometric mixture fraction (mass-based). Default
        //  0.0625 for LOX/Russian rocket kerosene; override in
        //  fgmProperties via 'Z_st'.
        scalar Z_st_;

        //- Per-variable Lewis numbers, keyed by field name ("Z", "C", and
        //  -- in the future differential-diffusion extension -- per species).
        //  Defaults to 1 (unity Lewis) for any name not listed.
        HashTable<scalar> Le_;

        //- Turbulent Schmidt number for the control-variable subgrid flux
        scalar Sct_;

        //- Progress-variable source scale (default 1). Diagnostic/tuning knob
        //  for the source-vs-diffusion balance of the burning branch: at low
        //  pressure the tabulated omega_C is ~p^2 weaker and the FGM burning
        //  branch can become unstable (Wang-2015 10 atm collapse). Set via
        //  'sourcePVscale' in fgmProperties.
        scalar sourcePVscale_;

        //- RUFPV-style limiting of the scalar-dissipation table coordinate
        //  (Gierth et al. 2022; Ihme & Pitsch). For a chi-axis (UFPV) table the
        //  per-cell chi_st = 2 D |grad Z|^2 f(Z_st)/f(Z~) is clamped to
        //  [chiClampMin_, chiClampMax_] BEFORE the source lookup. This breaks the
        //  destabilising instantaneous-|grad Z|^2 feedback (cooling -> grad Z
        //  relaxes -> chi drops -> tabulated source drops -> further cooling ->
        //  spurious cold collapse): the floor keeps the source on the burning
        //  branch; the cap (= chi_st,ign) excludes the extinguished branch.
        //  Defaults (0, GREAT) = no clamp -> legacy behaviour. Set via
        //  'chiClampMin'/'chiClampMax' in fgmProperties.
        scalar chiClampMin_;
        scalar chiClampMax_;

        //- Mixture-fraction variance closure regime, auto-selected from the
        //  momentum-transport "simulationType":
        //    les     -> mixing length L = Delta = V^(1/3)   (subgrid filter)
        //    ras     -> mixing length L = k^(3/2)/epsilon   (integral scale)
        //    laminar -> L = 0 (no unresolved variance; gZ -> 0)
        enum class varModel { laminar, ras, les };
        varModel varModel_;

        //- Indices of thermo species that are tabulated as Y_<sp>
        labelList tabSpecieIDs_;

        //- Tier-2: per-cell tabulated real-gas mixture coefficients (RG_*),
        //  filled from fgmTable_ each updateManifold and looked up by the
        //  SRKchungTaka mixture in place of the live O(n^2) pair mixing.
        //  Empty unless the table carries the RG_* fields, the mixture
        //  supports the hook, and 'tabulatedThermoCoeffs' is not disabled.
        PtrList<volScalarField> RGcoeffFields_;

        //- True when the tabulated-coefficient lookup is active.
        bool tabRealGasCoeffs_;

        //- Reusable per-cell scratch for the interpolated RG coefficients.
        List<scalar> RGcoeffBuf_;

        //- Mesh cell count for which the tabulated real-gas coefficient lookup
        //  was last armed. When it changes (a runtime redistribution or an AMR
        //  refinement move cells between processors / reallocate the coefficient
        //  and control-variable fields), the cached mixture field pointers --
        //  used to recover a cell index by address range -- are stale;
        //  updateManifold() then re-arms them via rearmRealGasCoeffTabulation()
        //  before any cell mixture is evaluated. -1 until the first successful
        //  arming (the guard only fires while tabRealGasCoeffs_ is true).
        label armedNCells_;

        //- Tier-4: optional per-cell differential-diffusion Lewis-number fields
        //  for the control variables Z and C, filled from fgmTable_ each
        //  updateManifold. Null unless the table carries the corresponding
        //  Le_<var> field; when present, Deff(var) uses the local
        //  Le(Z,gZ,c[,chi]) in place of the constant Le_ scalar. This is the
        //  standard differential-diffusion FGM closure (Le baked into the
        //  manifold from the real-fluid flamelet transport).
        autoPtr<volScalarField> LeZField_;
        autoPtr<volScalarField> LeCField_;

        //- True when at least one tabulated Lewis-number field is active.
        bool tabLewis_;

        //- thermoGPU: offload the per-cell property refresh (the
        //  RhoFluidThermo::calculate() loop -- Cp/Cv/psi/rho/mu/kappa at the
        //  tabulated (p, T, Y)) to the CUDA kernels. Opt-in via
        //  'gpuThermo on;' in fgmProperties; requires the SRKchungTaka
        //  (Chung) mixture and a visible CUDA device.
        Switch gpuThermo_;

        //- True once the GPU tables have been uploaded (lazy, first call).
        bool gpuArmed_;

        //- pEqnGPU: correctPressurePEP의 압력 방정식을 fvMatrix 우회로
        //  GPU에서 조립+Jacobi-PCG 솔브+플럭스 추출. 'gpuPEqn on;'
        //  (fgmProperties). 'Gauss linear orthogonal' laplacian 전용.
        Switch gpuPEqn_;

        //- pEqnGPU 검증 모드: CPU fvMatrix와 계수/플럭스 대조 출력
        Switch gpuPEqnCheck_;

        //- pEqnGPU 메시 업로드 완료 플래그 + 작업 버퍼
        mutable bool gpuPEqnArmed_;
        mutable List<double> gpuPEqnBuf_;   // 경계 연접 (bDiag,bSrc,phiB)
        mutable List<double> gpuPEqnFlux_;  // 내부면 플럭스 out

        //- fgmGPU: updateManifold의 셀 루프(대수 gZ/chi_st + 전 필드
        //  16-코너 보간)를 CUDA로 오프로드. 'gpuManifold on;' (fgmProperties).
        Switch gpuManifold_;

        //- 테이블 업로드 완료 플래그 + 보간 출력 호스트 버퍼
        mutable bool gpuManifoldArmed_;
        mutable label gpuNFields_;
        mutable List<double> gpuFgmOut_;

        //- Print the wall time of the per-step thermo property refresh
        //  (thermo_.correct() or gpuThermoCorrect()). 'thermoTimings on;'
        //  in fgmProperties; benchmarking aid, default off.
        Switch thermoTimings_;

        //- Reusable host-side gather/scatter buffers for the GPU evaluate
        //  (internal cells + all patch faces, grow-only).
        List<double> gpuP_, gpuT_, gpuY_;
        List<double> gpuRho_, gpuMu_, gpuKappa_, gpuCp_, gpuCv_, gpuPsi_;

        //- Multivariate convection field table (he, Z, C)
        multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;


    // Thermophysical transport

        autoPtr<fluidMulticomponentThermophysicalTransportModel>
            thermophysicalTransport;


private:

    // Private Member Functions

        //- Set rDeltaT for LTS
        virtual void setRDeltaT();

        //- Pressure-equilibrium-preserving pressure equation (called by the
        //  overridden pressureCorrector). Replaces the base
        //  isothermalFluid::correctPressure so the transcritical density
        //  contact does not radiate a spurious pressure spike.
        void correctPressurePEP();

        //- Update gZ, look up the manifold, set inactive-species Yi and
        //  the cached PV source from the FGM table.
        void updateManifold();

        //- Tier-2 setup: if the table carries RG_* coefficients and the
        //  mixture supports the tabulatedRealGasMixture hook, allocate the 13
        //  per-cell coefficient fields and enable the mixture lookup. Must run
        //  before the first updateManifold()/thermo correct so the fields are
        //  filled before any cell mixture is evaluated through the lookup.
        void setupRealGasCoeffTabulation();

        //- Re-arm the tabulated real-gas coefficient lookup after a mesh-cell-
        //  count change: refresh the cached mixture field pointers (the internal
        //  coefficient fields, the Opt-1 base-blend stencil fields and the Opt-2
        //  patch fields) to the reallocated fields, WITHOUT rebuilding the mesh-
        //  independent manifold node mixtures. Called from updateManifold() when
        //  mesh.nCells() differs from armedNCells_ (runtime redistribution or AMR
        //  refinement). Bit-identical to the freshly armed state.
        void rearmRealGasCoeffTabulation();

        //- Per-cell tabulated Lewis-number field for a control variable, or
        //  nullptr when the table does not tabulate Le for it (constant Le_
        //  fallback). Tier-4 differential-diffusion hook.
        const volScalarField* leField(const word& var) const;

        //- thermoGPU: one-time arming -- export the mixture tables via the
        //  tabulatedRealGasMixture::gpuThermoTables hook and upload them to
        //  the CUDA device. FatalError when 'gpuThermo on;' cannot be
        //  honoured (unsupported mixture / no device), so a requested GPU
        //  run never silently falls back to the CPU.
        void armGpuThermo();

        //- thermoGPU: replacement for thermo_.correct() -- batch-evaluate
        //  Cp/Cv/psi/rho/mu/kappa at the tabulated (p, T, Y) on the GPU for
        //  the internal cells and all patch faces, and scatter into the
        //  thermo fields. T is NOT re-inverted from he (it is the manifold
        //  temperature set in updateManifold; the CPU inversion just returns
        //  it to Newton tolerance).
        void gpuThermoCorrect();

        //- fgmGPU: replacement for the updateManifold he re-seed
        //  thermo_.he() = thermo_.he(p, T) -- batch-evaluate the absolute
        //  enthalpy ha(p, T_table, Y_table) on the GPU (janaf blend + SRK
        //  mixture departure, same tables as armGpuThermo) for internal
        //  cells and all patch faces. Arms the thermo tables if needed.
        void gpuHeReseed();

        //- Effective (dynamic) diffusivity [kg/m/s] for a transported control
        //  variable, using its Lewis number:
        //    Deff = alpha_laminar/Le + rho*nut/Sct
        //  Le is a per-cell tabulated Le(Z,gZ,c[,chi]) when the table provides
        //  Le_<var> (Tier-4 differential diffusion), else the constant Le_
        //  scalar (default 1). The single extension point for non-unity and
        //  per-species differential diffusion.
        tmp<volScalarField> Deff(const word& var);

        //- Mixture-fraction variance mixing-length squared L^2 [m^2] per cell,
        //  auto-selected from the momentum-transport simulationType (LES filter
        //  width / RAS integral length / laminar zero). Used in the algebraic
        //  variance closure Zvar = Cv * L^2 * |grad Z~|^2.
        tmp<scalarField> varianceLengthSqr();


public:

    // Public Data

        //- Reference to the fluid thermophysical properties
        const fluidMulticomponentThermo& thermo;

        //- Reference to the composition
        const PtrList<volScalarField>& Y;


    //- Runtime type information
    TypeName("fgmFluid");


    // Constructors

        //- Construct from region mesh
        fgmFluid(fvMesh& mesh);

        //- Disallow default bitwise copy construction
        fgmFluid(const fgmFluid&) = delete;


    //- Destructor
    virtual ~fgmFluid();


    // Member Functions

        //- Predict thermophysical transport
        virtual void thermophysicalTransportPredictor();

        //- Construct and solve the energy equation,
        //  convert to temperature
        //  and update thermophysical and transport properties
        virtual void thermophysicalPredictor();

        //- Construct and solve the momentum equation, adding a localized
        //  artificial (shear) viscosity (LADUCoeff) to damp the spurious
        //  velocity overshoot at the transcritical density interface
        virtual void momentumPredictor();

        //- Construct and solve the pressure equation in the PISO loop using a
        //  pressure-equilibrium-preserving (pressure-evolution) formulation
        //  that removes the spurious pressure spike at the transcritical
        //  density contact (see pressureCorrector.C).
        virtual void pressureCorrector();

        //- Correct the thermophysical transport
        virtual void thermophysicalTransportCorrector();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const fgmFluid&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
