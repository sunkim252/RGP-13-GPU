/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2026 RGP-13
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::tabulatedRealGasMixture

Description
    Type-agnostic hook on a real-gas multicomponent mixture for the FPV/FGM
    "Tier 2" optimisation: pre-tabulation of the composition-dependent
    real-gas mixture coefficients (the O(n^2) Chung/SRK pair-mixing output).

    The 13 coefficients
        bM, coef1, coef2, coef3, cM,                 (SRK EoS)
        sigmaM, epsilonkM, MM, VcM, TcM, omegaM,     (Chung mu/kappa)
        miuiM, kappaiM
    are pure functions of the mole-fraction composition X = X(Z, gZ, c[, chi]),
    independent of (p, T). They are therefore manifold quantities and can be
    looked up from the FGM table instead of being rebuilt per cell with the
    O(n^2) calculateRealGas() pair sum (the documented ~80% thermo cost).

    This abstract base lets:
      - the fgmTabulateRealGasCoeffs utility compute the coefficients from a
        composition with the REAL mixing rules (realGasCoeffs), guaranteeing
        bit-consistency between the tabulated values and the live path, and
      - the fgmFluid solver enable a per-cell lookup of the coefficients from
        fields it fills from the table (enableCoeffTabulation), without naming
        the concrete (energy-variant) mixture template type.

    The number and order of the coefficients is fixed by nCoeffs()/coeffNames().

\*---------------------------------------------------------------------------*/

#ifndef tabulatedRealGasMixture_H
#define tabulatedRealGasMixture_H

#include "scalar.H"
#include "List.H"
#include "word.H"
#include "scalarField.H"
#include "scalarList.H"
#include "labelList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class FGMTable;

/*---------------------------------------------------------------------------*\
                  Class tabulatedRealGasMixture Declaration
\*---------------------------------------------------------------------------*/

class tabulatedRealGasMixture
{
public:

    //- Number of tabulated real-gas mixture coefficients
    static constexpr label nCoeffs_ = 13;

    //- Canonical names of the tabulated coefficients (table keys are
    //  "RG_" + name). Order MUST match realGasCoeffs() and the lookup in
    //  calcMixture().
    static const wordList& coeffNames()
    {
        static const wordList names
        ({
            "bM", "coef1", "coef2", "coef3", "cM",
            "sigmaM", "epsilonkM", "MM", "VcM", "TcM", "omegaM",
            "miuiM", "kappaiM"
        });
        return names;
    }


    //- Destructor
    virtual ~tabulatedRealGasMixture()
    {}


    // Member Functions

        //- Compute the nCoeffs_ real-gas mixture coefficients from a mass-
        //  fraction composition Y (length = number of species, in the mixture
        //  species order). Uses the exact live mixing rules. coeffs is resized
        //  to nCoeffs_. Used by the tabulation utility.
        virtual void realGasCoeffs
        (
            const List<scalar>& Y,
            List<scalar>& coeffs
        ) const = 0;

        //- Enable the per-cell tabulated-coefficient lookup. refInternal is the
        //  internal field of the reference (species 0) mass-fraction field used
        //  to detect that a composition slice addresses an internal cell (vs a
        //  patch face, which falls back to live mixing). coeffFields holds the
        //  nCoeffs_ internal coefficient fields, in coeffNames() order, that the
        //  solver fills from the table each manifold update.
        virtual void enableCoeffTabulation
        (
            const scalarField& refInternal,
            const List<const scalarField*>& coeffFields
        ) const = 0;

        //- Opt-2: also enable the tabulated lookup on PATCH faces. patchRefs
        //  holds the species-0 mass-fraction boundary field of each patch (used
        //  to detect a patch-face composition slice by address range), and
        //  patchCoeffFields holds, per patch, the nCoeffs_ boundary coefficient
        //  fields the solver fills from the table. Default no-op: a backend that
        //  does not override keeps live mixing on patch faces.
        virtual void enablePatchCoeffTabulation
        (
            const List<const scalarField*>& patchRefs,
            const List<List<const scalarField*>>& patchCoeffFields
        ) const
        {}

        //- Opt-1: enable base-thermo blend node interpolation. nodeY[s][n] is
        //  the tabulated mass fraction of species s at manifold node n; table
        //  supplies the 16-node interpolation stencil; the Z/gZ/C/chi internal
        //  fields locate the stencil per recovered internal cell. Replaces the
        //  per-cell 106-species base blend with a 16-corner node blend. Default
        //  no-op: a backend that does not override keeps the live species blend.
        virtual void enableBaseBlendTabulation
        (
            const List<List<scalar>>& nodeY,
            const FGMTable& table,
            const scalarField& Zfield,
            const scalarField& gZfield,
            const scalarField& Cfield,
            const scalarField& chiField
        ) const
        {}

        //- Silent re-arm of the internal-cell coefficient lookup after a mesh-
        //  cell-count change (AMR refinement / runtime redistribution reallocate
        //  the fields). Same rebinding as enableCoeffTabulation but WITHOUT the
        //  diagnostic-count reset or the Info print, so it can run on every mesh
        //  change without log spam. Default no-op.
        virtual void refreshCoeffFields
        (
            const scalarField& refInternal,
            const List<const scalarField*>& coeffFields
        ) const
        {}

        //- Silent re-arm of the Opt-1 base-blend stencil field pointers after a
        //  mesh change. Re-binds only the table and the Z/gZ/C/chi manifold-
        //  coordinate field pointers; the (mesh-independent) pre-blended node
        //  mixtures are left intact and are NOT rebuilt. Default no-op.
        virtual void refreshBaseBlendFields
        (
            const FGMTable& table,
            const scalarField& Zfield,
            const scalarField& gZfield,
            const scalarField& Cfield,
            const scalarField& chiField
        ) const
        {}

        //- Silent re-arm of the Opt-2 patch-face coefficient pointers after a
        //  mesh change. Same rebinding as enablePatchCoeffTabulation but without
        //  the Info print. Default no-op.
        virtual void refreshPatchCoeffFields
        (
            const List<const scalarField*>& patchRefs,
            const List<List<const scalarField*>>& patchCoeffFields
        ) const
        {}

        //- Tier-4: evaluate the differential-diffusion Lewis numbers of the
        //  control variables from a mass-fraction composition Y at (p, T),
        //  using the EXACT live mixture transport (Chung/Takahashi) so the
        //  tabulated Le divides the SOLVER's own mu back into the solver's
        //  own diffusivities (bit-consistent, unlike an external Cantera
        //  evaluation whose dense-liquid Chung deviates):
        //      LeZ = Pr = mu*Cp/kappa        (Z diffuses like heat: D_Z=alpha)
        //      LeC = nu/D_C,  D_C = PV-mass-weighted mixture-avg Dimix
        //  pvIds lists the progress-variable species (mixture order); if their
        //  local mass sums to ~0, D_C falls back to alpha (LeC = LeZ).
        //  Returns false when the backend does not implement the evaluation
        //  (default), or when the state evaluation is degenerate.
        virtual bool lewisNumbers
        (
            const List<scalar>& Y,
            const scalar p,
            const scalar T,
            const labelList& pvIds,
            scalar& LeZ,
            scalar& LeC
        ) const
        {
            return false;
        }

        //- Root-selection hysteresis support: evaluate BOTH cubic-EOS
        //  branches (liquid Zmin, vapour Zmax) of the mixture at (p,T) for
        //  a mass-fraction composition Y, and return rho, psi=(drho/dp)_T,
        //  Cp and h for EACH branch. Returns false when the state has only
        //  one real root (no branch ambiguity -- nothing to hysteresis
        //  over) or the backend doesn't implement real-fluid EOS branching.
        //  Lets a caller (fgmFluid) pick whichever branch is closer to the
        //  cell's PREVIOUS-timestep density -- a "sticky" root selection
        //  using an actual cubic root at all times (never an interpolated
        //  non-root value, unlike the rootBlendTol logistic blend in
        //  SRKGasI.H, which was found to corrupt the cold-start pressure
        //  field; see PEP and LAD Spike Suppression wiki secs 14-16) --
        //  and then re-derive EVERY property the caller consumes (rho,
        //  psi, Cp, h) from that SAME chosen branch. A partial correction
        //  (rho alone) leaves psi/Cp/h thermodynamically inconsistent with
        //  the swapped rho, which was found to blow up the globally-
        //  coupled PEP pressure equation (wiki sec 18b). needPsi/needCpH
        //  (both default true) let a caller that only needs a subset skip
        //  the rest -- psi requires a 2nd cubic solve (forward FD), Cp/h
        //  require 4 additional log()-based departure evaluations; over
        //  the full (unmasked) domain every corrector, computing all of
        //  them unconditionally was measured to cost ~6x more wall-clock
        //  per iteration than the original rho-only correction (wiki sec
        //  18c). Skipped outputs are left at a safe (never-uninitialised)
        //  floor/ideal-gas value, not garbage.
        virtual bool eosRootBranchProperties
        (
            const List<scalar>& Y,
            const scalar p,
            const scalar T,
            scalar& rhoMin, scalar& rhoMax,
            scalar& psiMin, scalar& psiMax,
            scalar& CpMin,  scalar& CpMax,
            scalar& hMin,   scalar& hMax,
            const bool needPsi = true,
            const bool needCpH = true
        ) const
        {
            return false;
        }

        //- Root-selection hysteresis from ALREADY-KNOWN mixture EOS
        //  coefficients (bM/coef1-3/cM, Peneloux/SRK) and mixture molar
        //  mass Wm -- e.g. read directly from a per-cell coefficient
        //  TABLE (fgmFluid::RGcoeffFields_, populated by
        //  enableCoeffTabulation()) that the rest of the solver's
        //  thermo_.correct() already uses for O(1) lookups. Skips
        //  compositionToX()+calculateRealGas() entirely -- with ~100
        //  species that O(n^2) mixing loop dominates
        //  eosRootBranchProperties()'s cost by orders of magnitude, and
        //  re-deriving coefficients the table has ALREADY computed this
        //  same timestep was found to be the actual bottleneck (not the
        //  O(1) psi/Cp/h work trimmed by eosRootBranchProperties()'s
        //  needPsi/needCpH -- see PEP and LAD Spike Suppression wiki sec
        //  18f). Prefer this over eosRootBranchProperties() whenever the
        //  caller's coefficients are already tabulated.
        virtual bool eosRootBranchPropertiesFromCoeffs
        (
            const scalar bM, const scalar coef1, const scalar coef2,
            const scalar coef3, const scalar cM, const scalar Wm,
            const scalar p, const scalar T,
            scalar& rhoMin, scalar& rhoMax,
            scalar& psiMin, scalar& psiMax,
            const bool needPsi = true
        ) const
        {
            return false;
        }

        //- Disable the tabulated lookup (revert to live O(n^2) mixing).
        virtual void disableCoeffTabulation() const = 0;


    // thermoGPU offload support

        //- Export the species-constant / pair-mixing tables needed by the
        //  thermoGPU CUDA kernels, as flat arrays (pure Foam data — this
        //  interface carries NO CUDA dependency; the fgmFluid solver passes
        //  the arrays to the rgpThermoGPU C ABI). pair holds the 9 [n*n]
        //  row-major matrices in kernel order:
        //    COEF1, COEF2, COEF3, SIGMA3M, EPSILONKM0, OMEGAM0, MM0,
        //    MIUIM0, KAPPAIM.
        //  janafHigh/janafLow are the mass-basis Cp coefficients [7n];
        //  the janaf ranges follow the OF mixing convention (species 0).
        //  Returns false when the backend does not support GPU offload
        //  (default — e.g. the Ely-Hanley mixture, not yet kernelised).
        virtual bool gpuThermoTables
        (
            scalarList& W,
            scalarList& BM,
            scalarList& CM,
            scalarList& janafHigh,
            scalarList& janafLow,
            List<scalarList>& pair,
            scalar& TlowJ,
            scalar& ThighJ,
            scalar& Tcommon,
            bool& stableRoot
        ) const
        {
            return false;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
