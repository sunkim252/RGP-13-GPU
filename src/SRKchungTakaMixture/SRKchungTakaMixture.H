/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2023 OpenFOAM Foundation
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
    Foam::SRKchungTakaMixture

Description
    Multicomponent mixture class with real-gas mixing rules:
        EoS:       SRK (Soave-Redlich-Kwong)
        Thermo:    janafThermo
        Transport: Chung et al. (1988) for viscosity/conductivity,
                   Fuller/Takahashi for binary diffusion coefficients.

    Pre-computes pair-interaction matrices at construction time for
    efficient per-cell evaluation.

    Ported from realFluidFoam-8 SRKchungTakaMixture to the OpenFOAM-13
    mixture interface (scalarFieldListSlice-based composition).

    Original authors:
        Nam Danh Nguyen - PhD.st
        Advisor: Prof. Chun Sang Yoo
        Combustion & Propulsion Lab - Dept. of Mech. Engineering
        Ulsan Institute of Science and Technology (UNIST) - Korea

SourceFiles
    SRKchungTakaMixture.C

\*---------------------------------------------------------------------------*/

#ifndef SRKchungTakaMixture_H
#define SRKchungTakaMixture_H

#include "multicomponentMixture.H"
#include "FieldListSlice.H"
#include "Switch.H"
#include "PtrList.H"
#include "tabulatedRealGasMixture.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class SRKchungTakaMixture Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class SRKchungTakaMixture
:
    public multicomponentMixture<ThermoType>,
    public tabulatedRealGasMixture
{
public:

    // Public Typedefs

        //- Mixing type for thermodynamic properties
        typedef ThermoType thermoMixtureType;

        //- Mixing type for transport properties
        typedef ThermoType transportMixtureType;


private:

    // Private Data

        //- Temporary storage for the cell/face mixture thermo data
        mutable thermoMixtureType mixture_;

        //- Number of species
        label numberOfSpecies_;


    // Pre-computed matrices for real-fluid mixing rules

        //- List of species molecular weights
        List<scalar> ListW_;

        //- For SRK equation of state
        List<scalar> BM_;
        List<List<scalar>> COEF1_;
        List<List<scalar>> COEF2_;
        List<List<scalar>> COEF3_;

        //- Per-species Peneloux volume shift [m^3/kmol].
        //  Combined linearly on a mole-fraction basis for mixtures.
        List<scalar> CM_;

        //- Binary interaction matrix k_ij applied to the SRK a_m cross
        //  terms as (1 - k_ij). Diagonal = 0. Parsed from an optional
        //  top-level "binaryInteraction" subDict at construction time.
        //  Reference: Reid, Prausnitz & Poling (1987) Ch. 4; Oefelein
        //  (AIAA 2019-0634) Eqs. (24)-(25).
        List<List<scalar>> KIJ_;

        //- For viscosity and thermal conductivity (Chung's model)
        List<List<scalar>> SIGMA3M_;
        List<List<scalar>> EPSILONKM0_;
        List<List<scalar>> OMEGAM0_;
        List<List<scalar>> MM0_;
        List<List<scalar>> MIUIM0_;
        List<List<scalar>> KAPPAIM_;

        //- For mass diffusivity (Takahashi's correction)
        List<scalar> ListTc_;
        List<scalar> ListPc_;
        List<List<scalar>> MMD_;
        List<List<scalar>> SIGMD_;

        //- Tier-0: build/store the O(n^2) Takahashi binary-diffusion matrices
        //  (Tcmd/Pcmd) only when species diffusivity (chungTransport::Dimix)
        //  is actually consumed. FPV transports control variables (Z,C), not
        //  per-species YEqns, and never calls Dimix, so 'speciesDiffusion off'
        //  in physicalProperties skips this dead per-cell O(n^2) work. mu/kappa
        //  are unaffected (they use only the scalar Chung params + Xmd_).
        //  Default true preserves multicomponentFluid behaviour.
        Switch computeSpeciesDiffusion_;


    // Tier-2 tabulated real-gas coefficients (FPV/FGM manifold lookup)

        //- When true, calcMixture() replaces the per-cell O(n^2)
        //  calculateRealGas() pair sum with a lookup of the 13 manifold
        //  coefficients from coeffFields_, for internal-cell composition
        //  slices only. Set/cleared via enable/disableCoeffTabulation().
        mutable bool useTabulatedCoeffs_;

        //- Internal field of the reference (species 0) mass-fraction field.
        //  A composition slice whose &Y[0] lies within this field's contiguous
        //  storage addresses an internal cell (celli recovered by offset; where
        //  coeffFields_ are valid); anything else (patch faces, foreign field
        //  sets) falls back to live mixing.
        mutable const scalarField* refInternalField_;

        //- Internal coefficient fields filled from the table by the solver,
        //  in coeffNames() order (length nCoeffs_ when tabulation is active).
        mutable List<const scalarField*> coeffFields_;

        //- Remaining tabulated-vs-live consistency prints (decremented in
        //  calcMixture); reset on enableCoeffTabulation().
        mutable label coeffDiagCount_;

        //- Opt-2 (patch-face tabulation): species-0 mass-fraction boundary
        //  field per patch. A composition slice whose &Y[0] lies inside one of
        //  these contiguous patch fields addresses that patch face (facei by
        //  offset), so its coefficients are looked up from patchCoeffFields_
        //  instead of rebuilt by the O(n^2) live mixing. Empty -> patch faces
        //  fall back to live (pre-Opt-2 behaviour).
        mutable List<const scalarField*> patchRefFields_;

        //- Opt-2: per-patch, per-coefficient boundary fields filled from the
        //  table by the solver (outer index = patch, inner = coeffNames()
        //  order, length nCoeffs_).
        mutable List<List<const scalarField*>> patchCoeffFields_;


    // Opt-1 base-thermo blend node interpolation

        //- Pre-blended base thermo at each manifold node (length nTot when
        //  active). Built once at enable from the table node mass fractions, so
        //  calcMixture interpolates the 16 bracketing corner mixtures instead of
        //  summing the 106-species blend per internal cell. Bit-consistent: by
        //  linearity sum_corner w*nodeMix == sum_species Y*thermo (FP order
        //  apart), the same identity Tier-2 uses for its coefficients.
        mutable PtrList<thermoMixtureType> nodeMixtures_;

        //- Opt-1 active flag.
        mutable bool useBaseBlendTab_;

        //- Opt-1: the table (16-node stencil) and the manifold-coordinate
        //  internal fields used to locate the stencil for a recovered cell.
        mutable const FGMTable* fgmTablePtr_;
        mutable const scalarField* ZfieldPtr_;
        mutable const scalarField* gZfieldPtr_;
        mutable const scalarField* CfieldPtr_;
        mutable const scalarField* chiFieldPtr_;


    // Private Member Functions

        //- Convert a mass-fraction composition (slice or list) to mole
        //  fractions X and a non-negative mass-fraction copy Yl, using the
        //  exact same arithmetic as the live calcMixture() path so the
        //  tabulated coefficients stay bit-consistent.
        template<class CompType>
        void compositionToX
        (
            const CompType& Y,
            List<scalar>& X,
            List<scalar>& Yl
        ) const;

        //- Calculate real-gas mixture parameters from mole fractions
        void calculateRealGas
        (
            const List<scalar>& X,
            scalar& bM,
            scalar& coef1,
            scalar& coef2,
            scalar& coef3,
            scalar& cM,
            scalar& sigmaM,
            scalar& epsilonkM,
            scalar& MM,
            scalar& VcM,
            scalar& TcM,
            scalar& omegaM,
            scalar& miuiM,
            scalar& kappaiM
        ) const;

        //- Compute mixture from a composition slice (shared logic)
        const thermoMixtureType& calcMixture
        (
            const scalarFieldListSlice& Y
        ) const;


public:

    // Constructors

        //- Construct from a dictionary
        SRKchungTakaMixture(const dictionary&);

        //- Disallow default bitwise copy construction
        SRKchungTakaMixture
        (
            const SRKchungTakaMixture<ThermoType>&
        ) = delete;


    //- Destructor
    virtual ~SRKchungTakaMixture()
    {}


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "SRKchungTakaMixture<" + ThermoType::typeName() + '>';
        }

        //- Return the number of species in the mixture
        inline const label& numberOfSpecies() const
        {
            return numberOfSpecies_;
        }

        //- Return the mixture for thermodynamic properties
        const thermoMixtureType& thermoMixture
        (
            const scalarFieldListSlice&
        ) const;

        //- Return the mixture for transport properties
        const transportMixtureType& transportMixture
        (
            const scalarFieldListSlice&
        ) const;

        //- Return the mixture for transport properties
        const transportMixtureType& transportMixture
        (
            const scalarFieldListSlice&,
            const thermoMixtureType&
        ) const;


    // tabulatedRealGasMixture interface (Tier-2 manifold tabulation)

        //- Compute the nCoeffs_ real-gas mixture coefficients from a mass-
        //  fraction composition Y, in coeffNames() order. Reuses the exact
        //  live mixing rules (compositionToX + calculateRealGas).
        virtual void realGasCoeffs
        (
            const List<scalar>& Y,
            List<scalar>& coeffs
        ) const;

        //- Enable the per-cell tabulated-coefficient lookup.
        virtual void enableCoeffTabulation
        (
            const scalarField& refInternal,
            const List<const scalarField*>& coeffFields
        ) const;

        //- Opt-2: enable the tabulated lookup on patch faces as well.
        virtual void enablePatchCoeffTabulation
        (
            const List<const scalarField*>& patchRefs,
            const List<List<const scalarField*>>& patchCoeffFields
        ) const;

        //- Opt-1: enable the base-thermo blend node interpolation.
        virtual void enableBaseBlendTabulation
        (
            const List<List<scalar>>& nodeY,
            const FGMTable& table,
            const scalarField& Zfield,
            const scalarField& gZfield,
            const scalarField& Cfield,
            const scalarField& chiField
        ) const;

        //- Silent re-arm of the internal-cell coefficient pointers after a mesh
        //  change (redistribution / refinement). Rebinds refInternalField_ and
        //  coeffFields_ without the diagnostic reset or Info print.
        virtual void refreshCoeffFields
        (
            const scalarField& refInternal,
            const List<const scalarField*>& coeffFields
        ) const;

        //- Silent re-arm of the Opt-1 base-blend stencil field pointers after a
        //  mesh change. Re-binds fgmTablePtr_/ZfieldPtr_/gZfieldPtr_/CfieldPtr_/
        //  chiFieldPtr_ only; the mesh-independent node mixtures are NOT rebuilt.
        virtual void refreshBaseBlendFields
        (
            const FGMTable& table,
            const scalarField& Zfield,
            const scalarField& gZfield,
            const scalarField& Cfield,
            const scalarField& chiField
        ) const;

        //- Silent re-arm of the Opt-2 patch-face coefficient pointers after a
        //  mesh change. Rebinds patchRefFields_/patchCoeffFields_ without print.
        virtual void refreshPatchCoeffFields
        (
            const List<const scalarField*>& patchRefs,
            const List<List<const scalarField*>>& patchCoeffFields
        ) const;

        //- Tier-4: differential-diffusion Lewis numbers from a composition at
        //  (p, T) with the exact live transport (Chung mu/kappa/Cp + Takahashi
        //  Dimix). LeZ = Pr; LeC = nu/D_C (PV-mass-weighted Dimix). Builds the
        //  full Takahashi matrices regardless of the speciesDiffusion switch
        //  (utility-only path). Returns false on a degenerate state.
        virtual bool lewisNumbers
        (
            const List<scalar>& Y,
            const scalar p,
            const scalar T,
            const labelList& pvIds,
            scalar& LeZ,
            scalar& LeC
        ) const;

        //- Disable the tabulated lookup (revert to live mixing).
        virtual void disableCoeffTabulation() const;

        //- thermoGPU: export the species-constant / pair-mixing tables as
        //  flat arrays for the CUDA kernel upload (tabulatedRealGasMixture
        //  hook — pure Foam data, no CUDA dependency in this library).
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
        ) const;


    // thermoGPU bridge access (raw coefficient tables, read-only)
    // 생성자에서 빌드된 종-상수/쌍-혼합 테이블을 GPU 커널 업로드용으로
    // 노출한다 (rgpGpuBridge.H). k_ij는 COEF1..3에 이미 반영돼 있음.

        const List<scalar>& gpuListW() const { return ListW_; }
        const List<scalar>& gpuBM() const { return BM_; }
        const List<scalar>& gpuCM() const { return CM_; }
        const List<List<scalar>>& gpuCOEF1() const { return COEF1_; }
        const List<List<scalar>>& gpuCOEF2() const { return COEF2_; }
        const List<List<scalar>>& gpuCOEF3() const { return COEF3_; }
        const List<List<scalar>>& gpuSIGMA3M() const { return SIGMA3M_; }
        const List<List<scalar>>& gpuEPSILONKM0() const { return EPSILONKM0_; }
        const List<List<scalar>>& gpuOMEGAM0() const { return OMEGAM0_; }
        const List<List<scalar>>& gpuMM0() const { return MM0_; }
        const List<List<scalar>>& gpuMIUIM0() const { return MIUIM0_; }
        const List<List<scalar>>& gpuKAPPAIM() const { return KAPPAIM_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#ifdef NoRepository
    #include "SRKchungTakaMixture.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
