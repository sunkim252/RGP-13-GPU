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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class SRKchungTakaMixture Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class SRKchungTakaMixture
:
    public multicomponentMixture<ThermoType>
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


    // Private Member Functions

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
