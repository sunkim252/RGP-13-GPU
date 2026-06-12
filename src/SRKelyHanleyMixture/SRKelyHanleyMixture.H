/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Copyright (C) 2024-2026 RGP-13
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::SRKelyHanleyMixture

Description
    Multicomponent mixture class with real-gas mixing rules tailored to
    the Ely-Hanley extended-corresponding-states transport backend:

        EoS:        SRK (with optional Peneloux volume translation, k_ij)
        Thermo:     janafThermo
        Transport:  Ely-Hanley ECS via methane reference, plus Fuller +
                    Takahashi for binary diffusion coefficients

    Mixture rules follow Oefelein (AIAA 2019-0634):
      - SRK a_m, b_m use the quadratic mixing rules of Eqs. (24)-(25)
        with binary interaction coefficients k_ij (energy) and l_ij
        (volume):
            a_m = SUM_i SUM_j X_i X_j sqrt(a_i a_j) (1 - k_ij)
            b_m = SUM_i SUM_j X_i X_j (1/8)(b_i^1/3 + b_j^1/3)^3
                                            (1 - l_ij)
      - The ECS reducing ratios h_x, f_x of Eqs. (13)-(17) are taken in
        the van-der-Waals one-fluid limit (Leach-Chappelear-Leland
        shape factors phi_i = theta_i = 1 inside the *mixing* rule;
        the full shape factors are still applied per-species inside
        elyHanleyTransport when forming F_mu, F_lam). This collapses
        Eqs. (13)-(17) to
            V_c,m = SUM_i SUM_j X_i X_j V_c,ij
            T_c,m = (1/V_c,m) SUM_i SUM_j X_i X_j sqrt(T_ci T_cj)
                                          (1 - k_ij) V_c,ij
            V_c,ij = (1/8)(V_ci^1/3 + V_cj^1/3)^3 (1 - l_ij)
        which is the standard vdW1f mixing of Reid-Prausnitz-Poling.
      - omega_m uses arithmetic-mean omega_ij = (omega_i + omega_j)/2,
        which is mathematically identical to the linear mole-fraction
        average SUM_i X_i omega_i, so the linear form is retained.
      - M_m is kept as Kay's rule SUM_i X_i W_i (the implicit Eq. (15)
        harmonic-mean Wij is treated as a small correction; standard
        practice in ECS implementations).
      - Z_{c,m} is computed from the mole-weighted (P_{c,m}, V_{c,m},
        T_{c,m}) so the Enskog correction X_eta, X_lam inside
        elyHanleyTransport sees a consistent Z_{c,m}/Z_{c,o} ratio.

    Compared to SRKchungTakaMixture (which uses Chung's Lennard-Jones
    combining rules to land on T_c,m, V_c,m, omega_m), this class
    pre-computes the ECS combining matrices BIJ, VCIJ, TVCIJ from the
    pure-species (T_c, V_c, omega) plus k_ij, l_ij. The SRK a_m
    matrices (COEF1..3, with k_ij baked in) are identical to
    SRKchungTakaMixture so EoS results are bit-for-bit interchangeable
    when l_ij = 0.

SourceFiles
    SRKelyHanleyMixture.C

\*---------------------------------------------------------------------------*/

#ifndef SRKelyHanleyMixture_H
#define SRKelyHanleyMixture_H

#include "multicomponentMixture.H"
#include "FieldListSlice.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class SRKelyHanleyMixture Declaration
\*---------------------------------------------------------------------------*/

template<class ThermoType>
class SRKelyHanleyMixture
:
    public multicomponentMixture<ThermoType>
{
public:

    // Public Typedefs

        typedef ThermoType thermoMixtureType;
        typedef ThermoType transportMixtureType;


private:

    // Private Data

        //- Per-cell scratch mixture
        mutable thermoMixtureType mixture_;

        label numberOfSpecies_;


    // Pure-species data (constant in time)

        List<scalar> ListW_;     // molecular weights [kg/kmol]
        List<scalar> ListTc_;    // critical temperature [K]
        List<scalar> ListPc_;    // critical pressure [Pa]
        List<scalar> ListVc_;    // critical volume [cm^3/mol]
                                 // (rfSpecie native unit; the
                                 // elyHanleyTransport adapter converts
                                 // to m^3/kmol internally)
        List<scalar> ListOmega_; // acentric factor [-]


    // SRK equation-of-state pre-computed quantities (identical to
    // SRKchungTakaMixture)

        List<scalar> BM_;
        List<List<scalar>> COEF1_;
        List<List<scalar>> COEF2_;
        List<List<scalar>> COEF3_;

        //- Per-species Peneloux volume shift (baseline) [m^3/kmol]
        List<scalar> CM_;

        //- Per-species temperature-dependent Peneloux c(T) coefficients
        //  (cq0, cq1, cq2 [m^3/kmol], Tlo, Thi [K]); zero = constant Peneloux.
        List<scalar> CMq0_;
        List<scalar> CMq1_;
        List<scalar> CMq2_;
        List<scalar> CMTlo_;
        List<scalar> CMThi_;

        //- Binary interaction matrix k_ij (1 - k_ij) applied to SRK
        //  cross terms a_ij and to ECS T_c,ij. Diagonal = 0.
        List<List<scalar>> KIJ_;

        //- Binary interaction matrix l_ij (1 - l_ij) applied to the
        //  Lorentz combining rule for SRK b_ij and ECS V_c,ij
        //  (Oefelein 2019 Eqs. 16, 25). Diagonal = 0.
        List<List<scalar>> LIJ_;

        //- Pre-computed pair b_ij = (1/8)(b_i^1/3 + b_j^1/3)^3
        //  (1 - l_ij), for the Oefelein 2019 SRK b_m mixing rule.
        //  Diagonal entries reduce to the pure b_i.
        List<List<scalar>> BIJ_;

        //- Pre-computed pair V_c,ij used both for V_c,m and as the
        //  weight in T_c,m. Same Lorentz form as BIJ_ but expressed
        //  in cm^3/mol (rfSpecie native units).
        List<List<scalar>> VCIJ_;

        //- Pre-computed pair sqrt(T_ci T_cj) (1 - k_ij) * V_c,ij,
        //  used so that T_c,m = (SUM XiXj TVCIJ) / V_c,m (Oefelein
        //  2019 Eqs. 14, 17 in the shape-factor = 1 limit).
        List<List<scalar>> TVCIJ_;


    // Pre-computed matrices for Fuller / Takahashi binary diffusion
    // (identical to SRKchungTakaMixture)

        List<List<scalar>> MMD_;
        List<List<scalar>> SIGMD_;


    // Private Member Functions

        //- Compute SRK mixture parameters (a_m via three coefficients,
        //  b_m, Peneloux c_m) and Ely-Hanley mixture critical state
        //  (T_c,m, V_c,m, M_m, omega_m, Z_c,m) from mole fractions
        //  using the Oefelein 2019 mixing rules: quadratic Lorentz
        //  for b_m, V_c,m and quadratic vdW1f for T_c,m.
        void calculateRealGas
        (
            const List<scalar>& X,
            scalar& bM,
            scalar& coef1,
            scalar& coef2,
            scalar& coef3,
            scalar& cM,
            scalar& MM,
            scalar& VcM,
            scalar& TcM,
            scalar& omegaM,
            scalar& ZcM
        ) const;

        //- Build the per-cell mixture object
        const thermoMixtureType& calcMixture
        (
            const scalarFieldListSlice& Y
        ) const;


public:

    // Constructors

        SRKelyHanleyMixture(const dictionary&);

        SRKelyHanleyMixture(const SRKelyHanleyMixture<ThermoType>&) = delete;


    //- Destructor
    virtual ~SRKelyHanleyMixture()
    {}


    // Member Functions

        static word typeName()
        {
            return "SRKelyHanleyMixture<" + ThermoType::typeName() + '>';
        }

        inline const label& numberOfSpecies() const
        {
            return numberOfSpecies_;
        }

        const thermoMixtureType& thermoMixture
        (
            const scalarFieldListSlice&
        ) const;

        const transportMixtureType& transportMixture
        (
            const scalarFieldListSlice&
        ) const;

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
    #include "SRKelyHanleyMixture.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
