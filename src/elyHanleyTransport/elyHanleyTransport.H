/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Copyright (C) 2024-2025 RGP-13
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

Class
    Foam::elyHanleyTransport

Description
    Real-gas transport model for viscosity and thermal conductivity using
    Ely & Hanley's extended corresponding states (ECS) framework with
    methane as the reference fluid.

    Theory:
      - Ely, J.F. & Hanley, H.J.M., "Prediction of Transport Properties.
        1. Viscosity of Fluids and Mixtures", Ind. Eng. Chem. Fundam.
        20 (1981) 323-332.
      - Ely, J.F. & Hanley, H.J.M., "Prediction of Transport Properties.
        2. Thermal Conductivity of Pure Fluids and Mixtures", Ind. Eng.
        Chem. Fundam. 22 (1983) 90-97.
      - Hanley, H.J.M., McCarty, R.D. & Haynes, W.M., "The viscosity and
        thermal conductivity coefficients of dilute and dense fluid
        methane", J. Phys. Chem. Ref. Data 4 (1975) 135.
      - Leach, J.W., Chappelear, P.S. & Leland, T.W., "Use of molecular
        shape factors in vapor-liquid equilibrium calculations with the
        corresponding states principle", AIChE J. 14 (1968) 568.
      - Mason, E.A. & Monchick, L., "Heat Conductivity of Polyatomic and
        Polar Gases", J. Chem. Phys. 36 (1962) 1622-1639.
      - Huber, M.L., Laesecke, A. & Perkins, R.A., "Model for the Viscosity
        and Thermal Conductivity of Refrigerants", Ind. Eng. Chem. Res.
        42 (2003) 3163-3178.

    Core equations (Oefelein AIAA 2019-0634 Eqs. 39-54):
      mu(rho, T)     = mu_o(rho_o, T_o) * F_mu * X_eta
      lambda(rho, T) = F_lam * X_lam * lambda_o^trans(rho_o, T_o)
                     + lambda_int^target(T)
    where
      rho_o = rho * h_x,  T_o = T / f_x
      F_mu  = (Wx/Wo)^0.5 * f_x^0.5 * h_x^(-2/3)
      F_lam = (Wo/Wx)^0.5 * f_x^0.5 * h_x^(-2/3)
      h_x, f_x built from Leach-Chappelear-Leland shape factors,
      lambda_o^trans(rho_o, T_o) = methane reference *translational*
        contribution (Oefelein Eq. 52: lambda*_o = (15/4)(R_u/W_o) mu*_o,
        plus the HHM 1975 density-dependent translational corrections
        lambda**_o rho_o + Delta_lambda_o), and
      lambda_int^target(T) = 1.32 mu_dilute^target (Cp_ig^target - 5R/2)
        is the Modified Eucken (Mason-Monchick 1962) internal-mode
        contribution evaluated at the *target* fluid's actual T and Cp_ig.
        This split avoids transferring the methane reference's polyatomic
        internal-mode k onto light fluids (e.g. H2, T_c = 33 K) via the
        ECS T-scaling, which previously produced ~3x overprediction.

    Mass diffusivity is left to the Fuller + Takahashi pipeline already
    present in the repository (shared with chungTransport) so that only
    mu and kappa change when this transport model is selected.

    Drop-in replacement for chungTransport in the same template slot.

SourceFiles
    elyHanleyTransportI.H
    elyHanleyTransport.C

\*---------------------------------------------------------------------------*/

#ifndef elyHanleyTransport_H
#define elyHanleyTransport_H

#include "PtrList.H"

namespace Foam
{

template<class Thermo> class elyHanleyTransport;

template<class Thermo>
inline elyHanleyTransport<Thermo> operator+
(
    const elyHanleyTransport<Thermo>&,
    const elyHanleyTransport<Thermo>&
);

template<class Thermo>
inline elyHanleyTransport<Thermo> operator*
(
    const scalar,
    const elyHanleyTransport<Thermo>&
);

template<class Thermo>
Ostream& operator<<
(
    Ostream&,
    const elyHanleyTransport<Thermo>&
);


/*---------------------------------------------------------------------------*\
                      Class elyHanleyTransport Declaration
\*---------------------------------------------------------------------------*/

template<class Thermo>
class elyHanleyTransport
:
    public Thermo
{

protected:

    // Protected data (mixture-averaged quantities cached per cell)

        //- Mixture critical temperature [K]
        scalar TcM_;

        //- Mixture critical volume [m^3/kmol]
        scalar VcM_;

        //- Mixture molecular weight [kg/kmol]
        scalar MM_;

        //- Mixture acentric factor [-]
        scalar omegaM_;

        //- Mixture critical compressibility factor Zc = pc Vc / (R Tc) [-]
        scalar ZcM_;

        //- Reference fluid (methane) critical temperature [K]
        scalar TcO_;

        //- Reference fluid (methane) critical molar volume [m^3/kmol]
        scalar VcO_;

        //- Reference fluid (methane) molecular weight [kg/kmol]
        scalar MO_;

        //- Reference fluid (methane) acentric factor [-]
        scalar omegaO_;

        //- Reference fluid (methane) critical compressibility [-]
        scalar ZcO_;

        //- Members for diffusivity evaluation (Fuller + Takahashi).
        //  Kept for API compatibility with chungTransport.

            List<scalar> Ymd_;
            List<scalar> Xmd_;
            List<List<scalar>> Tcmd_;
            List<List<scalar>> Pcmd_;
            List<List<scalar>> Mmd_;
            List<List<scalar>> sigmd_;

        //- Composition-only Pedersen quantities, precomputed in updateTRANS()
        //  so they are NOT recomputed in pedersenMap() per (p,T) and per
        //  mu()/kappa() call. Pure refactor; results unchanged.
        //    mapTfac_  = Tc0/Tcmix,  mapPfac_ = Pc0_CH4/Pcmix
        //    baseConst_= (Tcmix/Tc0)^(-1/6) (Pcmix/Pc0)^(2/3)
        //    MWpow_    = MWmix^0.5173   (Pedersen alpha exponent on MW)
            scalar mapTfac_;
            scalar mapPfac_;
            scalar baseConst_;
            scalar MWpow_;

        //- Single-entry (p,T) cache of the last pedersenMap() result. The
        //  RhoFluidThermo cell loop evaluates mu(p,T) then kappa(p,T) with the
        //  SAME (p,T) on one transportMixture, so the second call reuses the
        //  first mapping (saves 2 methane SRK cubic solves per cell). Cache is
        //  invalidated (cacheT_ < 0) on every composition change. mutable:
        //  valid for OpenFOAM's serial per-process cell loop; NOT thread-safe
        //  under a shared transport object.
            mutable scalar cacheP_;
            mutable scalar cacheT_;
            mutable scalar cacheTRef_;
            mutable scalar cacheRhoO_;
            mutable scalar cacheBase_;


    // Protected Member Functions

        //- C^1-continuous clamp of x to [lo, hi] with a quadratic
        //  Hermite blend over a transition half-width w at each
        //  boundary. Returns x exactly on [lo + w, hi - w] and
        //  saturates smoothly to lo / hi outside [lo - w, hi + w].
        //
        //  Replaces hard max/min clipping of the reduced volume V_r
        //  and reduced temperature T_r inside phiShape / thetaShape:
        //  the hard clip leaves d(phi)/dT and d(theta)/dT C^0 but
        //  not C^1, which shows up as visible kinks in mu(T) and
        //  kappa(T) plots whenever T_r crosses the clip boundary
        //  (e.g. for H2O at T ~ 0.5 Tc ~ 324 K and T ~ 2 Tc ~ 1294 K).
        inline static scalar softClamp
        (
            const scalar x,
            const scalar lo,
            const scalar hi,
            const scalar w
        );

        //- Leach-Chappelear-Leland shape factor phi(V_r, T_r, omega).
        //  Returns the factor such that h_i = (Vc_i/Vc_o) * phi.
        inline scalar phiShape
        (
            const scalar Vri,
            const scalar Tri,
            const scalar omegai
        ) const;

        //- Leach-Chappelear-Leland shape factor theta(V_r, T_r, omega).
        //  Returns theta such that f_i = (Tc_i/Tc_o) * theta.
        inline scalar thetaShape
        (
            const scalar Vri,
            const scalar Tri,
            const scalar omegai
        ) const;

        //- Methane dilute (zero-density) viscosity mu*_o(T_o) [Pa.s].
        //  Sutherland fit to CoolProp methane in the dilute limit; shared
        //  by muRef (translational+initial-density+dense decomposition)
        //  and by the Modified Eucken internal-mode term in kappa() so
        //  that the ECS-mapped translational mu_o is consistent in both.
        inline scalar muStarO(const scalar TO) const;

        //- Methane reference viscosity mu_o(rho_o, T_o) [Pa.s].
        //  Three-term decomposition per Hanley-McCarty-Haynes 1975:
        //    mu_o = mu_o*(T_o) + mu_o**(T_o) * rho_o + Delta_mu_o(rho_o, T_o)
        //  Per Oefelein (AIAA 2019-0634) Eq. (47) the Enskog correction
        //  X_eta multiplies ONLY the dense correction Delta_mu_o:
        //    mu_o = mu*_o + mu**_o*rho + Delta_mu_o * X_eta
        inline scalar muRef
        (
            const scalar rhoO,
            const scalar TO,
            const scalar X_eta = 1.0
        ) const;

        //- Methane reference *translational* thermal conductivity [W/m/K].
        //  Per Oefelein Eq. (52) the dilute term is the monatomic-Eucken
        //  form lambda*_o = (15/4)(R_u/W_o) mu*_o(T_o); the density
        //  corrections lambda**_o rho_o + Delta_lambda_o (HHM 1975) are
        //  retained in the methane frame and then ECS-scaled by F_lam.
        //  The methane *internal-mode* contribution is intentionally
        //  excluded here and is reintroduced in kappa() via Modified
        //  Eucken on the target fluid's Cp_ig.
        //  Per Oefelein Eq. (51) X_lam multiplies ONLY the dense
        //  correction Delta_lambda_o:
        //    lambda_o = lambda*_o + lambda**_o*rho + Delta_lambda_o * X_lam
        inline scalar lambdaRef
        (
            const scalar rhoO,
            const scalar TO,
            const scalar X_lam = 1.0
        ) const;

        //- Methane reference mass density [kg/m^3] from the SRK EoS at the
        //  ECS-mapped reference state (T_o, P_o).  Replaces the density
        //  scaling rho_o = rho_x * h_x, which for dense fluids maps to an
        //  UNPHYSICAL methane density (e.g. O2 liquid -> rho_o ~ 850 > methane
        //  max ~538) that railed against the rho_o clamp and made the HHM
        //  dense term over-predict mu/kappa by 100s of %.  Solving the
        //  methane SRK cubic at (T_o, P_o = p*h_x/f_x) keeps rho_o physical
        //  (<= M_o/b ~ 538 kg/m^3), eliminating the rail.  Root chosen by
        //  minimum fugacity (stable phase), matching the Pedersen/marlim3
        //  corresponding-states implementation.  Validated O2 mu MAPE
        //  33 -> 3.2 %, H2O 87 -> 36.5 % (test/proto_eosdensity.py).
        inline scalar srkMethaneRho(const scalar T, const scalar P) const;

        //- Pedersen-Fredenslund corresponding-states mapping (Pedersen,
        //  Fredenslund & Thomassen 1989 / PVTSIM / Petrobras marlim3).
        //  Replaces the LCL theta,phi shape factors + Enskog X with the
        //  critical-property-ratio mapping (which aligns the target and
        //  methane critical points by construction, removing the
        //  phase-misalignment 'bump' near the target pseudo-critical) plus
        //  the smooth rotational-coupling alpha(rho_R, MW) correction.
        //  Fills the methane reference state (TRef [K], rhoO [kg/m3] via
        //  srkMethaneRho) and the scaling factor
        //    base = (Tcmix/TcO)^(-1/6) (Pcmix/PcO)^(2/3) (alpha_mix/alpha_0)
        //  with F_eta = base*sqrt(MWmix/MO), F_lam = base*sqrt(MO/MWmix).
        //  MWmix is the Pedersen mixture molar weight.
        inline void pedersenMap
        (
            const scalar p,
            const scalar T,
            scalar& TRef,
            scalar& rhoO,
            scalar& base,
            scalar& MWmix
        ) const;

        //- Recompute the composition-only Pedersen constants (mapTfac_,
        //  mapPfac_, baseConst_, MWpow_) from the current critical-property
        //  data. Called once per composition update (updateTRANS / ctor),
        //  hoisting the O(N^2) combining rule + composition powers out of the
        //  per-(p,T) pedersenMap() hot path. Also invalidates the (p,T) cache.
        inline void updatePedersenConst();

        //- pedersenMap() with a single-entry (p,T) cache (see cache members).
        inline void pedersenMapCached
        (
            const scalar p,
            const scalar T,
            scalar& TRef,
            scalar& rhoO,
            scalar& base,
            scalar& MWmix
        ) const;

        //- Enskog size-difference correction X_eta (Ely 1981 Eq. 53)
        inline scalar enskogXeta
        (
            const scalar Tx,
            const scalar fx,
            const scalar dfxdT
        ) const;

        //- Enskog size-difference correction X_lam (Ely 1981 Eq. 54)
        inline scalar enskogXlam
        (
            const scalar Tx,
            const scalar fx,
            const scalar dfxdT
        ) const;

public:

    // Constructors

        inline elyHanleyTransport
        (
            const Thermo& t,
            const scalar& TcM,
            const scalar& VcM,
            const scalar& MM,
            const scalar& omegaM,
            const scalar& ZcM,
            const List<scalar>& Ymd,
            const List<scalar>& Xmd,
            const List<List<scalar>>& Tcmd,
            const List<List<scalar>>& Pcmd,
            const List<List<scalar>>& Mmd,
            const List<List<scalar>>& sigmd
        );

        inline elyHanleyTransport(const word&, const elyHanleyTransport&);

        elyHanleyTransport(const word& name, const dictionary& dict);

        elyHanleyTransport(const Thermo& t, const dictionary& dict);

        inline autoPtr<elyHanleyTransport> clone() const;


    // Member Functions

        static word typeName()
        {
            return "elyHanley<" + Thermo::typeName() + '>';
        }

        //- Dynamic viscosity [kg/m/s]
        inline scalar mu(const scalar p, const scalar T) const;

        //- Thermal conductivity [W/m/K]
        inline scalar kappa(const scalar p, const scalar T) const;

        //- Takahashi correction coefficient for binary diffusion.
        //  Implemented identically to chungTransport::phi(PRR, TRR) so that
        //  Dimix() remains drop-in compatible.
        inline scalar phi(scalar PRR, scalar TRR) const;

        //- Mass diffusivity of species i (Fuller + Takahashi) [m^2/s]
        inline scalar Dimix
        (
            label speciei,
            const scalar p,
            const scalar T
        ) const;

        //- Update mixture-level quantities (native Ely-Hanley signature)
        inline void updateTRANS
        (
            const scalar& TcM,
            const scalar& VcM,
            const scalar& MM,
            const scalar& omegaM,
            const scalar& ZcM,
            const List<scalar>& Ymd,
            const List<scalar>& Xmd,
            const List<List<scalar>>& Tcmd,
            const List<List<scalar>>& Pcmd,
            const List<List<scalar>>& Mmd,
            const List<List<scalar>>& sigmd
        );

        //- Access functions
        inline scalar TcM()    const { return TcM_; }
        inline scalar VcM()    const { return VcM_; }
        inline scalar MM()     const { return MM_;  }
        inline scalar omegaM() const { return omegaM_; }
        inline scalar ZcM()    const { return ZcM_; }
        inline List<scalar> Ymd() const { return Ymd_; }
        inline List<scalar> Xmd() const { return Xmd_; }
        inline List<List<scalar>> Tcmd() const { return Tcmd_; }
        inline List<List<scalar>> Pcmd() const { return Pcmd_; }
        inline List<List<scalar>> Mmd()  const { return Mmd_;  }
        inline List<List<scalar>> sigmd() const { return sigmd_; }

        void write(Ostream& os) const;


    // Member Operators

        inline void operator=(const elyHanleyTransport&);
        inline void operator+=(const elyHanleyTransport&);
        inline void operator*=(const scalar);


    // Friend operators

        friend elyHanleyTransport operator+ <Thermo>
        (
            const elyHanleyTransport&,
            const elyHanleyTransport&
        );

        friend elyHanleyTransport operator* <Thermo>
        (
            const scalar,
            const elyHanleyTransport&
        );


    // Ostream Operator

        friend Ostream& operator<< <Thermo>
        (
            Ostream&,
            const elyHanleyTransport&
        );
};


} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "elyHanleyTransportI.H"

#ifdef NoRepository
    #include "elyHanleyTransport.C"
#endif

#endif
