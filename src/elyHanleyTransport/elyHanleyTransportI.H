/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Copyright (C) 2024-2025 RGP-13
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

\*---------------------------------------------------------------------------*/

#include "elyHanleyTransport.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * Constructors (inline) * * * * * * * * * * * * * //

template<class Thermo>
inline Foam::elyHanleyTransport<Thermo>::elyHanleyTransport
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
)
:
    Thermo(t),
    TcM_(TcM), VcM_(VcM), MM_(MM), omegaM_(omegaM), ZcM_(ZcM),
    // Methane reference constants.
    //   Critical point: T_c = 190.564 K, V_c = 99.2 cm^3/mol = 0.0992 m^3/kmol.
    //   Acentric factor omega = 0.011.
    //   Z_c,methane = 0.286.
    //   (Source: NIST Chemistry WebBook; consistent with Hanley-McCarty-
    //    Haynes 1975 reference state.)
    TcO_(190.564), VcO_(0.0992), MO_(16.043), omegaO_(0.011), ZcO_(0.286),
    Ymd_(Ymd), Xmd_(Xmd),
    Tcmd_(Tcmd), Pcmd_(Pcmd), Mmd_(Mmd), sigmd_(sigmd)
{}


template<class Thermo>
inline Foam::elyHanleyTransport<Thermo>::elyHanleyTransport
(
    const word& name,
    const elyHanleyTransport& tr
)
:
    Thermo(name, tr),
    TcM_(tr.TcM_), VcM_(tr.VcM_), MM_(tr.MM_),
    omegaM_(tr.omegaM_), ZcM_(tr.ZcM_),
    TcO_(tr.TcO_), VcO_(tr.VcO_), MO_(tr.MO_),
    omegaO_(tr.omegaO_), ZcO_(tr.ZcO_),
    Ymd_(tr.Ymd_), Xmd_(tr.Xmd_),
    Tcmd_(tr.Tcmd_), Pcmd_(tr.Pcmd_),
    Mmd_(tr.Mmd_), sigmd_(tr.sigmd_)
{}


template<class Thermo>
inline Foam::autoPtr<Foam::elyHanleyTransport<Thermo>>
Foam::elyHanleyTransport<Thermo>::clone() const
{
    return autoPtr<elyHanleyTransport<Thermo>>
    (
        new elyHanleyTransport<Thermo>(*this)
    );
}


// * * * * * * * * * * * * Shape factors (Leach 1968) * * * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::softClamp
(
    const scalar x,
    const scalar lo,
    const scalar hi,
    const scalar w
)
{
    // Identity-on-[lo, hi] soft clamp:
    //
    //   x <= lo - w       : lo                                  (saturated)
    //   lo - w < x < lo   : lo + w * p((x - (lo-w))/w)           (lower ramp)
    //   lo  <= x <= hi    : x                                    (identity)
    //   hi  < x < hi + w  : hi + w * p(1 - (x-hi)/w) ... mirrored (upper ramp)
    //   x >= hi + w       : hi                                  (saturated)
    //
    // p(u) = -4u^3 + 7u^4 - 3u^5 with p(0) = p(1) = 0,
    //        p'(0) = 0, p'(1) = 1, p''(0) = p''(1) = 0.
    //
    // The previous geometry had the ramp straddling lo / hi (identity
    // zone [lo+w, hi-w]) which forced the LCL polynomial to start
    // damping inside its nominal valid range [0.5, 2]. For a low-T_c
    // fluid (e.g. O2, T_c = 154.6 K) the ramp at T_r = 1.75 landed at
    // T = 270 K and the saturation at T_r = 2.25 = 348 K — the entire
    // smoothing window was 78 K, ~2 sample points at dT = 20 K, and
    // this produced a visible kink in mu(T), kappa(T) near 300 K
    // (cf. test/O2_phase2). The new geometry preserves the LCL valid
    // range [lo, hi] as exact identity and applies the C^2 quintic
    // ramp outside this band, so the polynomial is never artificially
    // damped inside its validated regime.
    //
    // The ramp polynomial is non-monotonic by ~|w/5| (a consequence of
    // landing the saturation slope and the identity slope on the same
    // endpoint pair); for the shape-factor calls (lo = 0.5, hi = 2.0)
    // this means R(x) can dip ~0.1 below lo or rise ~0.1 above hi
    // inside the ramp. The downstream LCL polynomial is analytic for
    // T_p > 0 so this brief excursion is well-behaved.
    if (x <= lo - w) return lo;
    if (x >= hi + w) return hi;
    if (x < lo)
    {
        const scalar u  = (x - (lo - w))/w;
        const scalar u3 = u*u*u;
        const scalar u4 = u3*u;
        const scalar u5 = u4*u;
        return lo + w*(-4.0*u3 + 7.0*u4 - 3.0*u5);
    }
    if (x > hi)
    {
        const scalar v  = 1.0 - (x - hi)/w;          // v in [0, 1]
        const scalar v3 = v*v*v;
        const scalar v4 = v3*v;
        const scalar v5 = v4*v;
        return hi - w*(-4.0*v3 + 7.0*v4 - 3.0*v5);
    }
    return x;
}


template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::phiShape
(
    const scalar Vri,
    const scalar Tri,
    const scalar omegai
) const
{
    // Leach, Chappelear & Leland 1968 (AIChE J. 14, 568) correlation
    // for the volume shape factor phi. Bounds on V_r, T_r are taken
    // from Oefelein (AIAA 2019-0634) Eqs. (45)-(46) but applied via a
    // C^2-continuous quintic soft clamp so dphi/dT and d2phi/dT^2 stay
    // continuous when T_r leaves the [0.5, 2.0] band.
    //
    // softClamp now uses identity-on-[lo, hi] geometry, so this call
    // evaluates the LCL polynomial on its full nominal valid range
    // [0.5, 2.0] without artificial damping inside the band.
    //
    // Half-width w = 0.5 in reduced units, i.e. 0.5*T_c (or 0.5*V_c)
    // wide on each saturation side. For low-T_c fluids (O2 T_c=154.6 K,
    // CH4 T_c=190.6 K) this puts the upper-T_r ramp at T in
    // [T_c, 1.5 T_c] = [154.6, 232] K wide for O2 ~ 77 K, ~4 sample
    // points at the test-case dT = 20 K so the ramp blends visually
    // smooth. For H2O (T_c = 647 K) the ramp is 324 K wide. The
    // previous w = 0.25 with the old straddle geometry shrank the
    // identity zone to [0.75, 1.75], moving the transition into
    // T = 270-348 K for O2 and producing a visible kink near 300 K
    // in test/O2_phase2.
    const scalar w = 0.5;
    const scalar Vp = softClamp(Vri, 0.5, 2.0, w);
    const scalar Tp = softClamp(Tri, 0.5, 2.0, w);
    // Leach-Chappelear-Leland 1968 Table 1 coefficients (methane ref).
    const scalar a1 =  0.3409;  // c_phi1
    const scalar a2 =  1.0054;  // c_phi2
    const scalar a3 = -0.1908;  // c_phi3
    const scalar a4 =  0.7960;  // c_phi4
    const scalar fPhi =
        a1*(Vp - a2) + a3*(Vp - a4)*log(Tp);
    return (1.0 + (omegai - omegaO_)*fPhi)*ZcO_/max(1e-20, ZcM_);
}


template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::thetaShape
(
    const scalar Vri,
    const scalar Tri,
    const scalar omegai
) const
{
    // Leach, Chappelear & Leland 1968 correlation for theta (Oefelein
    // Eq. 44). Soft-clamped on V_r, T_r with the same w = 0.5
    // identity-on-[0.5, 2] geometry as phiShape — see the comment
    // there for why w was widened from 0.25 (kink in mu, kappa near
    // 300 K for O2 in test/O2_phase2).
    const scalar w = 0.5;
    const scalar Vp = softClamp(Vri, 0.5, 2.0, w);
    const scalar Tp = softClamp(Tri, 0.5, 2.0, w);
    const scalar b1 =  0.0892;  // c_theta1
    const scalar b2 = -0.3680;  // c_theta2
    const scalar b3 =  0.8464;  // c_theta3
    const scalar b4 = -0.0651;  // c_theta4
    const scalar VpMinusVpo = Vp - 1.0;  // V+_o = 1 for reference
    const scalar fTheta =
        b1 + b2*log(Tp) + (b3 + b4/Tp)*VpMinusVpo;
    return 1.0 + (omegai - omegaO_)*fTheta;
}


// * * * * * * * * * * Methane dilute viscosity helper * * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::muStarO
(
    const scalar TO
) const
{
    // Sutherland fit to CoolProp methane in the dilute (zero-density)
    // limit, T in [100, 2000] K:
    //   mu*_o(T) = mu0 (T/T0)^1.5 (T0 + S)/(T + S),
    //   mu0 = 1.130893e-5 Pa.s @ T0 = 300 K, S = 195.78 K.
    // Median |err| 1.53 %, 95th 2.57 %, max 4.12 % on the fit grid.
    // Lower-bounded at 40 K to keep pow/(T+S) safe; the asymptotic
    // T^(1/2) tail makes the fit safe to extrapolate well past the
    // upper edge of the fit window, which matters for low-T_c targets
    // (e.g. H2 with f_x ~ 0.15 maps T = 300 K -> T_o ~ 2 kK).
    const scalar T_dilute = max(scalar(40), TO);
    const scalar muMu0 = 1.130893e-05;
    const scalar muT0  = 300.0;
    const scalar muS   = 195.7781;
    return muMu0*pow(T_dilute/muT0, 1.5)*(muT0 + muS)/(T_dilute + muS);
}


// * * * * * * * * * * Methane reference viscosity * * * * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::muRef
(
    const scalar rhoO,
    const scalar TO
) const
{
    // Hanley-McCarty-Haynes 1975 methane reference viscosity (Oefelein
    // AIAA 2019-0634 Eqs. 47-50). Three-term decomposition:
    //
    //   mu_o(rho, T) = mu*_o(T) + mu**_o(T) * rho + Delta_mu_o(rho, T)
    //
    //   mu*_o(T)         = sum_{n=1..9} a_n T^((n-4)/3)
    //   mu**_o(T)        = b1 + b2 (b3 - ln(T/b4))^2
    //   Delta_mu_o(rho,T) =
    //        exp(c1 + c2/T)
    //      * { exp[ (c3 + c4/T^1.5) rho^0.1
    //               + (rho/rho_c - 1) rho^0.5 (c5 + c6/T + c7/T^2) ] - 1 }
    //
    // Coefficients are stage-wise least-squares fitted to CoolProp
    // methane (Setzmann-Wagner 1991 EoS + Friend-Ingham-Ely 1991
    // transport) over T in [95, 2000] K and P in [1e5, 2.5e8] Pa, with
    // 2-phase points filtered out. Fit accuracy on the 1080-point grid:
    //   median |err| 0.16 %, 95th %ile 1.92 %, max 8.45 %.
    //
    // This replaces the previous Sutherland-only + cubic-in-rho_r
    // approximation, which had no T dependence in the dense correction
    // and saturated at rho_r = 3. That approximation produced the
    // unphysical "bump and decrease" of mu(T) below T_c in LOX/LH2/LH2O
    // operating points (because as T decreases below T_c the methane
    // reference state reaches the dense-liquid regime, where real mu
    // rises like exp(c2/T) but the saturated cubic was flat). The HHM
    // form has the explicit exp(c2/T) factor in Delta_mu_o that
    // captures this Arrhenius rise correctly.
    //
    // ECS-frame state regularisation. The LCL methane reference fluid
    // breaks correspondence with O2/H2/H2O at very low T_o (because
    // T_o approaches methane's triple point 90.7 K while the target
    // fluid is just normal LOX/LH2/LH2O at moderate T_r). Likewise
    // very high rho_o pushes us beyond the HHM fit's validated range
    // (T <=2 kK, rho up to ~3.5 rho_c,o = 569 kg/m^3). Both are
    // saturated via C^2 softClamp so the kappa, mu curves stay smooth
    // through the cap. The boundaries are deliberately generous so
    // they only fire in genuine LOX/LH2/LH2O states.
    //
    // For light fluids like H2 (T_c = 33 K, f_x ~ 0.17), T_o = T/f_x
    // can easily reach 5-10 kK at moderate physical T, so we use a
    // Sutherland-form dilute term (closed form, monotonic, exact
    // 1/2-power asymptote in T) rather than a 9-term polynomial: the
    // polynomial would saturate at the upper softClamp boundary and
    // produce a flat mu(T) past T = T_o,sat * f_x, which surfaced as
    // a constant ~9e-6 Pa.s mu_H2 above ~350 K.
    const scalar T_clip   = softClamp(TO,   133.0, 2000.0, 30.0); // for HHM dense fit
    const scalar rho_clip = softClamp(rhoO,   0.0,  569.0, 80.0);

    // ----- mu*_o(T) : Sutherland fit to CoolProp methane (zero-density)
    // Delegated to muStarO(); see the helper for the fit constants and
    // accuracy notes. Shared with kappa() so the ECS-mapped translational
    // mu_o is identical in both routes.
    const scalar muStar = muStarO(TO);

    // T-power abbreviations used by HHM dense (computed on T_clip)
    const scalar Tinv  = 1.0/T_clip;

    // ----- mu**_o(T) * rho : initial-density correction -----
    const scalar bmu1 =  3.0469006140e-08;
    const scalar bmu2 = -4.8747267400e-09;
    const scalar bmu3 = -4.1677739781e-01;
    const scalar bmu4 =  3.0580626168e+03;
    const scalar muStarStar =
        bmu1 + bmu2*sqr(bmu3 - log(T_clip/bmu4));

    // ----- Delta_mu_o(rho, T) : HHM dense correction -----
    const scalar cmu1 = -3.1860375669e+01;
    const scalar cmu2 =  3.9412904984e+02;
    const scalar cmu3 =  1.1627949756e+01;
    const scalar cmu4 = -2.3494844982e+03;
    const scalar cmu5 =  5.9482250880e-03;
    const scalar cmu6 =  8.4304524138e-01;
    const scalar cmu7 =  4.5651964133e+02;

    const scalar rho_r_o  = rho_clip/162.66;        // rho_c,methane = 162.66 kg/m^3
    const scalar T_sqrt   = sqrt(T_clip);
    const scalar T_15     = T_clip*T_sqrt;          // T^(3/2)
    const scalar rho01    = pow(rho_clip, 0.10);
    const scalar rho05    = sqrt(rho_clip);

    const scalar inner_raw =
          (cmu3 + cmu4/T_15)*rho01
        + (rho_r_o - 1.0)*rho05*(cmu5 + cmu6*Tinv + cmu7*Tinv*Tinv);
    // Clip the inner exponent only as fp64 overflow guard. The
    // prefactor exp(c1 + c2/T) is O(1e-13) so very large inner is
    // physically required to balance into μPa·s magnitudes; T_o and
    // rho_o softClamps above already keep us in the validated regime.
    const scalar inner = max(scalar(-50), min(inner_raw, scalar(50)));
    const scalar dMuDense =
        exp(cmu1 + cmu2*Tinv) * (exp(inner) - 1.0);

    return muStar + muStarStar*rho_clip + dMuDense;  // Pa.s
}


// * * * * * * * * * * Methane reference conductivity * * * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::lambdaRef
(
    const scalar rhoO,
    const scalar TO
) const
{
    // Methane *translational* thermal conductivity in the methane (ECS)
    // frame, decomposed analogously to muRef:
    //   lambda_o^trans(rho_o, T_o)
    //     = lambda*_o(T_o)                      (Oefelein Eq. 52)
    //     + lambda**_o(T_o) * rho_o             (initial-density)
    //     + Delta_lambda_o(rho_o, T_o)          (dense, HHM 1975)
    //
    // Per Oefelein AIAA 2019-0634 Eq. (52):
    //   lambda*_o(T_o) = (15/4) (R_u/W_o) mu*_o(T_o)
    // i.e. the dilute term is *monatomic-Eucken* and contains only the
    // translational contribution. The methane internal-mode (rotational
    // + vibrational) part of the full HHM 1975 conductivity is *not*
    // included here on purpose: it is reintroduced in kappa() via the
    // Mason-Monchick (1962) Modified Eucken evaluated at the *target*
    // fluid's actual T and Cp_ig, not at T_o = T/f_x.
    //
    // Why the change vs the previous Sutherland-fit-to-CoolProp methane
    // dilute lambda: the prior fit captured methane's full dilute k
    // (translational + internal). When ECS-scaled by F_lam onto a low-
    // T_c target (e.g. H2 with f_x ~ 0.15, T_o = 300/0.15 ~ 2 kK), the
    // methane internal contribution at T_o = 2 kK is ~10x larger than
    // methane's internal at 300 K, and was being mapped onto H2 even
    // though H2 has no such polyatomic internal modes activated at
    // 300 K. Result: ~3x overprediction of k_H2 at moderate (T, p).
    // Splitting the dilute term per Oefelein Eq. (52) and adding the
    // target-fluid Modified Eucken in kappa() removes this artefact.
    //
    // ECS-frame state regularisation: same caps as muRef (see comment
    // there) — needed because methane reference loses physical
    // correspondence with the target fluid at very low T_o / very
    // high rho_o.
    const scalar T_clip   = softClamp(TO,   133.0, 2000.0, 30.0);
    const scalar rho_clip = softClamp(rhoO,   0.0,  569.0, 80.0);

    // ----- lambda*_o(T) : Oefelein Eq. (52), monatomic Eucken on
    //   methane dilute viscosity. R_u/W_o = 8314.46/16.043 J/(kg.K).
    const scalar lamStar =
        (15.0/4.0)*(Foam::constant::thermodynamic::RR/MO_)*muStarO(TO);

    const scalar Tinv = 1.0/T_clip;

    // ----- lambda**_o(T) * rho : initial-density correction -----
    const scalar blam1 =  6.8442991270e-05;
    const scalar blam2 =  9.9053474710e-06;
    const scalar blam3 =  1.7297753616e+00;
    const scalar blam4 =  2.0455529588e+02;
    const scalar lamStarStar =
        blam1 + blam2*sqr(blam3 - log(T_clip/blam4));

    // ----- Delta_lambda_o(rho, T) : HHM dense correction -----
    const scalar clam1 = -2.0744527541e+01;
    const scalar clam2 = -2.5245743694e+01;
    const scalar clam3 =  9.9784450641e+00;
    const scalar clam4 = -2.1110820636e+02;
    const scalar clam5 =  1.6978607161e-02;
    const scalar clam6 =  1.5423931760e+00;
    const scalar clam7 = -3.7436227050e+01;

    const scalar rho_r_o = rho_clip/162.66;
    const scalar T_sqrt  = sqrt(T_clip);
    const scalar T_15    = T_clip*T_sqrt;
    const scalar rho01   = pow(rho_clip, 0.10);
    const scalar rho05   = sqrt(rho_clip);

    const scalar inner_raw =
          (clam3 + clam4/T_15)*rho01
        + (rho_r_o - 1.0)*rho05*(clam5 + clam6*Tinv + clam7*Tinv*Tinv);
    const scalar inner = max(scalar(-10), min(inner_raw, scalar(10)));
    const scalar dLamDense =
        exp(clam1 + clam2*Tinv) * (exp(inner) - 1.0);

    return lamStar + lamStarStar*rho_clip + dLamDense;   // W/m/K
}


// * * * * * * * * * * Enskog size-difference corrections * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::enskogXeta
(
    const scalar Tx,
    const scalar fx,
    const scalar dfxdT
) const
{
    // Ely 1981, J. Res. Natl. Bur. Stand. 86 (1981) 597 Eq. 23.
    //   X_eta = { [1 - d_eta (T/fx) dfxdT] * Zcx/Zco }^(1/2),  d_eta = 1.5
    const scalar d_eta = 1.5;
    const scalar core = 1.0 - d_eta*(Tx/max(fx, 1e-20))*dfxdT;
    const scalar zRatio = ZcM_/max(ZcO_, 1e-20);
    if (core*zRatio <= 0.0) return 1.0;
    return sqrt(core*zRatio);
}


template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::enskogXlam
(
    const scalar Tx,
    const scalar fx,
    const scalar dfxdT
) const
{
    // Ely & Hanley 1981b Eq. 27.
    //   X_lam = { [1 - d_lam (T/fx) dfxdT] * Zco/Zcx }^(3/2),  d_lam = 1.0
    const scalar d_lam = 1.0;
    const scalar core = 1.0 - d_lam*(Tx/max(fx, 1e-20))*dfxdT;
    const scalar zRatio = ZcO_/max(ZcM_, 1e-20);
    if (core*zRatio <= 0.0) return 1.0;
    return pow(core*zRatio, 1.5);
}


// * * * * * * * * * * Public mu, kappa, phi, Dimix * * * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::mu
(
    const scalar p,
    const scalar T
) const
{
    // Pure ECS / Ely-Hanley evaluation for every species. The previous
    // hybrid Chung short-circuit produced step jumps in mu(Y_mix) where
    // the polarity selector flipped (e.g. Y_H2O ~ 0.86 in O2/H2O), so it
    // has been removed in favour of a single continuous formula.
    // Mixture reducing ratios f_x, h_x via shape factors of a pseudo-
    // pure fluid with mixture critical parameters.
    const scalar Vri    = (this->rho(p, T) > 0)
                        ? (MM_/this->rho(p, T))/VcM_ : 1.0;
    const scalar Tri    = T/max(TcM_, 1e-20);
    const scalar phi_i  = phiShape(Vri, Tri, omegaM_);
    const scalar theta_i = thetaShape(Vri, Tri, omegaM_);

    const scalar hx = (VcM_/VcO_)*phi_i;
    const scalar fx = (TcM_/TcO_)*theta_i;

    const scalar rhoSI = this->rho(p, T);             // kg/m^3
    const scalar rhoO  = rhoSI*hx;                    // scaled to methane frame
    const scalar TO    = T/fx;

    // Scaling factor F_mu per Oefelein Eq. (41)
    const scalar F_mu =
        sqrt(MM_/MO_)*sqrt(fx)*pow(hx, -2.0/3.0);

    // Enskog correction with numerical dfx/dT
    const scalar dT = max(1.0, 1e-3*T);
    const scalar fx_p = (TcM_/TcO_)
        *thetaShape(Vri, (T + dT)/TcM_, omegaM_);
    const scalar fx_m = (TcM_/TcO_)
        *thetaShape(Vri, (T - dT)/TcM_, omegaM_);
    const scalar dfxdT = (fx_p - fx_m)/(2.0*dT);
    const scalar X_eta = enskogXeta(T, fx, dfxdT);

    const scalar muO = muRef(rhoO, TO);     // Pa.s

    return F_mu*muO*X_eta;
}


template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::kappa
(
    const scalar p,
    const scalar T
) const
{
    // Translational + internal split per Ely & Hanley (1983) and
    // Mason & Monchick (1962); see the class header for the references.
    //
    //   lambda(p, T) = F_lam * X_lam * lambda_o^trans(rho_o, T_o)
    //                + lambda_int^target(T)
    //
    // Translational piece is ECS-mapped from the methane reference;
    // internal-mode piece is evaluated *at the target fluid's actual
    // T and Cp_ig*, NOT at T_o = T/f_x. This avoids the pathology
    // where for low-T_c targets (e.g. H2: T_c = 33 K, f_x ~ 0.15)
    // mapping T = 300 K to T_o ~ 2 kK pulled the methane reference's
    // polyatomic internal-mode k onto a fluid that does not have
    // those modes activated at room T (~3x overprediction of k_H2
    // observed in test/H2_phase2 prior to this fix).

    const scalar Vri    = (this->rho(p, T) > 0)
                        ? (MM_/this->rho(p, T))/VcM_ : 1.0;
    const scalar Tri    = T/max(TcM_, 1e-20);
    const scalar phi_i   = phiShape(Vri, Tri, omegaM_);
    const scalar theta_i = thetaShape(Vri, Tri, omegaM_);

    const scalar hx = (VcM_/VcO_)*phi_i;
    const scalar fx = (TcM_/TcO_)*theta_i;

    const scalar rhoSI = this->rho(p, T);
    const scalar rhoO  = rhoSI*hx;
    const scalar TO    = T/fx;

    // F_lam, F_mu per Oefelein Eqs. (41)-(42).
    const scalar hx_m23 = pow(hx, -2.0/3.0);
    const scalar sqrtfx = sqrt(fx);
    const scalar F_lam = sqrt(MO_/MM_)*sqrtfx*hx_m23;
    const scalar F_mu  = sqrt(MM_/MO_)*sqrtfx*hx_m23;

    // Enskog X_lam with numerical dfx/dT (same as before).
    const scalar dT = max(1.0, 1e-3*T);
    const scalar fx_p = (TcM_/TcO_)
        *thetaShape(Vri, (T + dT)/TcM_, omegaM_);
    const scalar fx_m = (TcM_/TcO_)
        *thetaShape(Vri, (T - dT)/TcM_, omegaM_);
    const scalar dfxdT = (fx_p - fx_m)/(2.0*dT);
    const scalar X_lam = enskogXlam(T, fx, dfxdT);

    // ----- ECS translational contribution -----
    const scalar lamECS = F_lam*lambdaRef(rhoO, TO)*X_lam;     // W/m/K

    // ----- Modified Eucken internal-mode contribution -----
    // Mason-Monchick (1962); Reid-Prausnitz-Poling 5th ed. Eq. (10-3.14):
    //   lambda_int = f_int * mu_dilute_target * (Cp_ig - 5R/2)
    // with the empirical Eucken factor f_int = 1.32 (Stiel-Thodos /
    // Mason-Monchick polyatomic constant; see Reid-Prausnitz-Poling
    // for the derivation and Huber-Laesecke-Perkins 2003 for the same
    // form embedded in the modern REFPROP/CoolProp transport models).
    //
    // mu_dilute_target uses the ECS-mapped methane dilute viscosity
    // (mu*_o(T_o) scaled by F_mu): this is exactly the zero-density
    // limit of the existing mu(p, T) routine and ensures the same
    // dilute viscosity drives both the translational Eucken term in
    // lambdaRef *and* this internal-mode term.
    //
    // Cp_ig is the target *ideal-gas* Cp. The thermo chain
    //   janafThermo<EOS>::Cp(p, T) = JANAF_polynomial(T) + EOS::Cp(p, T)
    // returns Cp_real = Cp_ig + Cp_dep_SRK; using Cp_real here would
    // pull the pseudo-critical Cp peak (e.g. O2 at T_c = 154 K and
    // 100 bar shows Cp_real ~ 4500 J/(kg K) vs Cp_ig ~ 920) into
    // lambda_int, which then *decreases* once the (T, P) path passes
    // beyond the peak -- producing the unphysical "k(T) drops at low
    // T" behaviour in supercritical isobars (cf. test/O2_phase2 P=100
    // bar: k peaks near T = 170 K then drops toward T = 100 K instead
    // of rising toward the dense-liquid Lemmon-Jacobsen 2004 limit).
    //
    // Modified Eucken is *defined* on the dilute (ideal-gas) internal-
    // mode contribution C_int = Cp_ig - 5R/2; the dense Cp departure
    // is captured implicitly by lambda**_o*rho_o + Delta_lambda_o in
    // lambdaRef (translational density correction).
    //
    // Cleanest portable evaluation: call this->Cp(p_ref, T) at a low
    // reference pressure where the SRK Cp departure is negligible.
    // p_ref = 1 atm = 101325 Pa: at T = 300 K, B = b*p/RT ~ 9e-4 and
    // Cp_dep is O(0.01 %) of Cp_ig; even at T = 100 K (above O2 NBP =
    // 90 K so single-phase vapour at 1 atm) the SRK Cp departure is
    // < 1 % of Cp_ig. The extra cubic-Z solve cost is one per kappa()
    // call; acceptable given typical CFD call frequency.
    constexpr scalar p_ref_for_CpIg = 1.01325e5;
    const scalar muTargetDilute = F_mu*muStarO(TO);            // Pa.s
    const scalar CpIg           = this->Cp(p_ref_for_CpIg, T); // J/(kg.K)
    const scalar Rs             = this->R();                   // J/(kg.K)
    const scalar CpInt          = CpIg - 2.5*Rs;               // internal-mode Cp
    const scalar lamInt         = 1.32*muTargetDilute*max(scalar(0), CpInt);

    return lamECS + lamInt;
}


// * * * * * * * * * * * * * phi (Takahashi) * * * * * * * * * * * * * * * //
// Takahashi (1974) high-pressure correction identical to chungTransport.

template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::phi
(
    scalar PRR,
    scalar TRR
) const
{
    // Same Takahashi 1974 table as chungTransport. Implementation
    // duplicated here so that elyHanleyTransport is self-contained.
    const scalar A01 = 0.038042, B01 = 1.52267;
    const scalar A02 = 0.067433, B02 = 2.16794;
    const scalar A03 = 0.098317, B03 = 2.42910;
    const scalar A04 = 0.137610, B04 = 2.77605;
    const scalar A05 = 0.175081, B05 = 2.98256;
    const scalar A06 = 0.216376, B06 = 3.11384;
    const scalar A08 = 0.314051, B08 = 3.50264;
    const scalar A10 = 0.385736, B10 = 3.07773;
    const scalar C10 = 0.141211, E10 = 13.45454;
    const scalar A12 = 0.514553, B12 = 3.54744;
    const scalar C12 = 0.278407, E12 = 14.00000;
    const scalar A14 = 0.599184, B14 = 3.61216;
    const scalar C14 = 0.372683, E14 = 10.00900;
    const scalar A16 = 0.557725, B16 = 3.41882;
    const scalar C16 = 0.504894, E16 = 8.57519;
    const scalar A18 = 0.593007, B18 = 3.18415;
    const scalar C18 = 0.678469, E18 = 10.37483;
    const scalar A20 = 0.696001, B20 = 3.37660;
    const scalar C20 = 0.665702, E20 = 11.21674;
    const scalar A25 = 0.790770, B25 = 3.27984;
    const scalar A30 = 0.502100, B30 = 2.39031;
    const scalar C30 = 0.602907, E30 = 6.19043;
    const scalar A40 = 0.837452, B40 = 3.23513;
    const scalar A50 = 0.890390, B50 = 3.13001;

    scalar Y = 1.0;
    TRR = (TRR < 1.0) ? 1.0 : TRR;
    TRR = (TRR > 5.0) ? 5.0 : TRR;
    PRR = (PRR > 5.0) ? 5.0 : PRR;

    if (PRR < 0.1)
    {
        const scalar Y1 = 1.01*(1.0 - A01*pow(TRR, -B01));
        Y = 1.0 + (Y1 - 1.0)*PRR/0.1;
    }
    else if (PRR < 1.0)
    {
        // Linear interpolation in the low-pressure table
        const scalar Y01 = 1.01*(1.0 - A01*pow(TRR, -B01));
        const scalar Y10 = 1.02*(1.0 - A10*pow(TRR, -B10))*(1.0 - C10*pow(TRR, -E10));
        Y = Y01 + (Y10 - Y01)*(PRR - 0.1)/0.9;
    }
    else if (PRR < 2.0)
    {
        const scalar Y10 = 1.02*(1.0 - A10*pow(TRR, -B10))*(1.0 - C10*pow(TRR, -E10));
        const scalar Y20 = 1.04*(1.0 - A20*pow(TRR, -B20))*(1.0 - C20*pow(TRR, -E20));
        Y = Y10 + (Y20 - Y10)*(PRR - 1.0);
    }
    else if (PRR < 3.0)
    {
        const scalar Y20 = 1.04*(1.0 - A20*pow(TRR, -B20))*(1.0 - C20*pow(TRR, -E20));
        const scalar Y30 = 1.05*(1.0 - A30*pow(TRR, -B30))*(1.0 - C30*pow(TRR, -E30));
        Y = Y20 + (Y30 - Y20)*(PRR - 2.0);
    }
    else if (PRR < 4.0)
    {
        const scalar Y30 = 1.05*(1.0 - A30*pow(TRR, -B30))*(1.0 - C30*pow(TRR, -E30));
        const scalar Y40 = 1.06*(1.0 - A40*pow(TRR, -B40));
        Y = Y30 + (Y40 - Y30)*(PRR - 3.0);
    }
    else
    {
        const scalar Y40 = 1.06*(1.0 - A40*pow(TRR, -B40));
        const scalar Y50 = 1.07*(1.0 - A50*pow(TRR, -B50));
        Y = Y40 + (Y50 - Y40)*(PRR - 4.0);
    }

    if (Y < 1e-16) Y = 1e-16;
    return Y;
}


template<class Thermo>
inline Foam::scalar Foam::elyHanleyTransport<Thermo>::Dimix
(
    label speciei,
    const scalar p,
    const scalar T
) const
{
    // Fuller 1987 low-pressure binary diffusion coefficient D_ij times
    // Takahashi 1974 high-pressure correction; identical to chungTransport
    // so that switching between transport models only changes mu, kappa.
    scalar sum = 0.0;
    const scalar T175 = T*sqrt(T*sqrt(T));
    const scalar pAtm = p/101325.0;
    forAll(Xmd_, j)
    {
        if (speciei != j)
        {
            sum += Xmd_[j]/
                   (
                       phi(p/Pcmd_[j][speciei], T/Tcmd_[j][speciei])
                      *0.001*T175*sqrt(Mmd_[j][speciei])
                      /(pAtm*sqr(sigmd_[j][speciei]))
                   );
        }
    }
    return (sum == 0.0) ? 0.0 : ((1.0 - Ymd_[speciei])/sum)/1e4;
}


// * * * * * * * * * * * * * updateTRANS * * * * * * * * * * * * * * * * * //

template<class Thermo>
inline void Foam::elyHanleyTransport<Thermo>::updateTRANS
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
)
{
    TcM_ = TcM; VcM_ = VcM; MM_ = MM; omegaM_ = omegaM; ZcM_ = ZcM;
    Ymd_ = Ymd; Xmd_ = Xmd;
    Tcmd_ = Tcmd; Pcmd_ = Pcmd;
    Mmd_ = Mmd; sigmd_ = sigmd;
}


// Chung-style 8-arg updateTRANS adapter removed: SRKelyHanleyMixture now
// drives the native 5-arg signature directly, supplying its own
// mass-weighted Z_c,m instead of the previous Pitzer fallback.


// * * * * * * * * * * * * * Member Operators * * * * * * * * * * * * * * //

template<class Thermo>
inline void Foam::elyHanleyTransport<Thermo>::operator=
(
    const elyHanleyTransport<Thermo>& tr
)
{
    Thermo::operator=(tr);
    TcM_ = tr.TcM_; VcM_ = tr.VcM_; MM_ = tr.MM_;
    omegaM_ = tr.omegaM_; ZcM_ = tr.ZcM_;
    Ymd_ = tr.Ymd_; Xmd_ = tr.Xmd_;
    Tcmd_ = tr.Tcmd_; Pcmd_ = tr.Pcmd_;
    Mmd_ = tr.Mmd_; sigmd_ = tr.sigmd_;
}


template<class Thermo>
inline void Foam::elyHanleyTransport<Thermo>::operator+=
(
    const elyHanleyTransport<Thermo>& tr
)
{
    scalar X1 = this->Y()/this->W();
    Thermo::operator+=(tr);
    if (mag(this->Y()) > small)
    {
        X1 *= this->W()/this->Y();
        const scalar X2 = this->W()*tr.Y()/(tr.W()*this->Y());
        TcM_    = X1*TcM_    + X2*tr.TcM_;
        VcM_    = X1*VcM_    + X2*tr.VcM_;
        MM_     = X1*MM_     + X2*tr.MM_;
        omegaM_ = X1*omegaM_ + X2*tr.omegaM_;
        ZcM_    = X1*ZcM_    + X2*tr.ZcM_;
    }
}


template<class Thermo>
inline void Foam::elyHanleyTransport<Thermo>::operator*=(const scalar s)
{
    Thermo::operator*=(s);
}


// * * * * * * * * * * * * * Friend Operators * * * * * * * * * * * * * * //

template<class Thermo>
Foam::elyHanleyTransport<Thermo> Foam::operator+
(
    const elyHanleyTransport<Thermo>& t1,
    const elyHanleyTransport<Thermo>& t2
)
{
    Thermo tt
    (
        static_cast<const Thermo&>(t1) + static_cast<const Thermo&>(t2)
    );
    return elyHanleyTransport<Thermo>
    (
        tt,
        t1.TcM_, t1.VcM_, t1.MM_, t1.omegaM_, t1.ZcM_,
        t1.Ymd_, t1.Xmd_, t1.Tcmd_, t1.Pcmd_, t1.Mmd_, t1.sigmd_
    );
}


template<class Thermo>
Foam::elyHanleyTransport<Thermo> Foam::operator*
(
    const scalar s,
    const elyHanleyTransport<Thermo>& tr
)
{
    return elyHanleyTransport<Thermo>
    (
        s*static_cast<const Thermo&>(tr),
        tr.TcM_, tr.VcM_, tr.MM_, tr.omegaM_, tr.ZcM_,
        tr.Ymd_, tr.Xmd_, tr.Tcmd_, tr.Pcmd_, tr.Mmd_, tr.sigmd_
    );
}
