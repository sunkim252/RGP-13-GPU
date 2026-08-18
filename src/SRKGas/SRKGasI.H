/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2014-2023 OpenFOAM Foundation
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

#include "SRKGas.H"
#include "mathematicalConstants.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Xterm
(
    const scalar p,
    const scalar T
) const
{
    // Cache Z to avoid redundant cubic solve in dZdT
    const scalar Zv = Z(p, T);
    const scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);
    const scalar sqrtT = sqrt(T);
    const scalar first =
        3*coef2_*log((Zv + B)/Zv)/(4*b_*sqrtT);

    return
    (
        first
      - (
            -coef1_/b_ + 0.5*coef2_*sqrtT/b_
        )
       *(
            b_*p
           /((Zv + B)*Foam::constant::thermodynamic::RR*T*T)
          + (B/(sqr(Zv) + Zv*B))*dZdT(p, T, Zv)
        )
    );
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::dZdT
(
    const scalar p,
    const scalar T
) const
{
    return dZdT(p, T, Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::dZdT
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    const scalar aAlpha = coef1_ - coef2_*sqrt(T) + coef3_*T;
    const scalar A =
        aAlpha*p/sqr(Foam::constant::thermodynamic::RR*T);
    const scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);

    const scalar numerator =
       -(A + Zv + 2*B*Zv)*b_*p
       /(Foam::constant::thermodynamic::RR*T*T)
      + ((B - Zv)*(coef3_*T - coef1_)/(2*T))
       *(p/sqr(Foam::constant::thermodynamic::RR*T));

    return numerator/(3*sqr(Zv) - 2*Zv + A - B - sqr(B));
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Specie>
inline Foam::SRKGas<Specie>::SRKGas
(
    const Specie& sp,
    const scalar& b,
    const scalar& coef1,
    const scalar& coef2,
    const scalar& coef3,
    const scalar& c
)
:
    Specie(sp),
    b_(b),
    coef1_(coef1),
    coef2_(coef2),
    coef3_(coef3),
    c_(c),
    cq0_(0), cq1_(0), cq2_(0), cTlo_(0), cThi_(0)
{}


template<class Specie>
inline Foam::SRKGas<Specie>::SRKGas
(
    const word& name,
    const SRKGas& srk
)
:
    Specie(name, srk),
    b_(srk.b_),
    coef1_(srk.coef1_),
    coef2_(srk.coef2_),
    coef3_(srk.coef3_),
    c_(srk.c_),
    cq0_(srk.cq0_), cq1_(srk.cq1_), cq2_(srk.cq2_),
    cTlo_(srk.cTlo_), cThi_(srk.cThi_)
{}


template<class Specie>
inline Foam::autoPtr<Foam::SRKGas<Specie>>
Foam::SRKGas<Specie>::clone() const
{
    return autoPtr<SRKGas<Specie>>
    (
        new SRKGas<Specie>(*this)
    );
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::cAt(const scalar T) const
{
    // Constant Peneloux unless a c(T) liquid branch is configured.
    if (cTlo_ <= 0)
    {
        return c_;
    }
    if (T <= cTlo_)
    {
        return cq0_ + cq1_*T + cq2_*T*T;          // liquid quadratic
    }
    if (T >= cThi_)
    {
        return c_;                                 // gas baseline
    }
    // smoothstep ramp from the liquid value at cTlo_ to the gas baseline c_
    const scalar cL = cq0_ + cq1_*cTlo_ + cq2_*cTlo_*cTlo_;
    scalar s = (T - cTlo_)/(cThi_ - cTlo_);
    s = s*s*(3.0 - 2.0*s);
    return cL*(1.0 - s) + c_*s;
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::rho
(
    scalar p,
    scalar T
) const
{
    // RE-DIAGNOSIS TEST (2026-07-02): EOS-internal pressure floor to guarantee
    // POSITIVE density. rho = p/(Zv R T) ~ p, so a negative-pressure undershoot
    // in the pEqn (swirl low-p core + stiff cold LOX) yields a NEGATIVE density
    // (min rho = -1133 kg/m3 observed at the injector) -> garbage mass flux
    // rho*U -> runaway growing spike -> dt collapse/crash. Flooring p only
    // INSIDE the EOS keeps rho>0 and breaks the feedback WITHOUT changing the
    // solved p field (unlike the ping-ponging limitPressure clamp). Floor 1e4
    // Pa (0.1 bar); operating p = 5.25e6, so only near-/sub-zero cells are hit.
    p = max(p, scalar(1e4));

    // Branch-consistent blend, for the SAME reason as h/Cp/e/sp/CpMCv above
    // and -- critically -- to stay consistent with psi(). rho = p/(Zv R T)
    // is NONLINEAR in Zv, so rho(w*Zmin+(1-w)*Zmax) != w*rho(Zmin)+
    // (1-w)*rho(Zmax): blending at the Z level and blending at the rho level
    // are different numbers. psi() below deliberately differences each branch
    // against ITSELF and blends the two per-branch slopes with a frozen w, so
    // it is the exact derivative of the rho-LEVEL blend. Taking rho from the
    // Z-level blend instead would leave psi != (d rho/d p) of the rho the
    // solver actually uses -- an inconsistency the pEqn (which couples the
    // two through psis = psi/rho and correctRho(psi*dp)) has no way to
    // absorb. Blend here so the pair is consistent by construction.
    if (stableRoot_ && rootBlendTol_ > 0)
    {
        scalar Zmin, Zmax, lnPhiL, lnPhiV;
        if (cubicBranches(p, T, Zmin, Zmax, lnPhiL, lnPhiV))
        {
            const scalar w = blendWeight(lnPhiL, lnPhiV);
            return w*rhoFromZ(p, T, Zmin) + (1.0 - w)*rhoFromZ(p, T, Zmax);
        }
    }

    return rhoFromZ(p, T, this->Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::rhoFromZ
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    const scalar rhoSRK = p/(Zv*this->R()*T);

    // Peneloux volume translation: v_eff = v_SRK - c(T)  (molar basis).
    // In specific quantities this gives
    //   rho_eff = rho_SRK / (1 - c * rho_SRK / W)
    // where c has units of m^3/kmol and W of kg/kmol, so c/W is m^3/kg.
    // Reference: Peneloux, Rauzy & Freze, Fluid Phase Equilib. 8 (1982) 7-23.
    const scalar c = cAt(T);
    if (c != 0)
    {
        const scalar denom = 1.0 - c*rhoSRK/this->W();
        if (denom > small)
        {
            return rhoSRK/denom;
        }
    }

    return rhoSRK;
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::blendWeight
(
    const scalar lnPhiL,
    const scalar lnPhiV
) const
{
    const scalar dPhi = (lnPhiV - lnPhiL)/rootBlendTol_;
    return 1.0/(1.0 + exp(-max(scalar(-60), min(scalar(60), dPhi))));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::hFromZ
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);
    if (B <= 0)
    {
        B = 1e-16;
    }

    // A branch with Zv <= B is below the co-volume (unphysical, log((Zv+B)/Zv)
    // undefined/singular for Zv <= 0) -- this is exactly the branch that
    // cubicBranches()/Z() reject via lnPhiL = GREAT, so it always carries
    // ~zero blend weight. Evaluating the log here regardless (as the caller
    // does, at BOTH Zmin and Zmax, before applying that weight) trips
    // OpenFOAM's SIGFPE trap. Return the ideal-gas limit (zero departure);
    // it is multiplied by a ~0 weight by the caller so the value itself
    // doesn't matter, only that it stays finite.
    if (Zv <= B)
    {
        return (Foam::constant::thermodynamic::RR*T*(Zv - 1))/this->W();
    }

    const scalar sqrtT = sqrt(T);
    const scalar aAlpha = coef1_ - coef2_*sqrtT + coef3_*T;
    const scalar daAlpha = -coef2_/(2*sqrtT) + coef3_;

    if (B == -Zv)
    {
        return
            (Foam::constant::thermodynamic::RR*T*(Zv - 1))
           /this->W();
    }
    else
    {
        return
            (
                Foam::constant::thermodynamic::RR*T*(Zv - 1)
              + (T*daAlpha - aAlpha)/b_*log((Zv + B)/Zv)
            )/this->W();
    }
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::h
(
    scalar p,
    const scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor (see Z()/rho())

    if (stableRoot_ && rootBlendTol_ > 0)
    {
        scalar Zmin, Zmax, lnPhiL, lnPhiV;
        if (cubicBranches(p, T, Zmin, Zmax, lnPhiL, lnPhiV))
        {
            const scalar w = blendWeight(lnPhiL, lnPhiV);
            return w*hFromZ(p, T, Zmin) + (1.0 - w)*hFromZ(p, T, Zmax);
        }
    }

    return hFromZ(p, T, this->Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::CpFromZ
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    const scalar sqrtT = sqrt(T);
    const scalar aAlpha = coef1_ - coef2_*sqrtT + coef3_*T;
    const scalar daAlpha = -coef2_/(2*sqrtT) + coef3_;
    const scalar ddaAlpha = coef2_/(4*T*sqrtT);
    const scalar A =
        aAlpha*p/sqr(Foam::constant::thermodynamic::RR*T);
    scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);
    if (B <= 0)
    {
        B = 1e-16;
    }

    // See hFromZ(): a branch with Zv <= B is unphysical/GREAT-fugacity and
    // gets ~0 blend weight, but is still evaluated by the caller -- guard
    // against the log()/division singularity rather than SIGFPE.
    if (Zv <= B)
    {
        return 0;
    }

    const scalar M = (sqr(Zv) + B*Zv)/(Zv - B);
    const scalar N = daAlpha*B/(b_*Foam::constant::thermodynamic::RR);

    return
    (
        (T/b_)*ddaAlpha*log((Zv + B)/Zv)
      + Foam::constant::thermodynamic::RR*sqr(M - N)
       /(sqr(M) - A*(2*Zv + B))
      - Foam::constant::thermodynamic::RR
    )/this->W();
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Cp
(
    scalar p,
    scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor (see Z()/rho())

    if (stableRoot_ && rootBlendTol_ > 0)
    {
        scalar Zmin, Zmax, lnPhiL, lnPhiV;
        if (cubicBranches(p, T, Zmin, Zmax, lnPhiL, lnPhiV))
        {
            const scalar w = blendWeight(lnPhiL, lnPhiV);
            return w*CpFromZ(p, T, Zmin) + (1.0 - w)*CpFromZ(p, T, Zmax);
        }
    }

    return CpFromZ(p, T, this->Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::eFromZ
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    // e_departure = h_departure without the (Z-1) ideal-gas PV work term
    // Following the PengRobinsonGas pattern: e = h - RR*T*(Z-1)/W
    scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);
    if (B <= 0)
    {
        B = 1e-16;
    }

    // See hFromZ(): unphysical branch (~0 blend weight) -- avoid the
    // log()/division singularity.
    if (Zv <= B || B == -Zv)
    {
        // Departure vanishes
        return 0;
    }
    else
    {
        const scalar sqrtT = sqrt(T);
        const scalar aAlpha = coef1_ - coef2_*sqrtT + coef3_*T;
        const scalar daAlpha = -coef2_/(2*sqrtT) + coef3_;

        return
            (
                (T*daAlpha - aAlpha)/b_*log((Zv + B)/Zv)
            )/this->W();
    }
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::e
(
    scalar p,
    const scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor (see Z()/rho())

    if (stableRoot_ && rootBlendTol_ > 0)
    {
        scalar Zmin, Zmax, lnPhiL, lnPhiV;
        if (cubicBranches(p, T, Zmin, Zmax, lnPhiL, lnPhiV))
        {
            const scalar w = blendWeight(lnPhiL, lnPhiV);
            return w*eFromZ(p, T, Zmin) + (1.0 - w)*eFromZ(p, T, Zmax);
        }
    }

    return eFromZ(p, T, this->Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Cv
(
    scalar p,
    scalar T
) const
{
    // Cv = Cp - CpMCv; both are already branch-blend-aware, so composing
    // them here automatically inherits that consistency (no direct Zv use).
    return this->Cp(p, T) - this->CpMCv(p, T);
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::spFromZ
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    const scalar daAlpha = -coef2_/(2*sqrt(T)) + coef3_;
    scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);
    if (B <= 0)
    {
        B = 1e-16;
    }

    // See hFromZ(): unphysical branch (~0 blend weight) -- log(Zv-B) is
    // singular/undefined here, avoid it.
    if (Zv <= B)
    {
        return 0;
    }

    return
    (
        Foam::constant::thermodynamic::RR
       *(
            -log(p/Pstd)
          + log(Zv - B)
        )
      - daAlpha/b_*log((Zv + B)/Zv)
    )/this->W();
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::sp
(
    scalar p,
    const scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor (see Z()/rho())

    if (stableRoot_ && rootBlendTol_ > 0)
    {
        scalar Zmin, Zmax, lnPhiL, lnPhiV;
        if (cubicBranches(p, T, Zmin, Zmax, lnPhiL, lnPhiV))
        {
            const scalar w = blendWeight(lnPhiL, lnPhiV);
            return w*spFromZ(p, T, Zmin) + (1.0 - w)*spFromZ(p, T, Zmax);
        }
    }

    return spFromZ(p, T, this->Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::sv
(
    const scalar p,
    const scalar T
) const
{
    // sv = sp - CpMCv/T (approximate)
    return sp(p, T) - CpMCv(p, T)/max(T, small);
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::psi
(
    scalar p,
    scalar T
) const
{
    // psi = (drho/dp)_T, the REAL SRK isothermal compressibility, from a
    // forward finite difference of the Peneloux-corrected density rho(p,T) in
    // p (floored positive). Required by the PEP pressure-evolution pEqn
    // (fgmFluid::pressureCorrector), whose compressibility coefficient
    // psis = psi/rho must reflect the true dense-fluid stiffness -- the
    // ideal-gas rho/p is ~100x too soft for liquid LOX and mis-develops the
    // pressure-velocity coupling (Attempt-1 symptom: cold slab warmed 143->286K).
    //
    // HISTORY: psi was held at rho/p = 1/(Z R T) while the OLD continuity pEqn
    // (isothermalFluid::correctPressure) used it as the diagonal regularisation
    // psi*ddt(p), which the tiny real drho/dp destroyed (Track B 2026-06-30:
    // time scale 1.7e-14 vs 1.3e-10; the A1 compressibility floor was also
    // tested and reverted). That pEqn is now OVERRIDDEN by the PEP scheme, so
    // the real value is used here.
    p = max(p, scalar(1e4));   // RE-DIAGNOSIS TEST: EOS-internal p-floor (see rho())
    const scalar dp = 1e-3*p;

    if (stableRoot_ && rootBlendTol_ > 0)
    {
        // Branch-consistent finite difference. A naive FD of the blended
        // rho(p,T) (rho(p+dp)-rho(p))/dp re-evaluates the logistic weight w
        // at BOTH p and p+dp independently; if the blend crossing (width
        // ~rootBlendTol_ in lnPhi, which can correspond to a p-window far
        // narrower than the fixed dp=1e-3*p step) falls between them, the FD
        // effectively differences across the whole Zmin->Zmax jump instead
        // of sampling the local slope -- an artificially huge/erratic psi
        // that (via the globally-coupled elliptic pEqn) was observed to
        // inflate the WHOLE-domain pressure field during the cold-start
        // transient (2-3x nominal, PEP and LAD Spike Suppression wiki sec
        // 14), even far from any near-critical cell. Fix: freeze the blend
        // weight w at p, difference each branch (Zmin, Zmax) against ITSELF
        // at p and p+dp (each branch is the analytic continuation of one
        // cubic root and has no jump), then blend the two smooth per-branch
        // psi's with the same frozen w -- never finite-differences across
        // the crossing.
        scalar Zmin0, Zmax0, lnPhiL0, lnPhiV0;
        if (cubicBranches(p, T, Zmin0, Zmax0, lnPhiL0, lnPhiV0))
        {
            scalar Zmin1, Zmax1, lnPhiL1, lnPhiV1;
            cubicBranches(p + dp, T, Zmin1, Zmax1, lnPhiL1, lnPhiV1);

            const scalar dPhi = (lnPhiV0 - lnPhiL0)/rootBlendTol_;
            const scalar w =
                1.0/(1.0 + exp(-max(scalar(-60), min(scalar(60), dPhi))));

            const scalar rhoMin0 = rhoFromZ(p, T, Zmin0);
            const scalar rhoMin1 = rhoFromZ(p + dp, T, Zmin1);
            const scalar rhoMax0 = rhoFromZ(p, T, Zmax0);
            const scalar rhoMax1 = rhoFromZ(p + dp, T, Zmax1);

            const scalar psiMin = (rhoMin1 - rhoMin0)/dp;
            const scalar psiMax = (rhoMax1 - rhoMax0)/dp;
            const scalar dRhodP = w*psiMin + (1.0 - w)*psiMax;
            const scalar rho0 = w*rhoMin0 + (1.0 - w)*rhoMax0;

            return max(dRhodP, 1e-3*rho0/p);
        }
        // Single real root at p: no blend possible here, fall through to
        // the plain FD below (identical to the non-blended path).
    }

    const scalar rho0 = this->rho(p, T);
    const scalar rho1 = this->rho(p + dp, T);
    const scalar dRhodP = (rho1 - rho0)/dp;
    return max(dRhodP, 1e-3*rho0/p);   // floor at 0.1% of rho/p
}


template<class Specie>
inline bool Foam::SRKGas<Specie>::cubicBranches
(
    const scalar p,
    const scalar T,
    scalar& Zmin,
    scalar& Zmax,
    scalar& lnPhiL,
    scalar& lnPhiV
) const
{
    const scalar aAlpha = coef1_ - coef2_*sqrt(T) + coef3_*T;

    const scalar A =
        aAlpha*p/sqr(Foam::constant::thermodynamic::RR*T);
    const scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);

    const scalar a2 = -1.0;
    const scalar a1 = A - B - sqr(B);
    const scalar a0 = -A*B;

    const scalar Q = (3*a1 - a2*a2)/9.0;
    const scalar Rl = (9*a2*a1 - 27*a0 - 2*a2*a2*a2)/54.0;

    const scalar Q3 = Q*Q*Q;
    const scalar D = Q3 + Rl*Rl;

    if (D > 0)
    {
        const scalar D05 = sqrt(D);
        scalar S = (Rl + D05 < 0) ? -cbrt(mag(Rl + D05)) : cbrt(Rl + D05);
        scalar Tl = (D05 > Rl) ? -cbrt(mag(Rl - D05)) : cbrt(Rl - D05);
        Zmin = Zmax = S + Tl - a2/3.0;
        lnPhiL = lnPhiV = 0;
        return false;
    }

    const scalar th = ::acos(max(scalar(-1), min(scalar(1), Rl/sqrt(-Q3))));
    const scalar qm = 2*sqrt(-Q);
    const scalar pi = constant::mathematical::pi;
    const scalar r[3] =
    {
        qm*cos(th/3.0) - a2/3.0,
        qm*cos((th + 2*pi)/3.0) - a2/3.0,
        qm*cos((th + 4*pi)/3.0) - a2/3.0
    };

    Zmin = min(r[0], min(r[1], r[2]));
    Zmax = max(r[0], max(r[1], r[2]));

    lnPhiL =
        (Zmin > B)
      ? Zmin - 1 - log(Zmin - B) - (A/B)*log(1 + B/Zmin)
      : GREAT;
    lnPhiV = Zmax - 1 - log(Zmax - B) - (A/B)*log(1 + B/Zmax);
    return true;
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Z
(
    scalar p,
    scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor at the root solver: the (p,T) cache
    // key and every caller's B then use the same floored p (see rho()).
    // Tier-1: reuse the cubic root if (p,T) unchanged since the last solve
    // (rho/psi/Cp/mu/kappa all query Z at the same cell state).
    if (p == pZcache_ && T == TZcache_)
    {
        return Zcache_;
    }

    const scalar aAlpha = coef1_ - coef2_*sqrt(T) + coef3_*T;

    const scalar A =
        aAlpha*p/sqr(Foam::constant::thermodynamic::RR*T);
    const scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);

    const scalar a2 = -1.0;
    const scalar a1 = A - B - sqr(B);
    const scalar a0 = -A*B;

    const scalar Q = (3*a1 - a2*a2)/9.0;
    const scalar Rl = (9*a2*a1 - 27*a0 - 2*a2*a2*a2)/54.0;

    const scalar Q3 = Q*Q*Q;
    const scalar D = Q3 + Rl*Rl;

    scalar root = -1;

    if (D <= 0)
    {
        // Three real roots: smallest = liquid, largest = vapour, middle =
        // mechanically unstable (dp/dv > 0, never the stable phase). Select
        // the thermodynamically stable branch by minimum fugacity, i.e.
        // minimum Gibbs energy (mu = mu0 + R T ln f), using the SRK
        // analytic fugacity coefficient
        //   ln(phi) = Z - 1 - ln(Z - B) - (A/B) ln(1 + B/Z).
        // At p > psat(T) the liquid root wins, at p < psat the vapour root,
        // and both coincide on the saturation line (Maxwell construction).
        // The old max-root choice always returned the vapour branch, which
        // for compressed-liquid states (e.g. LOX at 100-120 K, ~5 MPa)
        // overestimates v by an order of magnitude. Pattern identical to
        // elyHanleyTransport::srkMethaneRho (Pedersen/marlim3, validated in
        // test/proto_eosdensity.py).
        const scalar th = ::acos(max(scalar(-1), min(scalar(1), Rl/sqrt(-Q3))));
        const scalar qm = 2*sqrt(-Q);
        const scalar pi = constant::mathematical::pi;
        const scalar r[3] =
        {
            qm*cos(th/3.0) - a2/3.0,
            qm*cos((th + 2*pi)/3.0) - a2/3.0,
            qm*cos((th + 4*pi)/3.0) - a2/3.0
        };

        const scalar Zmin = min(r[0], min(r[1], r[2]));
        const scalar Zmax = max(r[0], max(r[1], r[2]));

        if (stableRoot_)
        {
            // Minimum-fugacity (thermodynamically stable phase) selection.
            // SRK fugacity coefficient
            //   ln(phi) = Z - 1 - ln(Z - B) - (A/B) ln(1 + B/Z).
            // The smaller ln(phi) (= smaller molar Gibbs) is the stable phase:
            // the liquid root (Zmin) wins for p > psat(T), the vapour root
            // (Zmax) for p < psat, and they coincide on the saturation line
            // (Maxwell construction). This recovers the compressed-liquid
            // density (e.g. cryogenic LOX at 100-120 K, p < psat) that the
            // legacy max-root choice underestimates by up to an order of
            // magnitude. Trade-off: Z (hence h) is discontinuous across the
            // saturation line, so an h->T Newton inversion can fail ("Maximum
            // number of iterations exceeded") -- safe for the fgmFluid path
            // (T is read from the manifold table, not inverted from h).
            const scalar lnPhiL =
                (Zmin > B)
              ? Zmin - 1 - log(Zmin - B) - (A/B)*log(1 + B/Zmin)
              : GREAT;                  // liquid root below co-volume: reject
            const scalar lnPhiV =
                Zmax - 1 - log(Zmax - B) - (A/B)*log(1 + B/Zmax);

            if (rootBlendTol_ > 0)
            {
                // Smooth (logistic) blend across a ln(fugacity) deadband,
                // instead of the hard argmin below. Near the Widom line,
                // adjacent cells at nearly identical (T,p) can sit on
                // opposite sides of lnPhiL == lnPhiV for p perturbations as
                // small as 0.05-3%, so the hard switch flips the whole
                // liquid/vapour branch (rho jump 2-5x) cell-to-cell -- a
                // discontinuous-classifier artefact, not physics (see PEP
                // and LAD Spike Suppression wiki secs 6-10). w in (0,1)
                // interpolates Z smoothly liquid(Zmin)->vapour(Zmax) over a
                // band of width ~rootBlendTol_ in lnPhi, collapsing to the
                // exact hard switch far from the crossing.
                const scalar dPhi = (lnPhiV - lnPhiL)/rootBlendTol_;
                const scalar w =
                    1.0/(1.0 + exp(-max(scalar(-60), min(scalar(60), dPhi))));
                root = w*Zmin + (1.0 - w)*Zmax;
            }
            else
            {
                root = (lnPhiL <= lnPhiV) ? Zmin : Zmax;
            }
        }
        else
        {
            // Legacy default: always the largest root (vapour branch), matching
            // the OFv8 soaveRedlichKwong reference. Continuous in T, so the
            // enthalpy->temperature Newton inversion (species::thermo::T) is
            // robust (validated 67-day run), at the cost of compressed-liquid
            // specific-volume accuracy. See SRKGas.H stableRoot_.
            root = Zmax;
        }
    }
    else
    {
        // One root is real
        const scalar D05 = sqrt(D);
        scalar S = 0;

        if (Rl + D05 < 0)
        {
            S = -cbrt(mag(Rl + D05));
        }
        else
        {
            S = cbrt(Rl + D05);
        }
        scalar Tl = 0;
        if (D05 > Rl)
        {
            Tl = -cbrt(mag(Rl - D05));
        }
        else
        {
            Tl = cbrt(Rl - D05);
        }

        root = S + Tl - a2/3.0;
    }

    pZcache_ = p;
    TZcache_ = T;
    Zcache_ = root;
    return root;
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::CpMCvFromZ
(
    const scalar p,
    const scalar T,
    const scalar Zv
) const
{
    const scalar aAlpha = coef1_ - coef2_*sqrt(T) + coef3_*T;
    const scalar daAlpha = -coef2_/(2*sqrt(T)) + coef3_;

    const scalar A =
        aAlpha*p/sqr(Foam::constant::thermodynamic::RR*T);
    scalar B = b_*p/(Foam::constant::thermodynamic::RR*T);
    if (B <= 0)
    {
        B = 1e-16;
    }

    // See hFromZ(): unphysical branch (~0 blend weight) -- M = (...)/(Zv-B)
    // is singular here, avoid it.
    if (Zv <= B)
    {
        return 0;
    }

    const scalar M = (sqr(Zv) + B*Zv)/(Zv - B);
    const scalar N = daAlpha*B/(b_*Foam::constant::thermodynamic::RR);

    return this->R()*sqr(M - N)/(sqr(M) - A*(2*Zv + B));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::CpMCv
(
    scalar p,
    scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor (see Z()/rho())

    if (stableRoot_ && rootBlendTol_ > 0)
    {
        scalar Zmin, Zmax, lnPhiL, lnPhiV;
        if (cubicBranches(p, T, Zmin, Zmax, lnPhiL, lnPhiV))
        {
            const scalar w = blendWeight(lnPhiL, lnPhiV);
            return
                w*CpMCvFromZ(p, T, Zmin) + (1.0 - w)*CpMCvFromZ(p, T, Zmax);
        }
    }

    return CpMCvFromZ(p, T, this->Z(p, T));
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::alphav
(
    const scalar p,
    const scalar T
) const
{
    // Numerical differentiation: alphav = -(1/rho)(drho/dT)_p
    const scalar dT = min(T*1e-6, 1e-4);
    const scalar rhoT = rho(p, T);

    if (mag(rhoT) < small)
    {
        return 0;
    }

    const scalar rhoTp = rho(p, T + dT);
    const scalar rhoTm = rho(p, T - dT);

    return -(rhoTp - rhoTm)/(2*dT*rhoT);
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::c2
(
    scalar p,
    const scalar T
) const
{
    p = max(p, scalar(1e4)); // EOS p-floor (see Z()/rho())
    // Explicit speed of sound squared from Oefelein (AIAA 2019-0634) Eqs.
    // (88)-(91):
    //   c^2 = gamma * R * T / (Zp / Z)
    //   gamma = Cp / Cv
    // Cp, Cv already include departure contributions so they are evaluated
    // via numerical Cp - (Cp - Cv) = Cv at the same (p, T).
    // Here Zp and Z are packed implicitly through the CpMCv / Z routines.
    const scalar Zv = this->Z(p, T);
    const scalar Cpv = this->Cp(p, T);
    const scalar CpMCvv = this->CpMCv(p, T);
    const scalar gamma = Cpv/max(Cpv - CpMCvv, small);

    // (drho/dp)_T using analytic SRK pressure derivative; for efficiency we
    // use the isothermal finite difference since this routine is only used
    // for post-processing.
    const scalar dp = max(1e-3*p, 1.0);
    const scalar drho = rho(p + dp, T) - rho(p - dp, T);
    const scalar dRhodP_T = drho/(2.0*dp);

    if (dRhodP_T <= small)
    {
        return gamma*this->R()*T;
    }

    return gamma/dRhodP_T;
}


template<class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::lnPhi
(
    const scalar p,
    const scalar T
) const
{
    // Pure-species natural log of fugacity coefficient for SRK.
    // Soave, Chem. Eng. Sci. 27 (1972) 1197-1203, as presented e.g. in
    // Reid, Prausnitz & Poling "The Properties of Gases and Liquids",
    // 4th ed., McGraw-Hill 1987, Ch. 3.
    //   ln(phi) = (Z - 1) - ln(Z - B) - (A/B) ln((Z + B)/Z)
    // Matches the pure-fluid specialisation (b_i = b_m, delta_i = 1) of
    // Oefelein Eq. (99).
    const scalar sqrtT = sqrt(T);
    const scalar aAlpha = coef1_ - coef2_*sqrtT + coef3_*T;
    const scalar RR = Foam::constant::thermodynamic::RR;
    const scalar A = aAlpha*p/sqr(RR*T);
    scalar B = b_*p/(RR*T);
    if (B <= 0)
    {
        B = 1e-16;
    }

    const scalar Zv = this->Z(p, T);

    if (Zv - B <= small)
    {
        return 0;
    }

    return (Zv - 1.0) - log(Zv - B) - (A/B)*log((Zv + B)/Zv);
}


// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template<class Specie>
inline void Foam::SRKGas<Specie>::operator+=
(
    const SRKGas<Specie>& srk
)
{
    scalar X1 = this->Y()/this->W();
    Specie::operator+=(srk);

    if (mag(this->Y()) > small)
    {
        X1 *= this->W()/this->Y();
        const scalar X2 = this->W()*srk.Y()/(srk.W()*this->Y());

        b_ = X1*b_ + X2*srk.b_;
        coef1_ = X1*coef1_ + X2*srk.coef1_;
        coef2_ = X1*coef2_ + X2*srk.coef2_;
        coef3_ = X1*coef3_ + X2*srk.coef3_;
        // Peneloux shift is a linear quantity in mole fractions.
        c_ = X1*c_ + X2*srk.c_;
    }
}


template<class Specie>
inline void Foam::SRKGas<Specie>::operator*=(const scalar s)
{
    Specie::operator*=(s);
}


// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //

template<class Specie>
Foam::SRKGas<Specie> Foam::operator+
(
    const SRKGas<Specie>& srk1,
    const SRKGas<Specie>& srk2
)
{
    Specie sp
    (
        static_cast<const Specie&>(srk1)
      + static_cast<const Specie&>(srk2)
    );

    if (mag(sp.Y()) < small)
    {
        return SRKGas<Specie>
        (
            sp,
            srk1.b_,
            srk1.coef1_,
            srk1.coef2_,
            srk1.coef3_,
            srk1.c_
        );
    }
    else
    {
        const scalar X1 = sp.W()*srk1.Y()/(srk1.W()*sp.Y());
        const scalar X2 = sp.W()*srk2.Y()/(srk2.W()*sp.Y());

        return SRKGas<Specie>
        (
            sp,
            X1*srk1.b_ + X2*srk2.b_,
            X1*srk1.coef1_ + X2*srk2.coef1_,
            X1*srk1.coef2_ + X2*srk2.coef2_,
            X1*srk1.coef3_ + X2*srk2.coef3_,
            X1*srk1.c_ + X2*srk2.c_
        );
    }
}


template<class Specie>
Foam::SRKGas<Specie> Foam::operator*
(
    const scalar s,
    const SRKGas<Specie>& srk
)
{
    return SRKGas<Specie>
    (
        s*static_cast<const Specie&>(srk),
        srk.b_,
        srk.coef1_,
        srk.coef2_,
        srk.coef3_,
        srk.c_
    );
}


template<class Specie>
Foam::SRKGas<Specie> Foam::operator==
(
    const SRKGas<Specie>& srk1,
    const SRKGas<Specie>& srk2
)
{
    Specie sp
    (
        static_cast<const Specie&>(srk1)
     == static_cast<const Specie&>(srk2)
    );

    return SRKGas<Specie>
    (
        sp,
        NaN,
        NaN,
        NaN,
        NaN,
        NaN
    );
}


// ************************************************************************* //
