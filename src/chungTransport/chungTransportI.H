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

\*---------------------------------------------------------------------------*/

#include "rfSpecie.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Thermo>
inline Foam::chungTransport<Thermo>::chungTransport
(
    const Thermo& t,
    const scalar& sigmaM,
    const scalar& epsilonkM,
    const scalar& MM,
    const scalar& VcM,
    const scalar& TcM,
    const scalar& omegaM,
    const scalar& miuiM,
    const scalar& kappaiM,
    const List<scalar>& Ymd,
    const List<scalar>& Xmd,
    const List<List<scalar>>& Tcmd,
    const List<List<scalar>>& Pcmd,
    const List<List<scalar>>& Mmd,
    const List<List<scalar>>& sigmd
)
:
    Thermo(t),
    sigmaM_(sigmaM),
    epsilonkM_(epsilonkM),
    MM_(MM),
    VcM_(VcM),
    TcM_(TcM),
    omegaM_(omegaM),
    miuiM_(miuiM),
    kappaiM_(kappaiM),
    Ymd_(Ymd),
    Xmd_(Xmd),
    Tcmd_(Tcmd),
    Pcmd_(Pcmd),
    Mmd_(Mmd),
    sigmd_(sigmd),
    cacheP_(0), cacheT_(-1), cacheTstar_(0), cacheMur_(0),
    cacheEta0_(0), cacheY_(0), cacheG1_(0), sumXmd_(0)
{
    // Maintain the invariant sumXmd_ == Sum(Xmd_) at construction so mu()/kappa()
    // are correct even if called before updateTRANS (they read the cached sum).
    forAll(Xmd_, i) { sumXmd_ += Xmd_[i]; }
}


template<class Thermo>
inline Foam::chungTransport<Thermo>::chungTransport
(
    const word& name,
    const chungTransport& ch
)
:
    Thermo(name, ch),
    sigmaM_(ch.sigmaM_),
    epsilonkM_(ch.epsilonkM_),
    MM_(ch.MM_),
    VcM_(ch.VcM_),
    TcM_(ch.TcM_),
    omegaM_(ch.omegaM_),
    miuiM_(ch.miuiM_),
    kappaiM_(ch.kappaiM_),
    Ymd_(ch.Ymd_),
    Xmd_(ch.Xmd_),
    Tcmd_(ch.Tcmd_),
    Pcmd_(ch.Pcmd_),
    Mmd_(ch.Mmd_),
    sigmd_(ch.sigmd_),
    cacheP_(0), cacheT_(-1), cacheTstar_(0), cacheMur_(0),
    cacheEta0_(0), cacheY_(0), cacheG1_(0), sumXmd_(0)
{
    // Maintain sumXmd_ == Sum(Xmd_) (see the component constructor).
    forAll(Xmd_, i) { sumXmd_ += Xmd_[i]; }
}


template<class Thermo>
inline Foam::autoPtr<Foam::chungTransport<Thermo>>
Foam::chungTransport<Thermo>::clone() const
{
    return autoPtr<chungTransport<Thermo>>
    (
        new chungTransport<Thermo>(*this)
    );
}


// * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * * //

template<class Thermo>
inline void Foam::chungTransport<Thermo>::calculate
(
    scalar p,
    scalar T,
    scalar& Tstar,
    scalar& mur,
    scalar& eta0,
    scalar& Y,
    scalar& G1
) const
{
    // Single-entry (p,T) cache: RhoFluidThermo evaluates mu(p,T) then
    // kappa(p,T) with the same (p,T) on one transportMixture, so the second
    // call reuses this result -- saving the rho(p,T) cubic solve and the
    // collision-integral Omega* (exp/pow/sin). Invalidated on composition
    // change (updateTRANS). mutable: OpenFOAM serial per-process cell loop.
    if (T == cacheT_ && p == cacheP_)
    {
        Tstar = cacheTstar_; mur = cacheMur_; eta0 = cacheEta0_;
        Y = cacheY_; G1 = cacheG1_;
        return;
    }

    const scalar A=1.16145 , B=0.14874, C=0.52487 , D=0.77320 , E=2.16178;
    const scalar F=2.43787 , G= -6.435e-4, H=7.27371, S=18.0323, W=-0.76830;

    // Calculate T* (=Tstar) in eq (3)
    Tstar = T/epsilonkM_;

    // Calculate mu_r (=mur) in eq (8)
    mur = 131.3*miuiM_/sqrt(VcM_*TcM_);

    // Calculate F_c (=Fc) in eq (7)
    const scalar mur2 = sqr(mur);
    scalar Fc = 1.0 - 0.2756*omegaM_ + 0.059035*sqr(mur2) + kappaiM_;

    // Calculate the reduced collision integral Omega* (=Omegast) in eq (2)
    scalar Omegast = A*pow(Tstar, -B) + C*(exp(-D*Tstar)) + E*exp(-F*Tstar)
                         + G*pow(Tstar, B)*sin(S*pow(Tstar, W) - H);

    // Calculate eta_0 (=eta0) in eq (6)
    const scalar VcM23 = cbrt(sqr(VcM_));
    eta0 = (4.0785e-5)*sqrt(MM_*T)*Fc/(Omegast*VcM23);

    // Call rho from base class and change unit from [kg/m3] to [mol/l]
    scalar rho = 1e-3*(this->rho(p, T))/this->W();

    // Calculate Y term in eq (10)
    Y =  rho*VcM_/6.0;

    // Y-CAP (root guard, 2026-07-18): the dense correction G1=(1-0.5Y)/(1-Y)^3
    // has a hard SIGN-FLIP singularity at Y=1 (packing limit) and the Chung
    // correlation is only calibrated to Y ~< 0.5. Liquid kerosene sits at
    // Y ~ 0.53 in NORMAL operation, so off-manifold excursions of rho or VcM
    // (Tier-2 extrapolation, broken he->T inversion) readily push Y past 1 and
    // produce mu ~ -1e6 (numerically demonstrated). Freeze the correlation at
    // Y=0.9: continuous, sign-preserving, and only engaged on broken states.
    Y = min(Y, scalar(0.9));

    // Calculate G1 term in eq (10)
    const scalar oneMinusY = 1.0 - Y;
    G1 = (1.0 -  0.5*Y)/(oneMinusY*oneMinusY*oneMinusY);

    // store in the (p,T) cache for the matching kappa()/mu() call
    cacheP_ = p; cacheT_ = T;
    cacheTstar_ = Tstar; cacheMur_ = mur; cacheEta0_ = eta0;
    cacheY_ = Y; cacheG1_ = G1;
}


//- Return Dynamic Viscosity [kg/m.s] based on Chung's model (1988)
template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::mu
(
    const scalar p,
    const scalar T
) const
{
    const scalar sumX = sumXmd_;
    scalar finalMu = 0;

    if (sumX <= 1e-16)
    {
         finalMu = 0;
    }
    else
    {
        scalar Tstar=0, mur=0, eta0=0, Y=0, G1=0;
        calculate(p, T, Tstar, mur, eta0, Y, G1);

        // Specify the coefficients in table II, a0(i) -> a3(i), i=1,10
        const scalar a0[10] = {6.32402, 0.0012102, 5.28346, 6.62263, 19.74540,
                           -1.89992, 24.27450, 0.79716, -0.23816, 0.068629};
        const scalar a1[10] = {50.41190, -0.0011536, 254.209, 38.0957, 7.63034,
                           -12.53670, 3.44945, 1.11764, 0.067695,  0.34793};
        const scalar a2[10] = {-51.68, -0.0062571, -168.481, -8.46414, -14.35440,
                            4.98529, -11.29130, 0.012348, -0.8163, 0.59256};
        const scalar a3[10] = {1189.02, 0.037283, 3898.270, 31.4178, 31.52670,
                             -18.1507, 69.3464, -4.1161, 4.02528, -0.72663};

        // Calculate the coefficients Ai in eq (11)
        scalar MA[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        const scalar mur4 = sqr(mur)*sqr(mur);
        for (int i=0; i<10; i++)
        {
            MA[i] = a0[i] + a1[i]*omegaM_ + a2[i]*mur4
                  + a3[i]*kappaiM_;
        }

        // Calculate G2 term in eq (10)
        scalar G2 =
            (MA[0]*(1.0-exp(-MA[3]*Y))/Y + MA[1]*G1*exp(MA[4]*Y) + MA[2]*G1)
           /(MA[0]*MA[3] + MA[1] + MA[2]);

        // Calculate eta_k (=etak) in eq (10)
        scalar etak = eta0*(1.0/G2 + MA[5]*Y);

        // Calculate eta_p (=etap) in eq (10)
        const scalar VcM23 = cbrt(sqr(VcM_));
        scalar etap =
            ((40.785e-6/sqrt(Tstar))*sqrt(MM_*T)/VcM23)
           *MA[6]*Y*Y*G2*exp(MA[7]+MA[8]/Tstar + MA[9]/sqr(Tstar));

        // Return final viscosity and convert from poise [P] to [kg/m.s]
        finalMu = 0.1*(etak+etap);
    }

    // POSITIVITY FLOOR (root fix, 2026-07-18): off-manifold states (e.g. the
    // ignition-kick cells where the Tier-2 coefficients are extrapolated
    // outside the (Z,gZ,C,dh) hull) can drive G2/exp pathological and return a
    // NEGATIVE viscosity, which propagates to nuEff and crashes any sqrt
    // consumer (yPlus FO SIGFPE at every ignition kick). Transport
    // coefficients are positive by physics; floor at 1e-7 kg/(m s) -- two
    // decades below the smallest physical mu here (~1e-5) so the floor only
    // engages on broken states.
    // ...and a CEILING: pathological-but-POSITIVE values also occur (exp
    // blow-up at Tstar << 1 gave mu = 8.4e3 in the standalone scan) and would
    // pass a floor untouched, injecting enormous viscous stiffness. Physical
    // maximum here is liquid kerosene ~1.5e-3; cap at 0.05 (~30x margin).
    return min(max(finalMu, scalar(1e-7)), scalar(0.05));
}


//- Return Thermal Conductivity [W/mK] based on Chung et al. model (1988)
template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::kappa
(
    const scalar p, const scalar T
) const
{
    const scalar sumX = sumXmd_;
    scalar kappa = 0;

    if (sumX <= 0.001)
    {
        kappa = 0;
    }
    else
    {
        scalar Tstar=0, mur=0, eta0=0, Y=0, G1=0;
        calculate(p, T, Tstar, mur, eta0, Y, G1);

        // Specify the coefficients in table IV, b0(i) -> b3(i), i=1,7
        const scalar b0[7] = {2.41657, -0.50924, 6.61069, 14.54250, 0.79274,
                              -5.8634, 81.171};
        const scalar b1[7] = {0.74824, -1.50936, 5.62073, -8.91387, 0.82019,
                              12.8005, 114.1580};
        const scalar b2[7] = {-0.91858, -49.99120, 64.7599, -5.63794, -0.69369,
                               9.58926, -60.841};
        const scalar b3[7] = {121.721, 69.9834, 27.0389, 74.3435, 6.31734,
                              -65.5292, 466.7750};

        // Calculate the coefficients Bi in eq (13)
        scalar MB[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        const scalar mur4 = sqr(mur)*sqr(mur);
        for (int j=0; j<7; j++)
        {
            MB[j] = b0[j] + b1[j]*omegaM_ + b2[j]*mur4
                  + b3[j]*kappaiM_;
        }

        // Heat capacity at constant volume in [cal/mol.K].
        // Chung's alpha = Cv/R - 3/2 requires the IDEAL-GAS Cv0, NOT the
        // real Cv. The thermo chain returns
        //   janafThermo<EOS>::Cp(p,T) = JANAF_poly(T) + EOS::Cp(p,T)
        // i.e. Cp_real = Cp_ig + Cp_dep_SRK, so it carries the dense-fluid
        // departure (the pseudo-critical Cp peak, e.g. O2 near Tc/100 bar
        // shows Cp_real ~ 4500 vs Cp_ig ~ 920 J/kg/K). Feeding that into
        // alpha inflates Psi and over-predicts kappa in the dense region.
        // Recover Cp_ig by evaluating at a low reference pressure where the
        // SRK departure is negligible AND the fluid stays single-phase vapour.
        // 1 atm fails near a species' 1-atm boiling point (e.g. H2O at 373 K,
        // O2 at 90 K) where the cubic returns a compressed-liquid root, so
        // Cp(1 atm) carries a liquid departure and Cv0 jumps. p_ref = 100 Pa
        // keeps all species vapour over the operating range. Mirrors
        // elyHanleyTransport.
        const scalar pRefCpIg = 100.0;
        scalar CvIdeal = this->Cp(pRefCpIg, T) - this->R();
        scalar Cv = CvIdeal*this->W()*0.2388e-3;

        // Universal gas constant [cal/mol.K]
        scalar R = RR/(4.186*1e3);

        // Calculate alpha in eq (9)
        scalar alpha = (Cv/R) - 3.0/2.0;

        // Calculate beta in eq (9)
        // Chung et al. (1988) / Poling 5th ed. Eq (10-3.14): the omega^2
        // coefficient is 1.3168 (the previous 1.1368 was a transposition;
        // negligible for low-omega O2 but matters for H2O omega=0.344,
        // kerosene omega=0.467).
        scalar beta = 0.7862-0.7109*omegaM_ + 1.3168*sqr(omegaM_);

        // Calculate Tr in eq (9)
        scalar Tr = T/TcM_;

        // Calculate Z in eq (9)
        scalar Z = 2.0 + 10.5*sqr(Tr);

        // Calculate Psi in eq (9)
        scalar Psi = 1.0 + alpha*(0.215 + 0.28288*alpha - 1.061*beta
                   + 0.26665*Z)/(0.6366 + beta*Z + 1.061*alpha*beta);

        // Calculate lambda0 in eq (9)
        scalar lamda0 = 7.452*(eta0/MM_)*Psi;

        // Calculate H2 in eq (12)
        scalar H2 =
            (MB[0]*(1.0-exp(-MB[3]*Y))/Y + MB[1]*G1*exp(MB[4]*Y) + MB[2]*G1)
           /(MB[0]*MB[3] + MB[1] + MB[2]);

        // Calculate lambda_k in eq (12)
        scalar lamdak = lamda0*(1.0/H2 + MB[5]*Y);

        // Calculate lambda_p in eq (12)
        const scalar VcM23 = cbrt(sqr(VcM_));
        scalar lamdap =
            (3.039e-4*sqrt(TcM_/MM_)/VcM23)
           *MB[6]*sqr(Y)*H2*sqrt(Tr);

        // Convert from [cal/cm.s.K] to [W/m.K]
        kappa = 418.6798*(lamdak+lamdap);
    }

    // POSITIVITY FLOOR: same off-manifold pathology as mu() above -- a
    // negative conductivity corrupts alphaEff/energy diffusion. Floor at
    // 1e-4 W/(m K), two decades below the smallest physical kappa here.
    // Floor AND ceiling (see mu() above): physical max ~0.15 W/(m K)
    // (liquid O2); cap at 1.0.
    return min(max(kappa, scalar(1e-4)), scalar(1.0));
}


//- Return coefficient phi based on Takahashi model
template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::phi
(
    scalar PRR,
    scalar TRR
) const
{
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
    const scalar C25 = 0.000000, E25 = 0.00000;
    const scalar A30 = 0.502100, B30 = 2.39031;
    const scalar C30 = 0.602907, E30 = 6.19043;
    const scalar A40 = 0.837452, B40 = 3.23513;
    const scalar C40 = 0.000000, E40 = 0.00000;
    const scalar A50 = 0.890390, B50 = 3.13001;

    scalar Y=0, YA=0, YB=0;
    TRR = (TRR < 1.0) ? 1.0 : TRR;
    TRR = (TRR > 5.0) ? 5.0 : TRR;
    PRR = (PRR > 5.0) ? 5.0 : PRR;

    if (PRR < 0.1)
    {
        Y = 1.01 * (1.0 - A01 * pow(TRR, -B01));
        Y = 1.0 + (Y - 1.0)/(0.1-0.0)*(PRR-0.0);
    }
    else if ((PRR >= 0.1) && (PRR < 0.2))
    {
        YA = 1.01 * (1.0 - A01 * pow(TRR, -B01));
        YB = 1.01 * (1.0 - A02 * pow(TRR, -B02));
        Y  = YA + (YB-YA)/(0.2-0.1)*(PRR-0.1);
    }
    else if ((PRR >= 0.2) && (PRR < 0.3))
    {
        YA = 1.01 * (1.0 - A02 * pow(TRR, -B02));
        YB = 1.01 * (1.0 - A03 * pow(TRR, -B03));
        Y  = YA + (YB-YA)/(0.3-0.2)*(PRR-0.2);
    }
    else if ((PRR >= 0.3) && (PRR < 0.4))
    {
        YA = 1.01 * (1.0 - A03 * pow(TRR, -B03));
        YB = 1.01 * (1.0 - A04 * pow(TRR, -B04));
        Y  = YA + (YB-YA)/(0.4-0.3)*(PRR-0.3);
    }
    else if ((PRR >= 0.4) && (PRR < 0.5))
    {
        YA = 1.01 * (1.0 - A04 * pow(TRR, -B04));
        YB = 1.01 * (1.0 - A05 * pow(TRR, -B05));
        Y  = YA + (YB-YA)/(0.5-0.4)*(PRR-0.4);
    }
    else if ((PRR >= 0.5) && (PRR < 0.6))
    {
        YA = 1.01 * (1.0 - A05 * pow(TRR, -B05));
        YB = 1.01 * (1.0 - A06 * pow(TRR, -B06));
        Y  = YA + (YB-YA)/(0.6-0.5)*(PRR-0.5);
    }
    else if ((PRR >= 0.6) && (PRR < 0.8))
    {
        YA = 1.01 * (1.0 - A06 * pow(TRR, -B06));
        YB = 1.01 * (1.0 - A08 * pow(TRR, -B08));
        Y  = YA + (YB-YA)/(0.8-0.6)*(PRR-0.6);
    }
    else if ((PRR >= 0.8) && (PRR < 1.0))
    {
        YA = 1.01 * (1.0 - A08 * pow(TRR, -B08));
        YB = 1.02 * (1.0 - A10 * pow(TRR, -B10))*(1.0 - C10*pow(TRR, -E10));
        Y  = YA + (YB-YA)/(1.0-0.8)*(PRR-0.8);
    }
    else if ((PRR >= 1.0) && (PRR < 1.2))
    {
        YA = 1.02 * (1.0 - A10 * pow(TRR, -B10))*(1.0 - C10*pow(TRR, -E10));
        YB = 1.02 * (1.0 - A12 * pow(TRR, -B12))*(1.0 - C12*pow(TRR, -E12));
        Y  = YA + (YB-YA)/(1.2-1.0)*(PRR-1.0);
    }
    else if ((PRR >= 1.2) && (PRR < 1.4))
    {
        YA = 1.02 * (1.0 - A12 * pow(TRR, -B12))*(1.0 - C12*pow(TRR, -E12));
        YB = 1.02 * (1.0 - A14 * pow(TRR, -B14))*(1.0 - C14*pow(TRR, -E14));
        Y  = YA + (YB-YA)/(1.4-1.2)*(PRR-1.2);
    }
    else if ((PRR >= 1.4) && (PRR < 1.6))
    {
        YA = 1.02 * (1.0 - A14 * pow(TRR, -B14))*(1.0 - C14*pow(TRR, -E14));
        YB = 1.02 * (1.0 - A16 * pow(TRR, -B16))*(1.0 - C16*pow(TRR, -E16));
        Y  = YA + (YB-YA)/(1.6-1.4)*(PRR-1.4);
    }
    else if ((PRR >= 1.6) && (PRR < 1.8))
    {
        YA = 1.02 * (1.0 - A16 * pow(TRR, -B16))*(1.0 - C16*pow(TRR, -E16));
        YB = 1.03 * (1.0 - A18 * pow(TRR, -B18))*(1.0 - C18*pow(TRR, -E18));
        Y  = YA + (YB-YA)/(1.8-1.6)*(PRR-1.6);
    }
    else if ((PRR >= 1.8) && (PRR < 2.0))
    {
        YA = 1.03 * (1.0 - A18 * pow(TRR, -B18))*(1.0 - C18*pow(TRR, -E18));
        YB = 1.03 * (1.0 - A20 * pow(TRR, -B20))*(1.0 - C20*pow(TRR, -E20));
        Y  = YA + (YB-YA)/(2.0-1.8)*(PRR-1.8);
    }
    else if ((PRR >= 2.0) && (PRR < 2.5))
    {
        YA = 1.03 * (1.0 - A20 * pow(TRR, -B20))*(1.0 - C20*pow(TRR, -E20));
        YB = 1.04 * (1.0 - A25 * pow(TRR, -B25))*(1.0 - C25*pow(TRR, -E25));
        Y  = YA + (YB-YA)/(2.5-2.0)*(PRR-2.0);
    }
    else if ((PRR >= 2.5) && (PRR < 3.0))
    {
        YA = 1.04 * (1.0 - A25 * pow(TRR, -B25))*(1.0 - C25*pow(TRR, -E25));
        YB = 1.05 * (1.0 - A30 * pow(TRR, -B30))*(1.0 - C30*pow(TRR, -E30));
        Y  = YA + (YB-YA)/(3.0-2.5)*(PRR-2.5);
    }
    else if ((PRR >= 3.0) && (PRR < 4.0))
    {
        YA = 1.05 * (1.0 - A30 * pow(TRR, -B30))*(1.0 - C30*pow(TRR, -E30));
        YB = 1.06 * (1.0 - A40 * pow(TRR, -B40))*(1.0 - C40*pow(TRR, -E40));
        Y  = YA + (YB-YA)/(4.0-3.0)*(PRR-3.0);
    }
    else if ((PRR >= 4.0) && (PRR < 5.0))
    {
        YA = 1.06 * (1.0 - A40 * pow(TRR, -B40))*(1.0 - C40*pow(TRR, -E40));
        YB = 1.07 * (1.0 - A50 * pow(TRR, -B50));
        Y  = YA + (YB-YA)/(5.0-4.0)*(PRR-4.0);
    }
    else if (PRR >= 5.0)
    {
        Y = 1.07 * (1.0 - A50 * pow(TRR, -B50));
    }

    if (Y == 0) { Y = 1e-16; }
    return Y;
}


//- Return mass Diffusivity of specie ith [m^2/s] based on Fick's law using
//  binary Diffusion coefficients based on Fuller and Takahashi models
template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::Dimix
(
    label speciei,
    const scalar p,
    const scalar T
) const
{
    scalar sum = 0.0;
    scalar Dimix = 0.0;

    // Pre-compute values independent of j
    const scalar T175 = T*sqrt(T*sqrt(T));  // T^1.75 = T * T^0.5 * T^0.25
    const scalar pAtm = p/101325.0;

    forAll(Xmd_, j)
    {
        if (speciei != j)
        {
            sum = sum + Xmd_[j]/
                        (
                         phi(p/(Pcmd_[j][speciei]), T/(Tcmd_[j][speciei]))
                         *0.001*T175*sqrt(Mmd_[j][speciei])
                         /(pAtm*sqr(sigmd_[j][speciei]))
                        );
        }
    }

    if (sum == 0.0) { Dimix = 0; }
    else { Dimix = (1-Ymd_[speciei])/(sum); }

    return Dimix/1e4;
}


template<class Thermo>
inline void Foam::chungTransport<Thermo>::updateTRANS
(
    const scalar& sigmaM,
    const scalar& epsilonkM,
    const scalar& MM,
    const scalar& VcM,
    const scalar& TcM,
    const scalar& omegaM,
    const scalar& miuiM,
    const scalar& kappaiM,
    const List<scalar>& Ymd,
    const List<scalar>& Xmd,
    const List<List<scalar>>& Tcmd,
    const List<List<scalar>>& Pcmd,
    const List<List<scalar>>& Mmd,
    const List<List<scalar>>& sigmd
)
{
    sigmaM_    = sigmaM;
    epsilonkM_ = epsilonkM;
    MM_        = MM;
    VcM_       = VcM;
    TcM_       = TcM;
    omegaM_    = omegaM;
    miuiM_     = miuiM;
    kappaiM_   = kappaiM;
    Ymd_       = Ymd;
    Xmd_       = Xmd;
    sumXmd_ = 0; forAll(Xmd_, _i) { sumXmd_ += Xmd_[_i]; }
    Tcmd_      = Tcmd;
    Pcmd_      = Pcmd;
    Mmd_       = Mmd;
    sigmd_     = sigmd;

    cacheT_ = -1; // composition changed -> invalidate (p,T) cache
}


template<class Thermo>
inline void Foam::chungTransport<Thermo>::updateTRANS_noDiffusion
(
    const scalar& sigmaM,
    const scalar& epsilonkM,
    const scalar& MM,
    const scalar& VcM,
    const scalar& TcM,
    const scalar& omegaM,
    const scalar& miuiM,
    const scalar& kappaiM,
    const List<scalar>& Ymd,
    const List<scalar>& Xmd
)
{
    sigmaM_    = sigmaM;
    epsilonkM_ = epsilonkM;
    MM_        = MM;
    VcM_       = VcM;
    TcM_       = TcM;
    omegaM_    = omegaM;
    miuiM_     = miuiM;
    kappaiM_   = kappaiM;
    Ymd_       = Ymd;
    Xmd_       = Xmd;
    sumXmd_ = 0; forAll(Xmd_, _i) { sumXmd_ += Xmd_[_i]; }
    // Tcmd_/Pcmd_/Mmd_/sigmd_ deliberately NOT updated -- only Dimix reads
    // them and the no-diffusion path never calls Dimix. Skips the O(n^2)
    // matrix copies; mu/kappa use only the scalars above + Xmd_.

    cacheT_ = -1; // composition changed -> invalidate (p,T) cache
}


// * * * * * * * * * * * * * * * Access Functions * * * * * * * * * * * * * * //

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::sigmaM() const
{
    return sigmaM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::epsilonkM() const
{
    return epsilonkM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::MM() const
{
    return MM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::VcM() const
{
    return VcM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::TcM() const
{
    return TcM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::omegaM() const
{
    return omegaM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::miuiM() const
{
    return miuiM_;
}

template<class Thermo>
inline Foam::scalar Foam::chungTransport<Thermo>::kappaiM() const
{
    return kappaiM_;
}

template<class Thermo>
inline Foam::List<Foam::scalar> Foam::chungTransport<Thermo>::Ymd() const
{
    return Ymd_;
}

template<class Thermo>
inline Foam::List<Foam::scalar> Foam::chungTransport<Thermo>::Xmd() const
{
    return Xmd_;
}

template<class Thermo>
inline Foam::List<Foam::List<Foam::scalar>>
Foam::chungTransport<Thermo>::Tcmd() const
{
    return Tcmd_;
}

template<class Thermo>
inline Foam::List<Foam::List<Foam::scalar>>
Foam::chungTransport<Thermo>::Pcmd() const
{
    return Pcmd_;
}

template<class Thermo>
inline Foam::List<Foam::List<Foam::scalar>>
Foam::chungTransport<Thermo>::Mmd() const
{
    return Mmd_;
}

template<class Thermo>
inline Foam::List<Foam::List<Foam::scalar>>
Foam::chungTransport<Thermo>::sigmd() const
{
    return sigmd_;
}


// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template<class Thermo>
inline void Foam::chungTransport<Thermo>::operator=
(
    const chungTransport<Thermo>& ch
)
{
    Thermo::operator=(ch);

    sigmaM_    = ch.sigmaM_;
    epsilonkM_ = ch.epsilonkM_;
    MM_        = ch.MM_;
    VcM_       = ch.VcM_;
    TcM_       = ch.TcM_;
    omegaM_    = ch.omegaM_;
    miuiM_     = ch.miuiM_;
    kappaiM_   = ch.kappaiM_;
    // Tier-3: Ymd_/Xmd_/Tcmd_/Pcmd_/Mmd_/sigmd_ deliberately NOT copied. The
    // per-cell mixing mixture_ = sum Y_i*thermo_i only needs the base Thermo;
    // these List members are authoritatively set afterwards by
    // updateTRANS[_noDiffusion] (Xmd_/Ymd_/scalars) and the matrices are read
    // only by Dimix (FPV never calls it). Skipping the ~106x per-cell deep List
    // copies removes the dominant malloc/free churn (profiled ~62% of runtime).

    cacheT_ = -1; // invalidate (p,T) cache
}


template<class Thermo>
inline void Foam::chungTransport<Thermo>::operator+=
(
    const chungTransport<Thermo>& ch
)
{
    Thermo::operator+=(ch);

    sigmaM_    = ch.sigmaM_;
    epsilonkM_ = ch.epsilonkM_;
    MM_        = ch.MM_;
    VcM_       = ch.VcM_;
    TcM_       = ch.TcM_;
    omegaM_    = ch.omegaM_;
    miuiM_     = ch.miuiM_;
    kappaiM_   = ch.kappaiM_;
    // Tier-3: see operator= -- the 6 List members (Ymd_/Xmd_/Tcmd_/Pcmd_/Mmd_/
    // sigmd_) are set after mixing by updateTRANS[_noDiffusion] (or read only by
    // the never-called Dimix), so the per-cell deep copies are pure dead work.

    cacheT_ = -1; // invalidate (p,T) cache
}


template<class Thermo>
inline void Foam::chungTransport<Thermo>::operator*=
(
    const scalar s
)
{
    Thermo::operator*=(s);
}


// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //

template<class Thermo>
inline Foam::chungTransport<Thermo> Foam::operator+
(
    const chungTransport<Thermo>& ch1,
    const chungTransport<Thermo>& ch2
)
{
    Thermo t
    (
        static_cast<const Thermo&>(ch1) + static_cast<const Thermo&>(ch2)
    );

    return chungTransport<Thermo>
    (
        t,
        ch1.sigmaM_,
        ch1.epsilonkM_,
        ch1.MM_,
        ch1.VcM_,
        ch1.TcM_,
        ch1.omegaM_,
        ch1.miuiM_,
        ch1.kappaiM_,
        // Tier-3: empty List members -- the mixing temporary's Ymd_/Xmd_/Tcmd_/
        // Pcmd_/Mmd_/sigmd_ are never read (set later by updateTRANS); passing
        // empty Lists avoids copying the per-species matrices/vectors per cell.
        List<scalar>(),
        List<scalar>(),
        List<List<scalar>>(),
        List<List<scalar>>(),
        List<List<scalar>>(),
        List<List<scalar>>()
    );
}


template<class Thermo>
inline Foam::chungTransport<Thermo> Foam::operator*
(
    const scalar s,
    const chungTransport<Thermo>& ch
)
{
    return chungTransport<Thermo>
    (
        s*static_cast<const Thermo&>(ch),
        ch.sigmaM_,
        ch.epsilonkM_,
        ch.MM_,
        ch.VcM_,
        ch.TcM_,
        ch.omegaM_,
        ch.miuiM_,
        ch.kappaiM_,
        // Tier-3: empty List members (set later by updateTRANS) -- avoids the
        // per-cell copy of the per-species matrices/vectors into the scaled
        // mixing temporary.
        List<scalar>(),
        List<scalar>(),
        List<List<scalar>>(),
        List<List<scalar>>(),
        List<List<scalar>>(),
        List<List<scalar>>()
    );
}


// ************************************************************************* //
