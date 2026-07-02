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

Class
    Foam::SRKGas

Description
    Soave-Redlich-Kwong cubic equation of state for gases.

    Validated against NIST data.

Usage
    \table
        Property     | Description
        Tc           | Critical temperature [K]
        Pc           | Critical pressure [Pa]
        omega        | Acentric factor [-]
    \endtable

    The EOS parameters (b, coef1, coef2, coef3) are computed from Tc, Pc
    and omega in the dictionary constructor.

SourceFiles
    SRKGasI.H
    SRKGas.C

\*---------------------------------------------------------------------------*/

#ifndef SRKGas_H
#define SRKGas_H

#include "autoPtr.H"
#include "rfSpecie.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators

template<class Specie> class SRKGas;

template<class Specie>
inline SRKGas<Specie> operator+
(
    const SRKGas<Specie>&,
    const SRKGas<Specie>&
);

template<class Specie>
inline SRKGas<Specie> operator*
(
    const scalar,
    const SRKGas<Specie>&
);

template<class Specie>
inline SRKGas<Specie> operator==
(
    const SRKGas<Specie>&,
    const SRKGas<Specie>&
);

template<class Specie>
Ostream& operator<<
(
    Ostream&,
    const SRKGas<Specie>&
);



/*---------------------------------------------------------------------------*\
                           Class SRKGas Declaration
\*---------------------------------------------------------------------------*/

template<class Specie>
class SRKGas
:
    public Specie
{
    // Private Data

        //- SRK co-volume parameter [m^3/kmol]
        scalar b_;

        //- Decomposed aAlpha coefficients
        scalar coef1_;
        scalar coef2_;
        scalar coef3_;

        //- Peneloux volume translation [m^3/kmol].
        //  v_eff = v_SRK - c(T).  c_ is the constant / gas-baseline value.
        //  Default 0 preserves the original SRK behaviour.
        //  Reference: Peneloux, Rauzy & Freze, Fluid Phase Equilib. 8 (1982) 7-23.
        scalar c_;

        //- Optional temperature-dependent Peneloux translation c(T) [m^3/kmol].
        //  For strongly non-ideal fluids (e.g. H2O) a constant shift only
        //  matches one T while the SRK liquid-density error grows with T.
        //  Liquid branch c(T) = cq0_ + cq1_ T + cq2_ T^2 for T <= cTlo_;
        //  smoothstep ramp to the gas baseline c_ over [cTlo_, cThi_];
        //  c_ for T >= cThi_. Active only when cTlo_ > 0.
        scalar cq0_;
        scalar cq1_;
        scalar cq2_;
        scalar cTlo_;
        scalar cThi_;

        //- Tier-1 memoisation of the cubic-root solve: (p,T) -> Z, shared by
        //  rho/psi/Cp/mu/kappa within one cell (each calls Z(p,T) at the same
        //  state). Invalidated on updateEoS (composition change); sentinel
        //  pZcache_ < 0 forces a first-call miss. mutable: Z() is const.
        mutable scalar pZcache_ = -1;
        mutable scalar TZcache_ = -1;
        mutable scalar Zcache_ = 0;

        //- Cubic root-selection policy (global EoS option). false (default):
        //  always take the largest root (vapour branch) -- continuous in T,
        //  keeps the h->T inversion robust, but overestimates compressed-liquid
        //  specific volume in the 3-root (subcritical) regime. true: select the
        //  thermodynamically stable phase by minimum fugacity (liquid root for
        //  p > psat, vapour for p < psat), recovering the correct compressed-
        //  liquid density at the cost of a Z (hence h) discontinuity across the
        //  saturation line. Set by 'stableRoot on;' in any species' rfProperties
        //  (static, so a single entry switches the whole mixture). Intended for
        //  the fgmFluid path, where T is read from the manifold table so the
        //  saturation-line h discontinuity does not feed an h->T Newton solve.
        static bool stableRoot_;


    // Private Member Functions

        //- Return Xterm in dHdT equation
        inline scalar Xterm(const scalar p, const scalar T) const;

        //- Return derivative of Z w.r.t. T
        inline scalar dZdT(const scalar p, const scalar T) const;

        //- Return derivative of Z w.r.t. T with pre-computed Z value
        inline scalar dZdT
        (
            const scalar p,
            const scalar T,
            const scalar Zv
        ) const;


public:

    // Constructors

        //- Construct from components (Peneloux shift defaults to 0)
        inline SRKGas
        (
            const Specie& sp,
            const scalar& b,
            const scalar& coef1,
            const scalar& coef2,
            const scalar& coef3,
            const scalar& c = 0
        );

        //- Construct from name and dictionary
        SRKGas(const word& name, const dictionary& dict);

        //- Construct as named copy
        inline SRKGas(const word& name, const SRKGas&);

        //- Construct and return a clone
        inline autoPtr<SRKGas> clone() const;


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "SRKGas<" + word(Specie::typeName_()) + '>';
        }

        // Fundamental properties

            //- Is the equation of state is incompressible i.e. rho != f(p)
            static const bool incompressible = false;

            //- Is the equation of state is isochoric i.e. rho = const
            static const bool isochoric = false;

            //- Return density [kg/m^3]
            inline scalar rho(scalar p, scalar T) const;

            //- Return enthalpy contribution [J/kg]
            inline scalar h(const scalar p, const scalar T) const;

            //- Return Cp contribution [J/(kg K]
            inline scalar Cp(scalar p, scalar T) const;

            //- Return internal energy contribution [J/kg]
            inline scalar e(const scalar p, const scalar T) const;

            //- Return Cv contribution [J/(kg K]
            inline scalar Cv(scalar p, scalar T) const;

            //- Return entropy contribution to the integral of Cp/T [J/kg/K]
            inline scalar sp(const scalar p, const scalar T) const;

            //- Return entropy contribution to the integral of Cv/T [J/kg/K]
            inline scalar sv(const scalar p, const scalar T) const;

            //- Return compressibility [s^2/m^2]
            inline scalar psi(scalar p, scalar T) const;

            //- Return compression factor []
            inline scalar Z(scalar p, scalar T) const;

            //- Return (Cp - Cv) [J/(kg K]
            inline scalar CpMCv(scalar p, scalar T) const;

            //- Return volumetric coefficient of thermal expansion [1/T]
            inline scalar alphav(const scalar p, const scalar T) const;

            //- Return speed of sound squared [m^2/s^2].
            //  Derived from Oefelein (AIAA 2019-0634) Eqs. (88)-(91),
            //  c^2 = gamma R T / (Zp/Z) with Zp, Z_T evaluated from the SRK
            //  compressibility factor and its partial derivatives.
            inline scalar c2(const scalar p, const scalar T) const;

            //- Return pure-species natural logarithm of fugacity coefficient.
            //  SRK closed-form of Oefelein Eq. (99) specialised to a pure fluid
            //  (b_i = b_m, delta_i = 1) giving the classical
            //  ln(phi) = (Z-1) - ln(Z-B) - (A/B) ln((Z+B)/Z).
            //  Reference: Soave, Chem. Eng. Sci. 27 (1972) 1197-1203.
            inline scalar lnPhi(const scalar p, const scalar T) const;

            //- Access the Peneloux volume shift (baseline) [m^3/kmol]
            inline scalar c() const { return c_; }

            //- Access the temperature-dependent Peneloux c(T) coefficients
            inline scalar cq0()  const { return cq0_; }
            inline scalar cq1()  const { return cq1_; }
            inline scalar cq2()  const { return cq2_; }
            inline scalar cTlo() const { return cTlo_; }
            inline scalar cThi() const { return cThi_; }

            //- Peneloux translation c(T) [m^3/kmol] (constant if cTlo_ <= 0)
            inline scalar cAt(const scalar T) const;

            //- Update coefficients for mixture
            void updateEoS
            (
                const scalar& b,
                const scalar& coef1,
                const scalar& coef2,
                const scalar& coef3,
                const scalar& c = 0,
                const scalar& cq0 = 0,
                const scalar& cq1 = 0,
                const scalar& cq2 = 0,
                const scalar& cTlo = 0,
                const scalar& cThi = 0
            )
            {
                b_ = b;
                coef1_ = coef1;
                coef2_ = coef2;
                coef3_ = coef3;
                c_ = c;
                cq0_ = cq0;
                cq1_ = cq1;
                cq2_ = cq2;
                cTlo_ = cTlo;
                cThi_ = cThi;
                pZcache_ = -1;  // composition changed -> invalidate Z(p,T) cache
            }


    // IO

            //- Write to Ostream
            void write(Ostream& os) const;

    // Member Operators

        inline void operator+=(const SRKGas&);
        inline void operator*=(const scalar);


    // Friend operators

        friend SRKGas operator+ <Specie>
        (
            const SRKGas&,
            const SRKGas&
        );

        friend SRKGas operator* <Specie>
        (
            const scalar s,
            const SRKGas&
        );

        friend SRKGas operator== <Specie>
        (
            const SRKGas&,
            const SRKGas&
        );


    // Ostream Operator

        friend Ostream& operator<< <Specie>
        (
            Ostream&,
            const SRKGas&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "SRKGasI.H"

#ifdef NoRepository
    #include "SRKGas.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
