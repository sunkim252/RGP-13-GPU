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
    Foam::chungTransport

Description
    Real Gas Transport Package Using:
        Chung et al. (1988) model for Viscosity and Thermal Conductivity,
        Fuller (1987) and Takahashi et al. (1975) model for binary diffusion
        coefficients. Mass Diffusivity is based on mixture averaged model.

    Output:
        Dynamic Viscosity [kg/m.s]
        Thermal Conductivity [W/m.K]
        Mass Diffusivity [m2/s]

    Ported from realFluidFoam-8 chungTakaTransport.

SourceFiles
    chungTransportI.H
    chungTransport.C

\*---------------------------------------------------------------------------*/

#ifndef chungTransport_H
#define chungTransport_H

#include "PtrList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators

template<class Thermo> class chungTransport;

template<class Thermo>
inline chungTransport<Thermo> operator+
(
    const chungTransport<Thermo>&,
    const chungTransport<Thermo>&
);

template<class Thermo>
inline chungTransport<Thermo> operator*
(
    const scalar,
    const chungTransport<Thermo>&
);

template<class Thermo>
Ostream& operator<<
(
    Ostream&,
    const chungTransport<Thermo>&
);


/*---------------------------------------------------------------------------*\
                       Class chungTransport Declaration
\*---------------------------------------------------------------------------*/

template<class Thermo>
class chungTransport
:
    public Thermo
{

protected:

    // Protected data

        //- Members for Viscosity and Thermal Conductivity evaluation

            //- sigma in eq. (4) - Chung's Model
            scalar sigmaM_;

            //- epsilon/k in eq. (5) - Chung's Model
            scalar epsilonkM_;

            //- Molecular weight
            scalar MM_;

            //- Critical volume
            scalar VcM_;

            //- Critical temperature
            scalar TcM_;

            //- Acentric factor
            scalar omegaM_;

            //- Dipole moment parameter
            scalar miuiM_;

            //- Association factor
            scalar kappaiM_;

        //- Members for diffusivity evaluation

            //- List of mass fractions
            List<scalar> Ymd_;

            //- List of mole fractions
            List<scalar> Xmd_;

            //- List of Tcmd[i][j]
            List<List<scalar>> Tcmd_;

            //- List of Pcmd[i][j]
            List<List<scalar>> Pcmd_;

            //- List of Mmd[i][j]
            List<List<scalar>> Mmd_;

            //- List of sigmd[i][j]
            List<List<scalar>> sigmd_;

        //- Single-entry (p,T) cache of calculate()'s result so the consecutive
        //  mu(p,T)/kappa(p,T) calls on one cell reuse one rho-cubic + Omega*
        //  evaluation. Invalidated (cacheT_ < 0) on composition change.
        //  mutable: valid for OpenFOAM's serial per-process cell loop.
            mutable scalar cacheP_;
            mutable scalar cacheT_;
            mutable scalar cacheTstar_;
            mutable scalar cacheMur_;
            mutable scalar cacheEta0_;
            mutable scalar cacheY_;
            mutable scalar cacheG1_;


    // Protected Member Functions

        //- Calculate intermediate quantities for Chung model
        inline void calculate
        (
            scalar p,
            scalar T,
            scalar& Tstar,
            scalar& mur,
            scalar& eta0,
            scalar& Y,
            scalar& G1
        ) const;


public:

    // Constructors

        //- Construct from components
        inline chungTransport
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
        );

        //- Construct as named copy
        inline chungTransport(const word&, const chungTransport&);

        //- Construct from name and dictionary
        chungTransport(const word& name, const dictionary& dict);

        //- Construct from base thermo and dictionary
        chungTransport(const Thermo& t, const dictionary& dict);

        //- Construct and return a clone
        inline autoPtr<chungTransport> clone() const;


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "chung<" + Thermo::typeName() + '>';
        }

        //- Dynamic viscosity [kg/m/s]
        inline scalar mu(const scalar p, const scalar T) const;

        //- Thermal conductivity [W/m/K]
        inline scalar kappa(const scalar p, const scalar T) const;

        //- Takahashi correction coefficient for binary diffusion
        inline scalar phi(scalar PRR, scalar TRR) const;

        //- Mass diffusivity of specie ith [m^2/s]
        inline scalar Dimix
        (
            label speciei,
            const scalar p,
            const scalar T
        ) const;

        //- Update coefficients for mixture transport properties
        inline void updateTRANS
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
        );

        //- Tier-0: lightweight update for the no-species-diffusion path.
        //  Sets the scalar Chung params + Ymd_/Xmd_ (needed by mu/kappa) but
        //  SKIPS the O(n^2) Tcmd_/Pcmd_/Mmd_/sigmd_ copies (only Dimix uses
        //  them). mu/kappa stay bit-identical to the full updateTRANS path.
        inline void updateTRANS_noDiffusion
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
        );

        //- Access functions for member data
        inline scalar sigmaM() const;
        inline scalar epsilonkM() const;
        inline scalar MM() const;
        inline scalar VcM() const;
        inline scalar TcM() const;
        inline scalar omegaM() const;
        inline scalar miuiM() const;
        inline scalar kappaiM() const;
        inline List<scalar> Ymd() const;
        inline List<scalar> Xmd() const;
        inline List<List<scalar>> Tcmd() const;
        inline List<List<scalar>> Pcmd() const;
        inline List<List<scalar>> Mmd() const;
        inline List<List<scalar>> sigmd() const;

        //- Write to Ostream
        void write(Ostream& os) const;


    // Member Operators

        inline void operator=(const chungTransport&);
        inline void operator+=(const chungTransport&);
        inline void operator*=(const scalar);


    // Friend operators

        friend chungTransport operator+ <Thermo>
        (
            const chungTransport&,
            const chungTransport&
        );

        friend chungTransport operator* <Thermo>
        (
            const scalar,
            const chungTransport&
        );


    // Ostream Operator

        friend Ostream& operator<< <Thermo>
        (
            Ostream&,
            const chungTransport&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "chungTransportI.H"

#ifdef NoRepository
    #include "chungTransport.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
