/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
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
    Foam::takahashiTransport

Description
    Transport package computing:
    - Viscosity/Conductivity: Sutherland model
    - Diffusivity: Takahashi High-Pressure model + Fuller

SourceFiles
    takahashiTransportI.H
    takahashiTransport.C

\*---------------------------------------------------------------------------*/

#ifndef takahashiTransport_H
#define takahashiTransport_H

#include "PtrList.H"
#include "primitiveFields.H"
#include "specie.H"

namespace Foam {

template <class Thermo> class takahashiTransport;

template <class Thermo>
inline takahashiTransport<Thermo> operator+(const takahashiTransport<Thermo> &,
                                            const takahashiTransport<Thermo> &);

template <class Thermo>
inline takahashiTransport<Thermo> operator*(const scalar,
                                            const takahashiTransport<Thermo> &);

template <class Thermo>
Ostream &operator<<(Ostream &, const takahashiTransport<Thermo> &);

template <class Thermo> class takahashiTransport : public Thermo {
  // Sutherland coefficients
  scalar As_;
  scalar Ts_;

  // Takahashi coefficients
  scalar Tc_;
  scalar Vc_;
  scalar Pc_;
  scalar omega_;
  scalar diffusionVolume_; // Sigma_v

  //- Members for diffusivity evaluation (Mixture state)
  List<scalar> Ymd_;
  List<scalar> Xmd_;
  List<List<scalar>> Tcmd_;
  List<List<scalar>> Pcmd_;
  List<List<scalar>> Mmd_;
  List<List<scalar>> sigmd_;

public:
  // Constructors

  //- Construct from components
  inline takahashiTransport(const Thermo &t, const scalar &As, const scalar &Ts,
                            const scalar &Tc, const scalar &Vc,
                            const scalar &Pc, const scalar &omega,
                            const scalar &diffusionVolume,
                            const List<scalar> &Ymd, const List<scalar> &Xmd,
                            const List<List<scalar>> &Tcmd,
                            const List<List<scalar>> &Pcmd,
                            const List<List<scalar>> &Mmd,
                            const List<List<scalar>> &sigmd);

  //- Construct as named copy
  inline takahashiTransport(const word &, const takahashiTransport &);

  //- Construct from dictionary
  takahashiTransport(const dictionary &dict);

  //- Construct and return a clone
  inline autoPtr<takahashiTransport> clone() const;

  // Selector from dictionary
  inline static autoPtr<takahashiTransport> New(const dictionary &dict);

  // Member functions

  //- Return the instantiated type name
  static word typeName() { return "takahashi<" + Thermo::typeName() + '>'; }

  //- Return Dynamic viscosity [kg/ms]
  inline scalar mu(const scalar p, const scalar T) const;

  //- Return Thermal conductivity [W/mK]
  inline scalar kappa(const scalar p, const scalar T) const;

  //- Return Thermal diffusivity of enthalpy [kg/ms]
  inline scalar alphah(const scalar p, const scalar T) const;

  //- Return coefficient Phi based on Takahashi model
  inline scalar phi(scalar PRR, scalar TRR) const;

  //- Return mass Diffusivity of specie ith [m^2/s]
  inline scalar Dimix(label speciei, const scalar p, const scalar T) const;

  //- Update coefficients for mixture transport properties
  inline void updateTRANS(const List<scalar> &Ymd, const List<scalar> &Xmd,
                          const List<List<scalar>> &Tcmd,
                          const List<List<scalar>> &Pcmd,
                          const List<List<scalar>> &Mmd,
                          const List<List<scalar>> &sigmd);

  // Accessors for testing
  inline scalar As() const { return As_; }
  inline scalar Ts() const { return Ts_; }
  inline scalar Tc() const { return Tc_; }
  inline scalar Vc() const { return Vc_; }
  inline scalar Pc() const { return Pc_; }
  inline scalar omega() const { return omega_; }
  inline scalar diffusionVolume() const { return diffusionVolume_; }

  // IO

  //- Write to Ostream
  void write(Ostream &os) const;

  // Member operators

  inline void operator=(const takahashiTransport &);
  inline void operator+=(const takahashiTransport &);
  inline void operator*=(const scalar);

  // Friend operators

  friend takahashiTransport operator+
      <Thermo>(const takahashiTransport &, const takahashiTransport &);

  friend takahashiTransport operator*
      <Thermo>(const scalar, const takahashiTransport &);

  friend Ostream &operator<< <Thermo>(Ostream &, const takahashiTransport &);
};

} // End namespace Foam

#include "takahashiTransportI.H"

#ifdef NoRepository
#include "takahashiTransport.C"
#endif

#endif
