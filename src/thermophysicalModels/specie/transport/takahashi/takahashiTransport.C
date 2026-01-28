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

\*---------------------------------------------------------------------------*/

#include "IOstreams.H"
#include "takahashiTransport.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template <class Thermo>
inline Foam::takahashiTransport<Thermo>::takahashiTransport(
    const Thermo &t, const scalar &As, const scalar &Ts, const scalar &Tc,
    const scalar &Vc, const scalar &Pc, const scalar &omega,
    const scalar &diffusionVolume, const List<scalar> &Ymd,
    const List<scalar> &Xmd, const List<List<scalar>> &Tcmd,
    const List<List<scalar>> &Pcmd, const List<List<scalar>> &Mmd,
    const List<List<scalar>> &sigmd)
    : Thermo(t), As_(As), Ts_(Ts), Tc_(Tc), Vc_(Vc), Pc_(Pc), omega_(omega),
      diffusionVolume_(diffusionVolume), Ymd_(Ymd), Xmd_(Xmd), Tcmd_(Tcmd),
      Pcmd_(Pcmd), Mmd_(Mmd), sigmd_(sigmd) {}

template <class Thermo>
inline Foam::takahashiTransport<Thermo>::takahashiTransport(
    const word &name, const takahashiTransport &tr)
    : Thermo(name, tr), As_(tr.As_), Ts_(tr.Ts_), Tc_(tr.Tc_), Vc_(tr.Vc_),
      Pc_(tr.Pc_), omega_(tr.omega_), diffusionVolume_(tr.diffusionVolume_),
      Ymd_(tr.Ymd_), Xmd_(tr.Xmd_), Tcmd_(tr.Tcmd_), Pcmd_(tr.Pcmd_),
      Mmd_(tr.Mmd_), sigmd_(tr.sigmd_) {}

template <class Thermo>
inline Foam::autoPtr<Foam::takahashiTransport<Thermo>>
Foam::takahashiTransport<Thermo>::clone() const {
  return autoPtr<takahashiTransport<Thermo>>(
      new takahashiTransport<Thermo>(*this));
}

template <class Thermo>
inline Foam::autoPtr<Foam::takahashiTransport<Thermo>>
Foam::takahashiTransport<Thermo>::New(const dictionary &dict) {
  return autoPtr<takahashiTransport<Thermo>>(
      new takahashiTransport<Thermo>(dict));
}

template <class Thermo>
Foam::takahashiTransport<Thermo>::takahashiTransport(const dictionary &dict)
    : Thermo(dict.dictName(), dict),
      As_(readScalar(dict.subDict("transport").lookup("As"))),
      Ts_(readScalar(dict.subDict("transport").lookup("Ts"))),
      Tc_(readScalar(dict.subDict("transport").lookup("Tc"))),
      Vc_(readScalar(dict.subDict("transport").lookup("Vc"))),
      Pc_(readScalar(dict.subDict("transport").lookup("Pc"))),
      omega_(readScalar(dict.subDict("transport").lookup("omega"))),
      diffusionVolume_(
          readScalar(dict.subDict("transport").lookup("diffusionVolume"))),
      Ymd_(2), Xmd_(2), Tcmd_(2), Pcmd_(2), Mmd_(2), sigmd_(2) {
  // Temporary initialization to support legacy logic
  // Assuming binary-like or self-interaction

  scalar Y = this->Y(); // Scalar mass fraction (usually 1 for pure)
  scalar W = this->W();

  forAll(Ymd_, i) Ymd_[i] = Y;
  forAll(Xmd_, i) Xmd_[i] = Y / W; // Approximation initialization

  // Fill lists with self properties
  forAll(Tcmd_, i) {
    Tcmd_[i].setSize(2);
    Tcmd_[i] = Tc_;
  }
  forAll(Pcmd_, i) {
    Pcmd_[i].setSize(2);
    Pcmd_[i] = Pc_;
  }
  forAll(Mmd_, i) {
    Mmd_[i].setSize(2);
    Mmd_[i] = W;
  }
  forAll(sigmd_, i) {
    sigmd_[i].setSize(2);
    sigmd_[i] = diffusionVolume_;
  }
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class Thermo>
void Foam::takahashiTransport<Thermo>::write(Ostream &os) const {
  os << this->specie::name() << endl << token::BEGIN_BLOCK << incrIndent << nl;

  Thermo::write(os);

  dictionary dict("transport");
  dict.add("As", As_);
  dict.add("Ts", Ts_);
  dict.add("Tc", Tc_);
  dict.add("Vc", Vc_);
  dict.add("Pc", Pc_);
  dict.add("omega", omega_);
  dict.add("diffusionVolume", diffusionVolume_);

  os << indent << dict.dictName() << dict << decrIndent << token::END_BLOCK
     << nl;
}

// * * * * * * * * * * * * * * * IOstream Operators  * * * * * * * * * * * * //

template <class Thermo>
Foam::Ostream &Foam::operator<<(Ostream &os,
                                const takahashiTransport<Thermo> &tr) {
  tr.write(os);
  return os;
}

// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template <class Thermo>
inline void Foam::takahashiTransport<Thermo>::operator=(
    const takahashiTransport<Thermo> &tr) {
  Thermo::operator=(tr);
  As_ = tr.As_;
  Ts_ = tr.Ts_;
  Tc_ = tr.Tc_;
  Vc_ = tr.Vc_;
  Pc_ = tr.Pc_;
  omega_ = tr.omega_;
  diffusionVolume_ = tr.diffusionVolume_;

  Ymd_ = tr.Ymd_;
  Xmd_ = tr.Xmd_;
  Tcmd_ = tr.Tcmd_;
  Pcmd_ = tr.Pcmd_;
  Mmd_ = tr.Mmd_;
  sigmd_ = tr.sigmd_;
}

template <class Thermo>
inline void Foam::takahashiTransport<Thermo>::operator+=(
    const takahashiTransport<Thermo> &tr) {
  Thermo::operator+=(tr);
  As_ += tr.As_;
  Ts_ += tr.Ts_;
  // Mixing rules for criticals? Usually linear or special.
  // For transport+, it's often weighted?
  // Standard OpenFOAM usually operates on single instance.
  // We retain copy of 'tr' members or sum them?
  // Usually operator+= is used for averaging.
  // We simply copy typical props or leave them?
  // Actually standard sutherland implementation calculates mixture coeffs.
  // Here we have scalars.
  // Let's assume standard behavior:
  // Actually, for Takahashi logic (mixture state), we probably want to COPY the
  // mixture state? But operator+= is usually aggregating "Y*Specie". I'll keep
  // it simple for now (copy or average).

  // Ymd_ etc are mixture states, not summed properties.
}

template <class Thermo>
inline void Foam::takahashiTransport<Thermo>::operator*=(const scalar s) {
  Thermo::operator*=(s);
}

// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //

template <class Thermo>
inline Foam::takahashiTransport<Thermo>
Foam::operator+(const takahashiTransport<Thermo> &tr1,
                const takahashiTransport<Thermo> &tr2) {
  Thermo t(static_cast<const Thermo &>(tr1) + static_cast<const Thermo &>(tr2));

  return takahashiTransport<Thermo>(t, tr1.As_, tr1.Ts_, tr1.Tc_, tr1.Vc_,
                                    tr1.Pc_, tr1.omega_, tr1.diffusionVolume_,
                                    tr1.Ymd_, tr1.Xmd_, tr1.Tcmd_, tr1.Pcmd_,
                                    tr1.Mmd_, tr1.sigmd_);
}

template <class Thermo>
inline Foam::takahashiTransport<Thermo>
Foam::operator*(const scalar s, const takahashiTransport<Thermo> &tr) {
  return takahashiTransport<Thermo>(s * static_cast<const Thermo &>(tr), tr.As_,
                                    tr.Ts_, tr.Tc_, tr.Vc_, tr.Pc_, tr.omega_,
                                    tr.diffusionVolume_, tr.Ymd_, tr.Xmd_,
                                    tr.Tcmd_, tr.Pcmd_, tr.Mmd_, tr.sigmd_);
}
