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

#include "specie.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class Thermo>
inline void Foam::takahashiTransport<Thermo>::updateTRANS(
    const List<scalar> &Ymd, const List<scalar> &Xmd,
    const List<List<scalar>> &Tcmd, const List<List<scalar>> &Pcmd,
    const List<List<scalar>> &Mmd, const List<List<scalar>> &sigmd) {
  Ymd_ = Ymd;
  Xmd_ = Xmd;
  Tcmd_ = Tcmd;
  Pcmd_ = Pcmd;
  Mmd_ = Mmd;
  sigmd_ = sigmd;
}

// Sutherland Viscosity
template <class Thermo>
inline Foam::scalar Foam::takahashiTransport<Thermo>::mu(const scalar p,
                                                         const scalar T) const {
  return As_ * sqrt(T) / (1.0 + Ts_ / T);
}

// Modified Eucken Thermal Conductivity
template <class Thermo>
inline Foam::scalar
Foam::takahashiTransport<Thermo>::kappa(const scalar p, const scalar T) const {
  scalar Cv = this->Cv(p, T); // J/kgK
  scalar muVal = mu(p, T);
  scalar R = this->R(); // J/kgK

  // Modified Eucken
  // k = mu * Cv * (1.32 + 1.77 * R/Cv)
  return muVal * Cv * (1.32 + 1.77 * R / (Cv + SMALL));
}

template <class Thermo>
inline Foam::scalar
Foam::takahashiTransport<Thermo>::alphah(const scalar p, const scalar T) const {
  return kappa(p, T) / this->Cp(p, T);
}

// Takahashi Diffusivity
template <class Thermo>
inline Foam::scalar
Foam::takahashiTransport<Thermo>::Dimix(label speciei, const scalar p,
                                        const scalar T) const {
  scalar sum = 0.0;
  scalar Dimix = 0.0;
  forAll(Xmd_, j) {
    if (speciei != j) {
      // Check indices bound
      if (j < Pcmd_.size() && speciei < Pcmd_[j].size()) {
        scalar Pcrit_ij = Pcmd_[j][speciei];
        scalar Tcrit_ij = Tcmd_[j][speciei];
        scalar M_ij = Mmd_[j][speciei];
        scalar Sigma_ij = sigmd_[j][speciei];

        // Fuller et al. with Takahashi correction (phi)
        sum = sum +
              Xmd_[j] / (phi(p / (Pcrit_ij + SMALL), T / (Tcrit_ij + SMALL)) *
                         0.001 * pow(T, 1.75) * pow(M_ij, 0.5) /
                         ((p / 101325.0) * pow(Sigma_ij, 2) + SMALL));
      }
    }
  }

  if (sum < SMALL) {
    Dimix = 0.0;
  } else {
    Dimix = (1.0 - Ymd_[speciei]) / sum;
  }

  return Dimix / 1e4; // Convert cm2/s to m2/s? Legacy code used 1e4 division.
                      // Fuller equation result units:
                      // If P in atm (p/101325), T in K, Sigma in Angstrom?
                      // Fuller gives D in cm^2/s usually.
                      // 1 cm^2/s = 1e-4 m^2/s.
                      // So division by 1e4 is correct.
}

// Takahashi phi coefficient
template <class Thermo>
inline Foam::scalar Foam::takahashiTransport<Thermo>::phi(scalar PRR,
                                                          scalar TRR) const {
  const scalar A01 = 0.038042, B01 = 1.52267;
  const scalar A02 = 0.067433, B02 = 2.16794;
  const scalar A03 = 0.098317, B03 = 2.42910;
  const scalar A04 = 0.137610, B04 = 2.77605;
  const scalar A05 = 0.175081, B05 = 2.98256;
  const scalar A06 = 0.216376, B06 = 3.11384;
  const scalar A08 = 0.314051, B08 = 3.50264;
  //    const scalar C08 = 0.000000, E08 = 0.00000; //unused
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

  scalar Y = 0, YA = 0, YB = 0;
  TRR = (TRR < 1.0) ? 1.0 : TRR;
  TRR = (TRR > 5.0) ? 5.0 : TRR;
  PRR = (PRR > 5.0) ? 5.0 : PRR;

  if (PRR < 0.1) {
    Y = 1.01 * (1.0 - A01 * pow(TRR, -B01));
    Y = 1.0 + (Y - 1.0) / (0.1 - 0.0) * (PRR - 0.0);
  } else if ((PRR >= 0.1) && (PRR < 0.2)) {
    YA = 1.01 * (1.0 - A01 * pow(TRR, -B01));
    YB = 1.01 * (1.0 - A02 * pow(TRR, -B02));
    Y = YA + (YB - YA) / (0.2 - 0.1) * (PRR - 0.1);
  } else if ((PRR >= 0.2) && (PRR < 0.3)) {
    YA = 1.01 * (1.0 - A02 * pow(TRR, -B02));
    YB = 1.01 * (1.0 - A03 * pow(TRR, -B03));
    Y = YA + (YB - YA) / (0.3 - 0.2) * (PRR - 0.2);
  } else if ((PRR >= 0.3) && (PRR < 0.4)) {
    YA = 1.01 * (1.0 - A03 * pow(TRR, -B03));
    YB = 1.01 * (1.0 - A04 * pow(TRR, -B04));
    Y = YA + (YB - YA) / (0.4 - 0.3) * (PRR - 0.3);
  } else if ((PRR >= 0.4) && (PRR < 0.5)) {
    YA = 1.01 * (1.0 - A04 * pow(TRR, -B04));
    YB = 1.01 * (1.0 - A05 * pow(TRR, -B05));
    Y = YA + (YB - YA) / (0.5 - 0.4) * (PRR - 0.4);
  } else if ((PRR >= 0.5) && (PRR < 0.6)) {
    YA = 1.01 * (1.0 - A05 * pow(TRR, -B05));
    YB = 1.01 * (1.0 - A06 * pow(TRR, -B06));
    Y = YA + (YB - YA) / (0.6 - 0.5) * (PRR - 0.5);
  } else if ((PRR >= 0.6) && (PRR < 0.8)) {
    YA = 1.01 * (1.0 - A06 * pow(TRR, -B06));
    YB = 1.01 * (1.0 - A08 * pow(TRR, -B08));
    Y = YA + (YB - YA) / (0.8 - 0.6) * (PRR - 0.6);
  } else if ((PRR >= 0.8) && (PRR < 1.0)) {
    YA = 1.01 * (1.0 - A08 * pow(TRR, -B08));
    YB = 1.02 * (1.0 - A10 * pow(TRR, -B10)) * (1.0 - C10 * pow(TRR, -E10));
    Y = YA + (YB - YA) / (1.0 - 0.8) * (PRR - 0.8);
  } else if ((PRR >= 1.0) && (PRR < 1.2)) {
    YA = 1.02 * (1.0 - A10 * pow(TRR, -B10)) * (1.0 - C10 * pow(TRR, -E10));
    YB = 1.02 * (1.0 - A12 * pow(TRR, -B12)) * (1.0 - C12 * pow(TRR, -E12));
    Y = YA + (YB - YA) / (1.2 - 1.0) * (PRR - 1.0);
  } else if ((PRR >= 1.2) && (PRR < 1.4)) {
    YA = 1.02 * (1.0 - A12 * pow(TRR, -B12)) * (1.0 - C12 * pow(TRR, -E12));
    YB = 1.02 * (1.0 - A14 * pow(TRR, -B14)) * (1.0 - C14 * pow(TRR, -E14));
    Y = YA + (YB - YA) / (1.4 - 1.2) * (PRR - 1.2);
  } else if ((PRR >= 1.4) && (PRR < 1.6)) {
    YA = 1.02 * (1.0 - A14 * pow(TRR, -B14)) * (1.0 - C14 * pow(TRR, -E14));
    YB = 1.02 * (1.0 - A16 * pow(TRR, -B16)) * (1.0 - C16 * pow(TRR, -E16));
    Y = YA + (YB - YA) / (1.6 - 1.4) * (PRR - 1.4);
  } else if ((PRR >= 1.6) && (PRR < 1.8)) {
    YA = 1.02 * (1.0 - A16 * pow(TRR, -B16)) * (1.0 - C16 * pow(TRR, -E16));
    YB = 1.03 * (1.0 - A18 * pow(TRR, -B18)) * (1.0 - C18 * pow(TRR, -E18));
    Y = YA + (YB - YA) / (1.8 - 1.6) * (PRR - 1.6);
  } else if ((PRR >= 1.8) && (PRR < 2.0)) {
    YA = 1.03 * (1.0 - A18 * pow(TRR, -B18)) * (1.0 - C18 * pow(TRR, -E18));
    YB = 1.03 * (1.0 - A20 * pow(TRR, -B20)) * (1.0 - C20 * pow(TRR, -E20));
    Y = YA + (YB - YA) / (2.0 - 1.8) * (PRR - 1.8);
  } else if ((PRR >= 2.0) && (PRR < 2.5)) {
    YA = 1.03 * (1.0 - A20 * pow(TRR, -B20)) * (1.0 - C20 * pow(TRR, -E20));
    YB = 1.04 * (1.0 - A25 * pow(TRR, -B25)) * (1.0 - C25 * pow(TRR, -E25));
    Y = YA + (YB - YA) / (2.5 - 2.0) * (PRR - 2.0);
  } else if ((PRR >= 2.5) && (PRR < 3.0)) {
    YA = 1.04 * (1.0 - A25 * pow(TRR, -B25)) * (1.0 - C25 * pow(TRR, -E25));
    YB = 1.05 * (1.0 - A30 * pow(TRR, -B30)) * (1.0 - C30 * pow(TRR, -E30));
    Y = YA + (YB - YA) / (3.0 - 2.5) * (PRR - 2.5);
  } else if ((PRR >= 3.0) && (PRR < 4.0)) {
    YA = 1.05 * (1.0 - A30 * pow(TRR, -B30)) * (1.0 - C30 * pow(TRR, -E30));
    YB = 1.06 * (1.0 - A40 * pow(TRR, -B40)) * (1.0 - C40 * pow(TRR, -E40));
    Y = YA + (YB - YA) / (4.0 - 3.0) * (PRR - 3.0);
  } else if ((PRR >= 4.0) && (PRR < 5.0)) {
    YA = 1.06 * (1.0 - A40 * pow(TRR, -B40)) * (1.0 - C40 * pow(TRR, -E40));
    YB = 1.07 * (1.0 - A50 * pow(TRR, -B50));
    Y = YA + (YB - YA) / (5.0 - 4.0) * (PRR - 4.0);
  } else if (PRR >= 5.0) {
    Y = 1.07 * (1.0 - A50 * pow(TRR, -B50));
  }

  if (Y == 0) {
    Y = 1e-16;
  }
  return Y;
}
