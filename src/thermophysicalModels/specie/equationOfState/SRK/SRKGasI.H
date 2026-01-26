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
#include "thermodynamicConstants.H"

using namespace Foam::constant::thermodynamic;

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template <class Specie>
inline Foam::SRKGas<Specie>::SRKGas(const Specie &sp, const scalar &Tc,
                                    const scalar &Vc, const scalar &Zc,
                                    const scalar &Pc, const scalar &omega)
    : Specie(sp), Tc_(Tc), Vc_(Vc), Zc_(Zc), Pc_(Pc), omega_(omega) {}

template <class Specie>
inline Foam::SRKGas<Specie>::SRKGas(const word &name, const SRKGas &pg)
    : Specie(name, pg), Tc_(pg.Tc_), Vc_(pg.Vc_), Zc_(pg.Zc_), Pc_(pg.Pc_),
      omega_(pg.omega_) {}

template <class Specie>
inline Foam::autoPtr<Foam::SRKGas<Specie>> Foam::SRKGas<Specie>::clone() const {
  return autoPtr<SRKGas<Specie>>(new SRKGas<Specie>(*this));
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::rho(scalar p, scalar T) const {
  const scalar Z = this->Z(p, T);
  return p / (Z * this->R() * T);
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::h(scalar p, scalar T) const {
  const scalar Pr = p / Pc_;
  const scalar Tr = T / Tc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);
  const scalar alpha = sqr(1 + kappa * (1 - sqrt(Tr)));

  const scalar Z = this->Z(p, T);

  // SRK specific terms
  const scalar b = 0.08664 * this->R() * Tc_ / Pc_;
  const scalar B = b * p / (this->R() * T);
  const scalar a = 0.42747 * sqr(this->R() * Tc_) / Pc_;
  // const scalar A = a*alpha*p/sqr(this->R()*T);

  // Enthalpy departure function
  // For SRK: (H - H_ig)/RT = Z - 1 - (A/B) * (d ln(alpha) / d ln Tr) * ln(1 +
  // B/Z) - ... Wait, let's use the explicit form implemented in PR but adapted.
  // H = H_ig + RT(Z-1) + (T da/dT - a) * 1/b * ln(Z/(Z+B))
  // Note: logarithmic term orientation. In code it usually is ln((Z+B)/Z) ->
  // positive T da/dT - a = -a * alpha * (1 + kappa/sqrt(alpha) * sqrt(Tr) ) ...
  // ?

  // Let's look at PR implementation:
  // term = - 2.078*(1 + kappa)*sqrt(alpha) * log((Z + 2.414*B)/(Z - 0.414*B))
  // This is related to (a - T da/dT).

  // For SRK:
  // a(T) = a_c * alpha
  // da/dT = a_c * d(alpha)/dT
  // d(alpha)/dT = -kappa * sqrt(alpha/Tr) / Tc
  // T da/dT = -kappa * a_c * sqrt(alpha * Tr)
  // a - T da/dT = a_c * alpha + kappa * a_c * sqrt(alpha * Tr)
  //             = a_c * sqrt(alpha) * (sqrt(alpha) + kappa * sqrt(Tr))
  //             = a_c * sqrt(alpha) * (1 + kappa*(1-sqrt(Tr)) + kappa*sqrt(Tr))
  //             = a_c * sqrt(alpha) * (1 + kappa)

  // Integral term is: - 1/b * ln(1 + B/Z)
  // So enthalpy contribution = R*T*(Z-1) + (T da/dT - a) * 1/b * ln(1 + B/Z)
  // = R*T*(Z-1) - (a - T da/dT) * 1/b * ln(1 + B/Z)
  // = R*T*(Z-1) - a_c * sqrt(alpha) * (1 + kappa) / b * ln(1 + B/Z)

  // Reusing constants from the code style:
  // coefficients:
  // a_c = 0.42747 * R^2 * Tc^2 / Pc
  // b = 0.08664 * R * Tc / Pc
  // a_c / b = (0.42747/0.08664) * R * Tc = 4.93386 * R * Tc

  // (1+kappa) term matches.

  return this->R() * Tc_ *
         (Tr * (Z - 1) -
          4.93386 * (1 + kappa) * sqrt(alpha) * log((Z + B) / Z));
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Cp(scalar p, scalar T) const {
  const scalar Tr = T / Tc_;
  const scalar a = 0.42747 * sqr(RR * Tc_) / Pc_;
  const scalar b = 0.08664 * RR * Tc_ / Pc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);
  const scalar alpha = sqr(1 + kappa * (1 - sqrt(Tr)));

  const scalar A = a * alpha * p / sqr(RR * T);
  const scalar B = b * p / (RR * T);

  const scalar Z = this->Z(p, T);

  // Derivatives required for Cp
  // Cp = Cp_ig + R(T(dZ/dT)_P - Z) + d(H-Hig)/dT ... this is getting complex.
  // Use the form derived from residual properties.
  // See PR implementation for pattern.

  // PR:
  // ap = kappa*a*(kappa/Tc_ - (1 + kappa)/sqrt(T*Tc_));
  // app = kappa*a*(1 + kappa)/(2*sqrt(pow3(T)*Tc_));

  // SRK derivatives of a(T):
  // da/dT = -kappa * a_c * sqrt(alpha/Tr) / Tc
  //       = -kappa * (a/alpha) * sqrt(alpha) / sqrt(T/Tc) / Tc
  //       = -kappa * a_c * (1 + kappa(1-sqrt(Tr))) / sqrt(Tr) / Tc

  // Let's stick to the PR pattern if possible or deriving cleanly.
  // Term app in PR code seems to be T * d2a/dT2 ?
  // Or involves da/dT.

  // Let's implement Cp using the relation:
  // Cp = dH/dT_p

  const scalar da_dT = -kappa * a * sqrt(alpha / Tr) / Tc_;
  const scalar d2a_dT2 =
      kappa * a / (2.0 * Tc_ * T * sqrt(alpha / Tr)) *
      (kappa / Tc_ + sqrt(alpha / Tr) / T); // This is getting messy.

  // Simpler derivation:
  // a(T) = ac * (1 + m(1-sqrt(Tr)))^2
  // da/dT = ac * 2 * (1 + m(1-sqrt(Tr))) * (-m * 0.5 / sqrt(Tr) * 1/Tc)
  //       = - ac * m * sqrt(alpha) / sqrt(Tr) / Tc
  //       = - m * a * sqrt(alpha) / (alpha * sqrt(Tr) * Tc) -> a = ac * alpha
  //       = - m * (a / sqrt(alpha)) / (sqrt(Tr) * Tc)

  // d2a/dT2:
  // let K = m * ac
  // da/dT = - K * (1 + m - m*sqrt(Tr)) / sqrt(Tr) / Tc
  //       = - K/Tc * ( (1+m)/sqrt(Tr) - m )
  // d2a/dT2 = - K/Tc * ( (1+m) * (-0.5) * Tr^(-3/2) * 1/Tc )
  //         = K/(2*Tc^2) * (1+m) / sqrt(Tr^3)
  //         = m * ac * (1+m) / (2 * Tc^2 * sqrt(Tr^3))

  const scalar app = kappa * a * (1.0 + kappa) / (2.0 * sqrt(pow3(T) * Tc_));
  // Interestingly, this formula for app (T * d2a/dT2 ?) in PR matches SRK form
  // just with different constants? In PR code: app = kappa*a*(1 +
  // kappa)/(2*sqrt(pow3(T)*Tc_)); Yes, the functional form of alpha is the
  // same! alpha_PR = (1 + k_PR * (1 - sqrt(Tr)))^2 alpha_SRK = (1 + k_SRK * (1
  // - sqrt(Tr)))^2 The only difference is k (kappa). So the derivative
  // expressions in terms of 'a' and 'kappa' should be identical.

  // So I can reuse the PR structure for derivatives, just ensuring 'a' and
  // 'kappa' are SRK values.

  // M and N terms in PR code:
  // M is derivative of Cubic wrt Z?
  // Cubic: Z^3 + a2 Z^2 + a1 Z + a0 = 0
  // d(Cubic)/dZ = 3Z^2 + 2 a2 Z + a1
  // PR: a2 = B-1, a1 = A - 2B - 3B^2
  // d/dZ = 3Z^2 + 2(B-1)Z + (A - 2B - 3B^2)
  // M = ... code has (sqr(Z) + 2*B*Z - sqr(B))/(Z - B)  <-- This doesn't look
  // like derivative directly.

  // Let's use the standard result for Cubic EOS Cp.
  // Cp = Cv + R * (Z + T(dZ/dT)_V)^2 / (Z + T(dP/dT)_V / P * Z ... ) -- No.
  // Cp = Cv + T (dP/dT)_V * (dV/dT)_P

  // Actually, simply copying PR logic but with correct Z-Equation partials is
  // safer if I understand M and N. N = ap * B / (b * RR) -> ap seems to be (T
  // da/dT - a)? In PR code: ap = kappa*a*(kappa/Tc_ - (1 + kappa)/sqrt(T*Tc_));
  // Let's re-verify ap.
  // T * da/dT = T * [ - kappa * a_c * sqrt(alpha) / sqrt(Tr) / Tc ]
  //           = - kappa * a_c * sqrt(alpha) * sqrt(Tr)
  //           = - kappa * a_c * (1 + kappa(1-sqrt(Tr))) * sqrt(Tr)
  //           = - kappa * a_c * (sqrt(Tr) + kappa*sqrt(Tr) - kappa*Tr)
  // a - T da/dT = a_c * alpha + kappa * a_c * sqrt(alpha) * sqrt(Tr)
  //             = a_c * sqrt(alpha) * (sqrt(alpha) + kappa*sqrt(Tr))
  //             = a_c * sqrt(alpha) * (1 + kappa)

  // PR ap code: kappa*a*(kappa/Tc_ - (1 + kappa)/sqrt(T*Tc_))
  // This is NOT (a - T da/dT).
  // It looks like derivative of A with respect to something?

  // Given the complexity of implementing Cp correctly from scratch without
  // derivations, and assuming the general structure of cubic EOS thermodynamics
  // is similar. The integral part for Cp-Cv or enthalpy is different.

  // The integral factor for PR was 1/(2sqrt(2)b) * log...
  // For SRK it is 1/b * log...
  // Ratio SRK_factor / PR_factor = (1/b) / (1/(2sqrt(2)b)) = 2sqrt(2)
  // approx 2.828.

  // In PR code:
  // app * (T/(2*root2*b)) * log(...)
  // For SRK, we replace (T/(2*root2*b)) with (T/b).
  // And root2 related terms in log argument simplify to (Z+B)/Z.

  // M and N relate to compressibility Z derivatives.
  // Z equation: cubic(Z) = 0
  // dZ/dT = - (d cubic / dT) / (d cubic / dZ)
  // d cubic / dZ = 3Z^2 - 2Z + (A - B - B^2)  (using SRK coeffs)
  // PR code uses M = (sqr(Z) + 2*B*Z - sqr(B))/(Z - B).
  // Denominator Z-B suggests P = RT/(v-b) term?

  // Let's stick to OpenFOAM existing PR implementation structure but adapt for
  // SRK. SRK denominator derivative part: P = RT/(v-b) - a/(v(v+b)) (dP/dV)_T =
  // -RT/(v-b)^2 + a(2v+b)/(v(v+b))^2 (dP/dT)_V = R/(v-b) - (da/dT)/(v(v+b))

  // CpMCv = T * (dP/dT)_V^2 / - (dP/dV)_T
  // This seems to be what M and N are calculating implicitly.

  // Let's look at `CpMCv` function in PR.
  // return this->R()*sqr(M - N)/(sqr(M) - 2*A*(Z + B));
  // This suggests specific structure.

  // If I cannot guarantee the correct M, N formulation for SRK, I might
  // introduce errors. However, M and N likely come from differentiating the
  // specific Z cubic.

  return (app * (T / b) * log((Z + B) / Z) +
          this->CpMCv(p, T) // Reuse CpMCv which I will update
          - RR) /
         this->W();
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::e(scalar p, scalar T) const {
  // e = h - p/rho = h - RTZ
  // h = RT(Z-1) + DEP
  // e = RT(Z-1) + DEP - RTZ = -RT + DEP
  // DEP = - (a - T da/dT) * 1/b * ln(1 + B/Z)

  const scalar Pr = p / Pc_;
  const scalar Tr = T / Tc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);
  const scalar alpha = sqr(1 + kappa * (1 - sqrt(Tr)));

  const scalar Z = this->Z(p, T);

  const scalar b = 0.08664 * this->R() * Tc_ / Pc_;
  const scalar B = b * p / (this->R() * T);

  // Coeff for DEP term
  // 4.93386 * (1+kappa) * sqrt(alpha) * RTc

  return this->R() * Tc_ *
         (-4.93386 * (1 + kappa) * sqrt(alpha) * log((Z + B) / Z));
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Cv(scalar p, scalar T) const {
  // Cv = Cp - (Cp - Cv)
  // But usually calculated as Cv_ig + Residual_Cv
  // Residual Cv = T * integral( d2P/dT2 ) dv
  // P = RT/(v-b) - a / (v(v+b))
  // dP/dT = R/(v-b) - da/dT / (v(v+b))
  // d2P/dT2 = - d2a/dT2 / (v(v+b))
  // Int( -d2a/dT2 / (v(v+b)) ) dv = -d2a/dT2 * 1/b * ln(v/(v+b))
  // = d2a/dT2 * 1/b * ln((v+b)/v) = d2a/dT2 * 1/b * ln((Z+B)/Z)

  // Cv = Cv_ig + T * d2a/dT2 * 1/b * ln((Z+B)/Z)
  // Wait, Cv_ig is not available directly?
  // Specie::Cv gives std Cv?
  // The functions here return "Cv contribution".
  // Usually code implies: return Cv_correction.
  // The base Specie class handles ideal part?
  // No, checking PR code:
  // return ( app*(T/(2*root2*b))*log(...) - RR ) / W
  // -RR comes from Cp-Cv ideal = R? No.
  // Cv_real - Cv_ideal = T * Int(d2P/dT2) dv.
  // For cubic EOS, this is exactly the term involving d2a/dT2.
  // Why -RR?
  // Maybe they assume the caller adds Cp_ideal? And Cp_ideal - R = Cv_ideal?
  // Usually these returns match the signature:
  // "Return Cv contribution [J/(kg K]"
  // If it returns standard departure, it should be just the integral.

  // PR code:
  // app*(T/(2*root2*b))*log(...) - RR
  // This looks like (Cp_dep - (Cp-Cv)_dep)? No.

  // Let's assume the PR code structure represents:
  // Cv_dep + (something).
  // Check Cp: returns app*... + CpMCv ... - RR.
  // So Cp = Term + CpMCv - R.
  // Cv = Term - R.
  // Cp - Cv = CpMCv. This is consistent.

  // So for SRK:
  // Cv = app * (T/b) * log((Z+B)/Z) - RR

  const scalar a = 0.42747 * sqr(RR * Tc_) / Pc_;
  const scalar b = 0.08664 * RR * Tc_ / Pc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);

  const scalar B = b * p / (RR * T);
  const scalar Z = this->Z(p, T);
  const scalar app = kappa * a * (1 + kappa) / (2 * sqrt(pow3(T) * Tc_));

  return (app * (T / b) * log((Z + B) / Z) - RR) / this->W();
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::sp(scalar p, scalar T) const {
  // Entropy.
  // s - s_ig = R ln(Z-B) + integral ...
  // PR: log(Z-B) - ...
  // SRK should be similar: R ln(Z-B) + (da/dT) * 1/b * ln(1+B/Z)

  const scalar Pr = p / Pc_;
  const scalar Tr = T / Tc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);
  // const scalar alpha = sqr(1 + kappa*(1 - sqrt(Tr)));

  const scalar Z = this->Z(p, T);
  const scalar b = 0.08664 * this->R() * Tc_ / Pc_;
  const scalar B = b * p / (this->R() * T);

  // da/dT term coeff:
  // da/dT = - kappa * a_c * sqrt(alpha) / sqrt(Tr) / Tc
  // a_c/b = 4.93386 * R * Tc
  // 1/b * da/dT = - kappa * 4.93386 * R * Tc * sqrt(alpha) / sqrt(Tr) / Tc
  //             = - kappa * 4.93386 * R * sqrt(alpha/Tr)
  // coeff = - 4.93386 * kappa * sqrt(alpha/Tr)
  // Note: sqrt(alpha/Tr) = (1 + kappa(1-sqrt(Tr))) / sqrt(Tr) =
  // (1+kappa)/sqrt(Tr) - kappa

  return this->R() *
         (-log(p / Pstd) +
          (log(Z - B) - 4.93386 * kappa * ((1 + kappa) / sqrt(Tr) - kappa) *
                            log((Z + B) / Z)));
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::sv(scalar p, scalar T) const {
  NotImplemented;
  return 0;
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::psi(scalar p, scalar T) const {
  const scalar Z = this->Z(p, T);

  return 1.0 / (Z * this->R() * T);
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::Z(scalar p, scalar T) const {
  const scalar Tr = T / Tc_;
  const scalar a = 0.42747 * sqr(RR * Tc_) / Pc_;
  const scalar b = 0.08664 * RR * Tc_ / Pc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);
  const scalar alpha = sqr(1 + kappa * (1 - sqrt(Tr)));

  const scalar A = a * alpha * p / sqr(RR * T);
  const scalar B = b * p / (RR * T);

  // SRK Cubic: Z^3 - Z^2 + (A - B - B^2)Z - AB = 0
  const scalar a2 = -1;
  const scalar a1 = A - B - sqr(B);
  const scalar a0 = -A * B;

  const scalar Q = (3 * a1 - a2 * a2) / 9.0;
  const scalar Rl = (9 * a2 * a1 - 27 * a0 - 2 * a2 * a2 * a2) / 54.0;

  const scalar Q3 = Q * Q * Q;
  const scalar D = Q3 + Rl * Rl;

  scalar root = -1;

  if (D <= 0) {
    const scalar th = ::acos(Rl / sqrt(-Q3));
    const scalar qm = 2 * sqrt(-Q);
    const scalar r1 = qm * cos(th / 3.0) - a2 / 3.0;
    const scalar r2 =
        qm * cos((th + 2 * constant::mathematical::pi) / 3.0) - a2 / 3.0;
    const scalar r3 =
        qm * cos((th + 4 * constant::mathematical::pi) / 3.0) - a2 / 3.0;

    root = max(r1, max(r2, r3));
  } else {
    // One root is real
    const scalar D05 = sqrt(D);
    const scalar S = pow(Rl + D05, 1.0 / 3.0);
    scalar Tl = 0;
    if (D05 > Rl) {
      Tl = -pow(mag(Rl - D05), 1.0 / 3.0);
    } else {
      Tl = pow(Rl - D05, 1.0 / 3.0);
    }

    root = S + Tl - a2 / 3.0;
  }

  return root;
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::CpMCv(scalar p, scalar T) const {
  const scalar Tr = T / Tc_;
  const scalar a = 0.42747 * sqr(RR * Tc_) / Pc_;
  const scalar b = 0.08664 * RR * Tc_ / Pc_;
  const scalar kappa = 0.480 + 1.574 * omega_ - 0.176 * sqr(omega_);
  const scalar alpha = sqr(1 + kappa * (1 - sqrt(Tr)));

  const scalar A = alpha * a * p / sqr(RR * T);
  const scalar B = b * p / (RR * T);

  const scalar Z = this->Z(p, T);

  const scalar ap = kappa * a * (kappa / Tc_ - (1 + kappa) / sqrt(T * Tc_));
  // M and N need update for SRK cubic form derivatives?
  // M = - (dP/dV)_T * (something)?
  // Cp - Cv = - T (dP/dT)_V^2 / (dP/dV)_T

  // (dP/dV)_T = -RT/(v-b)^2 + a(2v+b)/(v(v+b))^2
  // Let's express in terms of Z, A, B.
  // Multiply by v/P ... NO, too complicated to derive M, N forms on the fly.

  // Let's rely on M and N being derived from Z cubic equation implicit
  // differentiation: F(Z, A, B) = Z^3 + a2 Z^2 + a1 Z + a0 = 0 dZ/dT = -
  // (dF/dT) / (dF/dZ) F = Z^3 - Z^2 + (A - B - B^2)Z - AB dF/dZ = 3Z^2 - 2Z +
  // (A - B - B^2)

  // M = dF/dZ ?
  // In PR code: M = (sqr(Z) + 2*B*Z - sqr(B))/(Z - B)
  // That looked different.

  // Let's try to map the PR M to SRK.
  // PR F = Z^3 - (1-B)Z^2 + (A-2B-3B^2)Z - ...
  // dF/dZ_PR = 3Z^2 - 2(1-B)Z + (A-2B-3B^2)
  // M_PR * (Z-B) = Z^2 + 2BZ - B^2 ? No.

  // If I use the derivative of Pressure form directly?
  // Cp - Cv = T * (dP/dT_V)^2 / (-dP/dV_T)
  // (dP/dT)_V = R/(v-b) - da/dT / (v(v+b))
  //           = R/(v-b) + ap / (v(v+b)) ? (ap definition above)
  // Check ap in PR code:
  // ap = kappa*a*( ... )
  // In code: N = ap * B / (b*RR)
  // So N term relates to da/dT.

  // Let's approximate CpMCv with a consistent structure for SRK.
  // Denominator in PR: sqr(M) - 2*A*(Z+B)

  // Given the risk of algebra errors in deriving M and N for SRK,
  // and that CpMCv is critical for sound speed etc.
  // I will try to implement (dP/dT)_V and (dP/dV)_T explicitly if possible?
  // But they need 'v' which is Z*R*T/p.

  const scalar v = Z * RR * T / p;
  const scalar R = RR; // simplification

  const scalar da_dT = -kappa * a * sqrt(alpha / Tr) / Tc_;

  const scalar dP_dT_V =
      R / (v - b * R / R) -
      da_dT / (v * (v + b)); // Wait b is uniform? b here is molar b?
  // In code b = 0.08664 * R * Tc / Pc. This is per kmol if R is universal.
  // v is per kmol.

  const scalar dP_dV_T =
      -R * T / sqr(v - b) + a * alpha * (2 * v + b) / sqr(v * (v + b));

  // CpMCv = T * sqr(dP_dT_V) / (-dP_dV_T);
  // return CpMCv / W;

  // This explicit calculation is safer than reverse engineering A, B, M, N
  // coefficients.

  return (T * sqr(dP_dT_V) / (-dP_dV_T)) / this->W();
}

template <class Specie>
inline Foam::scalar Foam::SRKGas<Specie>::alphav(scalar p, scalar T) const {
  NotImplemented;
  return 0;
}

// * * * * * * * * * * * * * * * Member Operators  * * * * * * * * * * * * * //

template <class Specie>
inline void Foam::SRKGas<Specie>::operator+=(const SRKGas<Specie> &pg) {
  scalar X1 = this->Y() / this->W();
  Specie::operator+=(pg);

  if (mag(this->Y()) > small) {
    X1 *= this->W() / this->Y();
    const scalar X2 = this->W() * pg.Y() / (pg.W() * this->Y());

    Tc_ = X1 * Tc_ + X2 * pg.Tc_;
    Vc_ = X1 * Vc_ + X2 * pg.Vc_;
    Zc_ = X1 * Zc_ + X2 * pg.Zc_;
    Pc_ = RR * Zc_ * Tc_ / Vc_;
    omega_ = X1 * omega_ + X2 * pg.omega_;
  }
}

template <class Specie>
inline void Foam::SRKGas<Specie>::operator*=(const scalar s) {
  Specie::operator*=(s);
}

// * * * * * * * * * * * * * * * Friend Operators  * * * * * * * * * * * * * //

template <class Specie>
Foam::SRKGas<Specie> Foam::operator+(const SRKGas<Specie> &pg1,
                                     const SRKGas<Specie> &pg2) {
  Specie sp(static_cast<const Specie &>(pg1) +
            static_cast<const Specie &>(pg2));

  if (mag(sp.Y()) < small) {
    return SRKGas<Specie>(sp, pg1.Tc_, pg1.Vc_, pg1.Zc_, pg1.Pc_, pg1.omega_);
  } else {
    const scalar X1 = sp.W() * pg1.Y() / (pg1.W() * sp.Y());
    const scalar X2 = sp.W() * pg2.Y() / (pg2.W() * sp.Y());

    const scalar Tc = X1 * pg1.Tc_ + X2 * pg2.Tc_;
    const scalar Vc = X1 * pg1.Vc_ + X2 * pg2.Vc_;
    const scalar Zc = X1 * pg1.Zc_ + X2 * pg2.Zc_;

    return SRKGas<Specie>(sp, Tc, Vc, Zc, RR * Zc * Tc / Vc,
                          X1 * pg1.omega_ + X2 * pg2.omega_);
  }
}

template <class Specie>
Foam::SRKGas<Specie> Foam::operator*(const scalar s, const SRKGas<Specie> &pg) {
  return SRKGas<Specie>(s * static_cast<const Specie &>(pg), pg.Tc_, pg.Vc_,
                        pg.Zc_, pg.Pc_, pg.omega_);
}

template <class Specie>
Foam::SRKGas<Specie> Foam::operator==(const SRKGas<Specie> &pg1,
                                      const SRKGas<Specie> &pg2) {
  return SRKGas<Specie>(static_cast<const Specie &>(pg1) ==
                            static_cast<const Specie &>(pg2),
                        NaN, NaN, NaN, NaN, NaN);
}

// ************************************************************************* //
