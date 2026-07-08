/*---------------------------------------------------------------------------*\
  RGP-13 — GPU thermophysical property kernels (device-side)

  Phase 1: Peng-Robinson EoS + JANAF Cp + Sutherland transport
  셀 단위 독립 연산. 공간 구배 / 인접 셀 참조 없음.
\*---------------------------------------------------------------------------*/

#ifndef RGP13_thermoKernels_cuh
#define RGP13_thermoKernels_cuh

#include <cuda_runtime.h>
#include <cmath>

namespace RGP13
{

// ─── 상수 ──────────────────────────────────────────────────────────────
constexpr double RR = 8.314462618;     // J/(mol·K)
constexpr double Tstd = 298.15;

// ─── JANAF 계수 구조체 ────────────────────────────────────────────────
struct JanafCoeffs
{
    double Tlow, Thigh, Tcommon;
    double low[7];       // a1..a7 (저온 영역)
    double high[7];      // a1..a7 (고온 영역)
    double W;            // 분자량 [kg/mol]
};

// ─── Peng-Robinson 파라미터 ───────────────────────────────────────────
struct PRCoeffs
{
    double Tc;           // 임계 온도 [K]
    double Pc;           // 임계 압력 [Pa]
    double omega;        // acentric factor
};

// ─── Sutherland 수송 파라미터 ─────────────────────────────────────────
struct SutherlandCoeffs
{
    double As;           // mu = As * sqrt(T) / (1 + Ts/T)
    double Ts;
    double Pr;           // Prandtl number (for kappa = mu*Cp/Pr)
};

// ─── 통합 물성 계수 ──────────────────────────────────────────────────
struct ThermoCoeffs
{
    JanafCoeffs     janaf;
    PRCoeffs        pr;
    SutherlandCoeffs suth;
};


// =====================================================================
//  Device 함수 (inline)
// =====================================================================

// ─── JANAF Cp [J/(kg·K)] ────────────────────────────────────────────
__device__ __forceinline__
double janaf_Cp(const JanafCoeffs& c, double T)
{
    const double* a = (T < c.Tcommon) ? c.low : c.high;
    // NASA 7-coeff Cp/R, 이미 R/W 스케일 적용된 계수 가정
    return ((((a[4]*T + a[3])*T + a[2])*T + a[1])*T + a[0]);
}

// ─── JANAF hs (sensible enthalpy) [J/kg] ────────────────────────────
__device__ __forceinline__
double janaf_hs(const JanafCoeffs& c, double T)
{
    const double* a = (T < c.Tcommon) ? c.low : c.high;
    double ha = (((((a[4]/5.0*T + a[3]/4.0)*T + a[2]/3.0)*T
                   + a[1]/2.0)*T + a[0])*T + a[5]);

    // hf (formation enthalpy at Tstd)
    const double* al = c.low;
    double hf = (((((al[4]/5.0*Tstd + al[3]/4.0)*Tstd + al[2]/3.0)*Tstd
                    + al[1]/2.0)*Tstd + al[0])*Tstd + al[5]);

    return ha - hf;
}

// ─── Peng-Robinson EoS ──────────────────────────────────────────────
//     Z^3 + c2*Z^2 + c1*Z + c0 = 0
//     rho = p*W / (Z*R*T)

__device__ __forceinline__
double pr_compressibilityZ(const PRCoeffs& c, double p, double T)
{
    double Tr = T / c.Tc;
    double kappa_pr = 0.37464 + 1.54226*c.omega - 0.26992*c.omega*c.omega;
    double alpha = 1.0 + kappa_pr*(1.0 - sqrt(Tr));
    alpha *= alpha;

    double a = 0.45724 * RR*RR * c.Tc*c.Tc / c.Pc * alpha;
    double b = 0.07780 * RR * c.Tc / c.Pc;

    double A = a * p / (RR*RR * T*T);
    double B = b * p / (RR * T);

    // Cubic:  Z^3 - (1-B)*Z^2 + (A - 3B^2 - 2B)*Z - (AB - B^2 - B^3) = 0
    double c2 = -(1.0 - B);
    double c1 = A - 3.0*B*B - 2.0*B;
    double c0 = -(A*B - B*B - B*B*B);

    // Cardano's method — select largest real root (vapor phase)
    double Q = (3.0*c1 - c2*c2) / 9.0;
    double R_ = (9.0*c2*c1 - 27.0*c0 - 2.0*c2*c2*c2) / 54.0;
    double D = Q*Q*Q + R_*R_;

    double Z;
    if (D >= 0.0)
    {
        double sqrtD = sqrt(D);
        double S = cbrt(R_ + sqrtD);
        double Tv = cbrt(R_ - sqrtD);
        Z = S + Tv - c2 / 3.0;
    }
    else
    {
        double theta = acos(R_ / sqrt(-Q*Q*Q));
        double sqrtQ = 2.0 * sqrt(-Q);
        // Three real roots, pick the largest (vapor)
        double Z1 = sqrtQ * cos(theta / 3.0) - c2 / 3.0;
        double Z2 = sqrtQ * cos((theta + 2.0*M_PI) / 3.0) - c2 / 3.0;
        double Z3 = sqrtQ * cos((theta + 4.0*M_PI) / 3.0) - c2 / 3.0;
        Z = fmax(Z1, fmax(Z2, Z3));
    }

    return fmax(Z, B + 1e-10);   // Z > B always (physical)
}

__device__ __forceinline__
double pr_rho(const PRCoeffs& prc, double W, double p, double T)
{
    double Z = pr_compressibilityZ(prc, p, T);
    return p * W / (Z * RR * T);
}

__device__ __forceinline__
double pr_psi(const PRCoeffs& prc, double W, double p, double T)
{
    // psi = drho/dp at constant T  ≈  rho/p  for ideal, exact for PR
    double Z = pr_compressibilityZ(prc, p, T);
    return W / (Z * RR * T);   // = rho / p
}

// ─── Departure function: Cp_departure for PR EoS [J/(mol·K)] ──────
//     (simplified — full departure includes dZ/dT terms)

// ─── Sutherland transport ───────────────────────────────────────────
__device__ __forceinline__
double sutherland_mu(const SutherlandCoeffs& c, double T)
{
    return c.As * sqrt(T) / (1.0 + c.Ts / T);
}

__device__ __forceinline__
double sutherland_kappa(const SutherlandCoeffs& c, double mu, double Cp)
{
    return mu * Cp / c.Pr;
}


// ─── Newton iteration: T from sensible enthalpy ────────────────────
__device__ __forceinline__
double The_newton(const JanafCoeffs& jc, double he, double p, double T0)
{
    double T = T0;

    #pragma unroll 4
    for (int iter = 0; iter < 100; ++iter)
    {
        double f  = janaf_hs(jc, T) - he;
        double df = janaf_Cp(jc, T);

        double Tnew = T - f / df;

        // Clamp to valid range
        Tnew = fmax(jc.Tlow, fmin(Tnew, jc.Thigh));

        if (fabs(Tnew - T) < T * 1e-4) return Tnew;
        T = Tnew;
    }
    return T;
}


// =====================================================================
//  Global kernel
// =====================================================================

__global__
void thermoCalculateKernel
(
    // Input (read-only)
    const double* __restrict__ he,
    const double* __restrict__ p,

    // Input/Output
    double* __restrict__ T,

    // Output
    double* __restrict__ Cp,
    double* __restrict__ Cv,
    double* __restrict__ psi,
    double* __restrict__ rho,
    double* __restrict__ mu,
    double* __restrict__ kappa,

    // Coefficients (constant memory candidate)
    const ThermoCoeffs coeffs,

    // Local cell count (NOT mesh.nCells() — MPI-decomposed size)
    const int nCells
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nCells) return;

    const JanafCoeffs&      jc = coeffs.janaf;
    const PRCoeffs&         pc = coeffs.pr;
    const SutherlandCoeffs& sc = coeffs.suth;

    // 1. Temperature from enthalpy (Newton)
    T[i] = The_newton(jc, he[i], p[i], T[i]);

    // 2. Thermodynamic properties
    Cp[i]  = janaf_Cp(jc, T[i]);
    // Cv ≈ Cp - R/W  (ideal gas contribution; PR departure omitted Phase 1)
    Cv[i]  = Cp[i] - RR / jc.W;

    // 3. Equation of state
    rho[i] = pr_rho(pc, jc.W, p[i], T[i]);
    psi[i] = pr_psi(pc, jc.W, p[i], T[i]);

    // 4. Transport
    mu[i]    = sutherland_mu(sc, T[i]);
    kappa[i] = sutherland_kappa(sc, mu[i], Cp[i]);
}


} // End namespace RGP13

#endif

// ************************************************************************* //
