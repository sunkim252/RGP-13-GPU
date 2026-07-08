/*---------------------------------------------------------------------------*\
  RGP-13 chemGPU — 등압 단열 반응기 RHS (호스트/디바이스 공유 구현)

  상태벡터 y[n+1] = { c_0..c_{n-1} [kmol/m3], T [K] }, p 고정.
    dc_s/dt = ωdot_s = Σ_r (νR-νL)_{rs} q_r
    dT/dt   = -Σ_s h_mol,s ωdot_s / Σ_s c_s cp_mol,s      (정압)
  가역 반응은 NASA7 Gibbs 기반 평형상수 Kc로 역방향 산출 (CHEMKIN 표준).
  nvcc가 include하면 __host__ __device__, g++이면 일반 inline — CPU
  레퍼런스(OF ODESolver)와 GPU가 '동일 소스'의 RHS를 쓴다.
\*---------------------------------------------------------------------------*/

#ifndef rgpChemRHS_H
#define rgpChemRHS_H

#include "rgpChemTypes.H"
#include <math.h>

#ifdef __CUDACC__
#define RGP_HD __host__ __device__ __forceinline__
#else
#define RGP_HD inline
#endif

namespace rgpchem
{

//- NASA7: 무차원 cp/R, h/RT, s/R (T 범위는 Tcommon 분기; 계수는 GRI 그대로)
RGP_HD void nasa7
(
    const double* a,      // [7]
    const double T,
    double& cpOverR,
    double& hOverRT,
    double& sOverR
)
{
    const double T2 = T*T, T3 = T2*T, T4 = T3*T;
    cpOverR = a[0] + a[1]*T + a[2]*T2 + a[3]*T3 + a[4]*T4;
    hOverRT = a[0] + a[1]*T/2 + a[2]*T2/3 + a[3]*T3/4 + a[4]*T4/5 + a[5]/T;
    sOverR  = a[0]*log(T) + a[1]*T + a[2]*T2/2 + a[3]*T3/3 + a[4]*T4/4 + a[6];
}


//- RHS: dydt[n+1] 채움. mech는 (디바이스에선 __constant__) 참조.
RGP_HD void chemRHS
(
    const rgpChemMech& m,
    const double p,
    const double* y,      // [n+1]
    double* dydt          // [n+1]
)
{
    const int n = m.nSpecies;
    double T = y[n];
    if (T < 250.0) { T = 250.0; }      // 가드 (Jacobian FD 섭동 보호)
    if (T > 5000.0) { T = 5000.0; }

    // 종 열역학 (무차원)
    double hRT[RGP_CHEM_MAX_SPECIES], gRT[RGP_CHEM_MAX_SPECIES];
    double cpR[RGP_CHEM_MAX_SPECIES];
    for (int s = 0; s < n; s++)
    {
        const double* a = (T < m.Tcommon[s]) ? m.janafLow[s] : m.janafHigh[s];
        double cpr, hrt, sr;
        nasa7(a, T, cpr, hrt, sr);
        cpR[s] = cpr;
        hRT[s] = hrt;
        gRT[s] = hrt - sr;             // g/RT
    }

    const double lnT = log(T);
    const double RTinv = 1.0/(m.RR*T);
    const double lnP0RT = log(m.pRef*RTinv);   // ln(p0/(RR T)) [kmol/m3]

    double wdot[RGP_CHEM_MAX_SPECIES];
    for (int s = 0; s < n; s++) { wdot[s] = 0.0; }

    for (int r = 0; r < m.nReactions; r++)
    {
        // 정방향 속도상수
        double kf = m.A[r]*exp(m.beta[r]*lnT - m.Ta[r]/T);

        // 3체 농도
        double cM = 0.0;
        if (m.tbType[r] != RGP_TB_NONE)
        {
            for (int s = 0; s < n; s++) { cM += m.eff[r][s]*y[s]; }
            if (cM < 0.0) { cM = 0.0; }
        }

        // 폴오프: k = kinf * Pr/(1+Pr) * F
        if (m.tbType[r] == RGP_TB_LIND || m.tbType[r] == RGP_TB_TROE)
        {
            const double k0 = m.A0[r]*exp(m.beta0[r]*lnT - m.Ta0[r]/T);
            const double Pr = fmax(k0*cM/fmax(kf, 1e-300), 1e-300);
            double F = 1.0;
            if (m.tbType[r] == RGP_TB_TROE)
            {
                const double al = m.troe[r][0];
                const double T3c = m.troe[r][1];
                const double T1c = m.troe[r][2];
                const double T2c = m.troe[r][3];
                double Fcent = (1.0 - al)*((T3c > 0.0) ? exp(-T/T3c) : 0.0)
                             + al*((T1c > 0.0) ? exp(-T/T1c) : 0.0)
                             + ((T2c > 0.0) ? exp(-T2c/T) : 0.0);
                Fcent = fmax(Fcent, 1e-300);
                const double logFc = log10(Fcent);
                const double cT = -0.4 - 0.67*logFc;
                const double nT = 0.75 - 1.27*logFc;
                const double logPr = log10(Pr);
                const double f1 = (logPr + cT)/(nT - 0.14*(logPr + cT));
                F = pow(10.0, logFc/(1.0 + f1*f1));
            }
            kf *= (Pr/(1.0 + Pr))*F;
        }

        // 역방향: kr = kf/Kc, ln(Kc) = -Σν g/RT + Δν ln(p0/RT)
        double kr = 0.0;
        double dnu = 0.0;
        if (m.reversible[r])
        {
            double sumG = 0.0;
            for (int s = 0; s < n; s++)
            {
                const double nu = m.nuR[r][s] - m.nuL[r][s];
                sumG += nu*gRT[s];
                dnu += nu;
            }
            const double lnKc = -sumG + dnu*lnP0RT;
            kr = kf*exp(-lnKc);
        }

        // 질량작용
        double qf = kf, qr = kr;
        for (int s = 0; s < n; s++)
        {
            const double cs = fmax(y[s], 0.0);
            if (m.nuL[r][s] == 1.0) { qf *= cs; }
            else if (m.nuL[r][s] == 2.0) { qf *= cs*cs; }
            else if (m.nuL[r][s] != 0.0) { qf *= pow(cs, m.nuL[r][s]); }
            if (m.reversible[r])
            {
                if (m.nuR[r][s] == 1.0) { qr *= cs; }
                else if (m.nuR[r][s] == 2.0) { qr *= cs*cs; }
                else if (m.nuR[r][s] != 0.0) { qr *= pow(cs, m.nuR[r][s]); }
            }
        }
        double q = qf - qr;
        if (m.tbType[r] == RGP_TB_M) { q *= cM; }

        for (int s = 0; s < n; s++)
        {
            const double nu = m.nuR[r][s] - m.nuL[r][s];
            if (nu != 0.0) { wdot[s] += nu*q; }
        }
    }

    // 에너지 (정압 단열)
    double sumH = 0.0, sumCcp = 0.0;
    for (int s = 0; s < n; s++)
    {
        dydt[s] = wdot[s];
        sumH += hRT[s]*wdot[s];                       // Σ (h/RT) ω
        sumCcp += fmax(y[s], 0.0)*cpR[s];             // Σ c cp/R
    }
    dydt[n] = -T*sumH/fmax(sumCcp, 1e-300);
}

//- 부분피벗 LU (host/device 공유 — 디버그 대조를 위해 단일 소스)
RGP_HD bool luDecomposeHD(double* Am, int* piv, const int n)
{
    for (int k = 0; k < n; k++)
    {
        int ip = k;
        double amax = fabs(Am[k*n + k]);
        for (int i = k + 1; i < n; i++)
        {
            const double v = fabs(Am[i*n + k]);
            if (v > amax) { amax = v; ip = i; }
        }
        if (amax < 1e-300) { return false; }
        piv[k] = ip;
        if (ip != k)
        {
            for (int j = 0; j < n; j++)
            {
                const double t = Am[k*n + j];
                Am[k*n + j] = Am[ip*n + j];
                Am[ip*n + j] = t;
            }
        }
        const double dinv = 1.0/Am[k*n + k];
        for (int i = k + 1; i < n; i++)
        {
            const double l = Am[i*n + k]*dinv;
            Am[i*n + k] = l;
            for (int j = k + 1; j < n; j++)
            {
                Am[i*n + j] -= l*Am[k*n + j];
            }
        }
    }
    return true;
}

RGP_HD void luSolveHD(const double* Am, const int* piv, double* b, const int n)
{
    for (int k = 0; k < n; k++)
    {
        const int ip = piv[k];
        if (ip != k) { const double t = b[k]; b[k] = b[ip]; b[ip] = t; }
        for (int i = k + 1; i < n; i++) { b[i] -= Am[i*n + k]*b[k]; }
    }
    for (int i = n - 1; i >= 0; i--)
    {
        double s = b[i];
        for (int j = i + 1; j < n; j++) { s -= Am[i*n + j]*b[j]; }
        b[i] = s/Am[i*n + i];
    }
}


//- Rosenbrock34 1스텝 (Shampine 계수, OF Rosenbrock34::solve와 동일 산술).
//  host/device 공유 — GPU 커널과 디버그 대조가 같은 코드를 실행한다.
//  반환: errNorm (max-norm, OF normaliseError와 동일).
RGP_HD double rosenbrock34Step
(
    const rgpChemMech& m,
    const double p,
    const double h,
    const double relTol,
    const double absTol,
    const double* y0,     // [neq]
    double* y,            // out [neq]
    double* wk            // 작업공간 [neq*(neq+7)] : J,LU 재사용 겸 k1..k4,dy0,dy
)
{
    const double a21 = 2.0, a31 = 48.0/25.0, a32 = 6.0/25.0,
        c21 = -8.0, c31 = 372.0/25.0, c32 = 12.0/5.0,
        c41 = -112.0/125.0, c42 = -54.0/125.0, c43 = -2.0/5.0,
        b1 = 19.0/9.0, b2 = 1.0/2.0, b3 = 25.0/108.0, b4 = 125.0/108.0,
        e1 = 34.0/108.0, e2 = 7.0/36.0, e4 = 125.0/108.0,
        gam = 1.0/2.0;

    const int n = m.nSpecies;
    const int neq = n + 1;

    double* LU = wk;                    // neq*neq
    double* k1 = wk + neq*neq;
    double* k2 = k1 + neq;
    double* k3 = k2 + neq;
    double* k4 = k3 + neq;
    double* dy0 = k4 + neq;
    double* dy = dy0 + neq;
    double* ytmp = dy + neq;
    int piv[RGP_CHEM_NEQ];

    chemRHS(m, p, y0, dy0);

    // FD Jacobian → LU = I/(γh) - J
    for (int j = 0; j < neq; j++)
    {
        for (int i = 0; i < neq; i++) { ytmp[i] = y0[i]; }
        const double d = fmax(fabs(y0[j])*1e-7, 1e-14);
        ytmp[j] += d;
        chemRHS(m, p, ytmp, dy);
        const double dinv = 1.0/d;
        for (int i = 0; i < neq; i++)
        {
            LU[i*neq + j] = -(dy[i] - dy0[i])*dinv;
        }
    }
    const double diag = 1.0/(gam*h);
    for (int i = 0; i < neq; i++) { LU[i*neq + i] += diag; }

    if (!luDecomposeHD(LU, piv, neq)) { return 1e300; }

    for (int i = 0; i < neq; i++) { k1[i] = dy0[i]; }
    luSolveHD(LU, piv, k1, neq);

    for (int i = 0; i < neq; i++) { ytmp[i] = y0[i] + a21*k1[i]; }
    chemRHS(m, p, ytmp, dy);
    for (int i = 0; i < neq; i++) { k2[i] = dy[i] + c21*k1[i]/h; }
    luSolveHD(LU, piv, k2, neq);

    for (int i = 0; i < neq; i++)
    {
        ytmp[i] = y0[i] + a31*k1[i] + a32*k2[i];
    }
    chemRHS(m, p, ytmp, dy);
    for (int i = 0; i < neq; i++)
    {
        k3[i] = dy[i] + (c31*k1[i] + c32*k2[i])/h;
    }
    luSolveHD(LU, piv, k3, neq);

    for (int i = 0; i < neq; i++)
    {
        k4[i] = dy[i] + (c41*k1[i] + c42*k2[i] + c43*k3[i])/h;
    }
    luSolveHD(LU, piv, k4, neq);

    double errNorm = 0.0;
    for (int i = 0; i < neq; i++)
    {
        y[i] = y0[i] + b1*k1[i] + b2*k2[i] + b3*k3[i] + b4*k4[i];
        const double err = e1*k1[i] + e2*k2[i] + e4*k4[i];
        const double sc =
            absTol + relTol*fmax(fabs(y0[i]), fabs(y[i]));
        const double r = fabs(err)/sc;
        if (r > errNorm) { errNorm = r; }
    }
    return errNorm;
}

} // namespace rgpchem

#endif

/* ************************************************************************* */
