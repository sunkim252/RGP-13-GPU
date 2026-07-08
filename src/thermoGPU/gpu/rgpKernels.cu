/*---------------------------------------------------------------------------*\
  RGP-13 thermoGPU — SRK EoS + Chung(1988) 수송 CUDA 커널

  CPU 레퍼런스 경로의 1:1 포팅 (Tier-0, no-diffusion):
    SRKchungTakaMixture::calcMixture  (compositionToX + calculateRealGas
                                       + X 보정 + updateTRANS_noDiffusion)
    SRKGasI.H                          rho(p,T): Cardano Z + Peneloux
    chungTransportI.H                  mu(p,T), kappa(p,T)
    janafThermoI.H                     Cp_ig (질량기준 계수 블렌드)

  설계 규칙: 이 파일은 OpenFOAM 헤더를 include하지 않는다.
  (OF Scalar.H의 j0/j1/y0/y1 Bessel 선언이 CUDA 내장과 충돌)
\*---------------------------------------------------------------------------*/

#include "rgpKernelTypes.H"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// * * * * * * * * * * * * * * 디바이스 테이블 상태 * * * * * * * * * * * * * //

namespace
{
    struct DeviceTables
    {
        int     nSpecies   = 0;
        int     stableRoot = 0;
        double  RR = 0, TlowJ = 0, ThighJ = 0, Tcommon = 0;

        double* W = nullptr;
        double* BM = nullptr;
        double* CM = nullptr;
        double* janafHigh = nullptr;
        double* janafLow = nullptr;
        double* COEF1 = nullptr;
        double* COEF2 = nullptr;
        double* COEF3 = nullptr;
        double* SIGMA3M = nullptr;
        double* EPSILONKM0 = nullptr;
        double* OMEGAM0 = nullptr;
        double* MM0 = nullptr;
        double* MIUIM0 = nullptr;
        double* KAPPAIM = nullptr;
    };

    struct DeviceBuffers
    {
        int     capCells = 0;
        double* p = nullptr;
        double* T = nullptr;
        double* Y = nullptr;      // capCells * nSpecies
        double* rho = nullptr;
        double* mu = nullptr;
        double* kappa = nullptr;
    };

    DeviceTables  gTab;
    DeviceBuffers gBuf;
    char          gErr[256] = "no error";

    int fail(cudaError_t e, const char* where)
    {
        snprintf(gErr, sizeof(gErr), "%s: %s", where, cudaGetErrorString(e));
        return int(e);
    }

    void freeTables()
    {
        double** ptrs[] =
        {
            &gTab.W, &gTab.BM, &gTab.CM, &gTab.janafHigh, &gTab.janafLow,
            &gTab.COEF1, &gTab.COEF2, &gTab.COEF3, &gTab.SIGMA3M,
            &gTab.EPSILONKM0, &gTab.OMEGAM0, &gTab.MM0, &gTab.MIUIM0,
            &gTab.KAPPAIM
        };
        for (auto pp : ptrs) { if (*pp) { cudaFree(*pp); *pp = nullptr; } }
        gTab.nSpecies = 0;
    }

    void freeBuffers()
    {
        double** ptrs[] =
        {
            &gBuf.p, &gBuf.T, &gBuf.Y, &gBuf.rho, &gBuf.mu, &gBuf.kappa
        };
        for (auto pp : ptrs) { if (*pp) { cudaFree(*pp); *pp = nullptr; } }
        gBuf.capCells = 0;
    }
}


// * * * * * * * * * * * * * * * 디바이스 함수 * * * * * * * * * * * * * * * //

namespace rgp
{

constexpr double SMALL = 1e-15;   // Foam::small

// 커널 파라미터 팩 (값 복사로 전달)
struct KParams
{
    int     n;
    int     stableRoot;
    double  RR, TlowJ, ThighJ, Tcommon;
    const double *W, *BM, *CM, *jH, *jL;
    const double *C1, *C2, *C3, *S3, *EK0, *OM0, *MM0, *MI0, *KA;
};


//- SRK 압축인자 Z: SRKGasI.H Z(p,T)의 1:1 포팅 (Cardano + 근 선택)
__device__ double srkZ
(
    const double p, const double T,
    const double bM, const double coef1, const double coef2,
    const double coef3, const double RR, const int stableRoot
)
{
    const double aAlpha = coef1 - coef2*sqrt(T) + coef3*T;

    const double RT = RR*T;
    const double A = aAlpha*p/(RT*RT);
    const double B = bM*p/RT;

    const double a2 = -1.0;
    const double a1 = A - B - B*B;
    const double a0 = -A*B;

    const double Q  = (3.0*a1 - a2*a2)/9.0;
    const double Rl = (9.0*a2*a1 - 27.0*a0 - 2.0*a2*a2*a2)/54.0;

    const double Q3 = Q*Q*Q;
    const double D  = Q3 + Rl*Rl;

    double root = -1.0;

    if (D <= 0.0)
    {
        // 실근 3개: 최소=액체, 최대=기체.
        double arg = Rl/sqrt(-Q3);
        arg = fmax(-1.0, fmin(1.0, arg));
        const double th = acos(arg);
        const double qm = 2.0*sqrt(-Q);
        const double pi = 3.141592653589793238462643383279502884;

        const double r0 = qm*cos(th/3.0) - a2/3.0;
        const double r1 = qm*cos((th + 2.0*pi)/3.0) - a2/3.0;
        const double r2 = qm*cos((th + 4.0*pi)/3.0) - a2/3.0;

        const double Zmin = fmin(r0, fmin(r1, r2));
        const double Zmax = fmax(r0, fmax(r1, r2));

        if (stableRoot)
        {
            // 최소 fugacity(안정상) 선택:
            //   ln(phi) = Z - 1 - ln(Z - B) - (A/B) ln(1 + B/Z)
            const double lnPhiL =
                (Zmin > B)
              ? Zmin - 1.0 - log(Zmin - B) - (A/B)*log(1.0 + B/Zmin)
              : 1e300;                       // co-volume 이하 액체근: 기각
            const double lnPhiV =
                Zmax - 1.0 - log(Zmax - B) - (A/B)*log(1.0 + B/Zmax);
            root = (lnPhiL <= lnPhiV) ? Zmin : Zmax;
        }
        else
        {
            root = Zmax;                     // legacy: 항상 기체근
        }
    }
    else
    {
        // 실근 1개 (CPU와 동일한 부호 처리)
        const double D05 = sqrt(D);
        double S = (Rl + D05 < 0.0) ? -cbrt(fabs(Rl + D05)) : cbrt(Rl + D05);
        double Tl = (D05 > Rl) ? -cbrt(fabs(Rl - D05)) : cbrt(Rl - D05);
        root = S + Tl - a2/3.0;
    }

    return root;
}


//- SRK rho [kg/m3]: SRKGasI.H rho(p,T) (EOS 내부 p-floor + Peneloux)
__device__ double srkRho
(
    double p, const double T,
    const double bM, const double coef1, const double coef2,
    const double coef3, const double cM,
    const double Wmix, const double RR, const int stableRoot
)
{
    p = fmax(p, 1e4);                        // EOS 내부 압력 floor
    const double Zv = srkZ(p, T, bM, coef1, coef2, coef3, RR, stableRoot);
    const double Rmass = RR/Wmix;
    const double rhoSRK = p/(Zv*Rmass*T);

    // Peneloux (믹스처는 상수 cM; updateEoS가 cq*=0으로 설정)
    if (cM != 0.0)
    {
        const double denom = 1.0 - cM*rhoSRK/Wmix;
        if (denom > SMALL)
        {
            return rhoSRK/denom;
        }
    }
    return rhoSRK;
}


//- SRK Cp departure [J/(kg K)]: SRKGasI.H Cp(p,T)
__device__ double srkCpDep
(
    const double p, const double T,
    const double bM, const double coef1, const double coef2,
    const double coef3, const double Wmix, const double RR,
    const int stableRoot
)
{
    const double sqrtT = sqrt(T);
    const double aAlpha = coef1 - coef2*sqrtT + coef3*T;
    const double daAlpha = -coef2/(2.0*sqrtT) + coef3;
    const double ddaAlpha = coef2/(4.0*T*sqrtT);

    const double RT = RR*T;
    const double A = aAlpha*p/(RT*RT);
    double B = bM*p/RT;
    if (B <= 0.0) { B = 1e-16; }

    const double Zv = srkZ(p, T, bM, coef1, coef2, coef3, RR, stableRoot);

    const double M = (Zv*Zv + B*Zv)/(Zv - B);
    const double N = daAlpha*B/(bM*RR);
    const double MmN = M - N;

    return
    (
        (T/bM)*ddaAlpha*log((Zv + B)/Zv)
      + RR*MmN*MmN/(M*M - A*(2.0*Zv + B))
      - RR
    )/Wmix;
}


//- 셀 하나의 rho/mu/kappa (CPU calcMixture → mu/kappa 경로 1:1)
__global__ void rgpEvaluateKernel
(
    const KParams  kp,
    const int      nCells,
    const double* __restrict__ pF,
    const double* __restrict__ TF,
    const double* __restrict__ YF,
    double* __restrict__ rhoF,
    double* __restrict__ muF,
    double* __restrict__ kappaF
)
{
    const int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    const int n = kp.n;
    const double RR = kp.RR;
    const double* Yc = YF + (size_t)celli*n;

    double X[RGP_GPU_MAX_SPECIES];

    // ── compositionToX (원시 Y → 몰분율) ──────────────────────────────
    double sumXb = 0.0;
    for (int i = 0; i < n; i++) { sumXb += Yc[i]/kp.W[i]; }
    if (sumXb == 0.0) { sumXb = 1e-30; }
    for (int i = 0; i < n; i++)
    {
        X[i] = (Yc[i]/kp.W[i])/sumXb;
        if (X[i] <= 0.0) { X[i] = 0.0; }
    }

    // ── 믹스처 몰질량 (specie 질량 블렌드: Wmix = ΣY / Σ(Y/W)) ─────────
    double sumY = 0.0, sumYoW = 0.0;
    for (int i = 0; i < n; i++)
    {
        sumY += Yc[i];
        sumYoW += Yc[i]/kp.W[i];
    }
    const double Wmix = (sumYoW != 0.0) ? sumY/sumYoW : kp.W[0];
    const double Rmass = RR/Wmix;

    // ── calculateRealGas: 쌍 혼합 (대각 + 2×상삼각) ────────────────────
    double bM = 0, coef1 = 0, coef2 = 0, coef3 = 0, cM = 0;
    double sigma3M = 0, epsilonkM0 = 0, omegaM0 = 0, MM0v = 0, miuiM0 = 0;
    double kappaiM = 0;

    for (int i = 0; i < n; i++)
    {
        bM += X[i]*kp.BM[i];
        cM += X[i]*kp.CM[i];

        const double Xi2 = X[i]*X[i];
        const int ii = i*n + i;
        coef1      += Xi2*kp.C1[ii];
        coef2      += Xi2*kp.C2[ii];
        coef3      += Xi2*kp.C3[ii];
        sigma3M    += Xi2*kp.S3[ii];
        epsilonkM0 += Xi2*kp.EK0[ii];
        omegaM0    += Xi2*kp.OM0[ii];
        MM0v       += Xi2*kp.MM0[ii];
        miuiM0     += Xi2*kp.MI0[ii];
        kappaiM    += Xi2*kp.KA[ii];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            const double txx = 2.0*X[i]*X[j];
            const int ij = i*n + j;
            coef1      += txx*kp.C1[ij];
            coef2      += txx*kp.C2[ij];
            coef3      += txx*kp.C3[ij];
            sigma3M    += txx*kp.S3[ij];
            epsilonkM0 += txx*kp.EK0[ij];
            omegaM0    += txx*kp.OM0[ij];
            MM0v       += txx*kp.MM0[ij];
            miuiM0     += txx*kp.MI0[ij];
            kappaiM    += txx*kp.KA[ij];
        }
    }

    if (sigma3M == 0.0) { sigma3M = 1e-30; }
    const double sigmaM = cbrt(sigma3M);

    double epsilonkM = epsilonkM0/sigma3M;
    if (epsilonkM == 0.0) { epsilonkM = 1e-30; }

    const double sigmaOverConst = sigmaM/0.809;
    double VcM = sigmaOverConst*sigmaOverConst*sigmaOverConst;
    if (VcM == 0.0) { VcM = 1e-30; }

    const double TcM = 1.2593*epsilonkM;
    const double omegaM = omegaM0/sigma3M;

    const double MM0_term = MM0v/(epsilonkM*sigmaM*sigmaM);
    double MM = MM0_term*MM0_term;
    if (MM == 0.0) { MM = 1e-30; }

    const double miuiM = sqrt(sqrt(miuiM0*sigma3M*epsilonkM));

    // ── X 보정 (calcMixture: updateTRANS 직전) ────────────────────────
    double sumXcorr = 0.0;
    for (int i = 0; i < n; i++) { X[i] += 1e-40; sumXcorr += X[i]; }
    double sumXmd = 0.0;
    for (int i = 0; i < n; i++) { X[i] /= sumXcorr; sumXmd += X[i]; }

    // ── JANAF 질량기준 계수 블렌드 (Σ (Y/ΣY)·a_i) ─────────────────────
    //   (kappa의 Cv_ideal용. OF 관례: 온도범위/Tcommon은 종-0 승계)
    double aJ[7];
    {
        const double* tab = (TF[celli] < kp.Tcommon) ? kp.jL : kp.jH;
        const double invSumY = (sumY != 0.0) ? 1.0/sumY : 0.0;
        for (int c = 0; c < 7; c++)
        {
            double acc = 0.0;
            for (int i = 0; i < n; i++)
            {
                acc += Yc[i]*tab[7*i + c];
            }
            aJ[c] = acc*invSumY;
        }
    }

    const double p = pF[celli];
    const double T = TF[celli];

    // ── rho (SRK + Peneloux) ──────────────────────────────────────────
    const double rho =
        srkRho(p, T, bM, coef1, coef2, coef3, cM, Wmix, RR, kp.stableRoot);
    rhoF[celli] = rho;

    // ── Chung calculate(p,T): 공통 중간량 ─────────────────────────────
    const double A_=1.16145, B_=0.14874, C_=0.52487, D_=0.77320, E_=2.16178;
    const double F_=2.43787, G_=-6.435e-4, H_=7.27371, S_=18.0323, W_=-0.76830;

    const double Tstar = T/epsilonkM;
    const double mur = 131.3*miuiM/sqrt(VcM*TcM);
    const double mur2 = mur*mur;
    const double mur4 = mur2*mur2;
    const double Fc = 1.0 - 0.2756*omegaM + 0.059035*mur4 + kappaiM;

    const double Omegast =
        A_*pow(Tstar, -B_) + C_*exp(-D_*Tstar) + E_*exp(-F_*Tstar)
      + G_*pow(Tstar, B_)*sin(S_*pow(Tstar, W_) - H_);

    const double VcM23 = cbrt(VcM*VcM);
    const double eta0 = (4.0785e-5)*sqrt(MM*T)*Fc/(Omegast*VcM23);

    const double rhoMolL = 1e-3*rho/Wmix;    // [mol/l]
    const double Ych = rhoMolL*VcM/6.0;
    const double oneMinusY = 1.0 - Ych;
    const double G1 = (1.0 - 0.5*Ych)/(oneMinusY*oneMinusY*oneMinusY);

    // ── Chung mu ──────────────────────────────────────────────────────
    double muOut = 0.0;
    if (sumXmd > 1e-16)
    {
        const double a0[10] = {6.32402, 0.0012102, 5.28346, 6.62263, 19.74540,
                           -1.89992, 24.27450, 0.79716, -0.23816, 0.068629};
        const double a1[10] = {50.41190, -0.0011536, 254.209, 38.0957, 7.63034,
                           -12.53670, 3.44945, 1.11764, 0.067695, 0.34793};
        const double a2c[10] = {-51.68, -0.0062571, -168.481, -8.46414,
                           -14.35440, 4.98529, -11.29130, 0.012348, -0.8163,
                            0.59256};
        const double a3[10] = {1189.02, 0.037283, 3898.270, 31.4178, 31.52670,
                           -18.1507, 69.3464, -4.1161, 4.02528, -0.72663};

        double MA[10];
        for (int i = 0; i < 10; i++)
        {
            MA[i] = a0[i] + a1[i]*omegaM + a2c[i]*mur4 + a3[i]*kappaiM;
        }

        const double G2 =
            (MA[0]*(1.0 - exp(-MA[3]*Ych))/Ych + MA[1]*G1*exp(MA[4]*Ych)
           + MA[2]*G1)
           /(MA[0]*MA[3] + MA[1] + MA[2]);

        const double etak = eta0*(1.0/G2 + MA[5]*Ych);
        const double etap =
            ((40.785e-6/sqrt(Tstar))*sqrt(MM*T)/VcM23)
           *MA[6]*Ych*Ych*G2*exp(MA[7] + MA[8]/Tstar + MA[9]/(Tstar*Tstar));

        muOut = 0.1*(etak + etap);           // [P] → [kg/m/s]
    }
    muF[celli] = muOut;

    // ── Chung kappa ───────────────────────────────────────────────────
    double kappaOut = 0.0;
    if (sumXmd > 0.001)
    {
        const double b0[7] = {2.41657, -0.50924, 6.61069, 14.54250, 0.79274,
                              -5.8634, 81.171};
        const double b1[7] = {0.74824, -1.50936, 5.62073, -8.91387, 0.82019,
                              12.8005, 114.1580};
        const double b2[7] = {-0.91858, -49.99120, 64.7599, -5.63794, -0.69369,
                               9.58926, -60.841};
        const double b3[7] = {121.721, 69.9834, 27.0389, 74.3435, 6.31734,
                              -65.5292, 466.7750};

        double MB[7];
        for (int j = 0; j < 7; j++)
        {
            MB[j] = b0[j] + b1[j]*omegaM + b2[j]*mur4 + b3[j]*kappaiM;
        }

        // Cv_ideal: 실제 thermo 체인 Cp(p_ref=100Pa, T) - R
        //   Cp(100,T) = JANAF_ig(블렌드 계수) + SRK Cp departure(100, T)
        const double pRef = 100.0;
        const double CpJanaf =
            ((((aJ[4]*T + aJ[3])*T + aJ[2])*T + aJ[1])*T + aJ[0]);
        const double CpRef =
            CpJanaf
          + srkCpDep(pRef, T, bM, coef1, coef2, coef3, Wmix, RR,
                     kp.stableRoot);

        const double CvIdeal = CpRef - Rmass;
        const double CvCal = CvIdeal*Wmix*0.2388e-3;   // [cal/mol K]
        const double Rcal = RR/(4.186*1e3);

        const double alpha = (CvCal/Rcal) - 1.5;
        const double beta =
            0.7862 - 0.7109*omegaM + 1.3168*omegaM*omegaM;
        const double Tr = T/TcM;
        const double Zch = 2.0 + 10.5*Tr*Tr;
        const double Psi =
            1.0 + alpha*(0.215 + 0.28288*alpha - 1.061*beta + 0.26665*Zch)
           /(0.6366 + beta*Zch + 1.061*alpha*beta);

        const double lamda0 = 7.452*(eta0/MM)*Psi;

        const double H2 =
            (MB[0]*(1.0 - exp(-MB[3]*Ych))/Ych + MB[1]*G1*exp(MB[4]*Ych)
           + MB[2]*G1)
           /(MB[0]*MB[3] + MB[1] + MB[2]);

        const double lamdak = lamda0*(1.0/H2 + MB[5]*Ych);
        const double lamdap =
            (3.039e-4*sqrt(TcM/MM)/VcM23)*MB[6]*Ych*Ych*H2*sqrt(Tr);

        kappaOut = 418.6798*(lamdak + lamdap);   // [cal/cm/s/K] → [W/m/K]
    }
    kappaF[celli] = kappaOut;
}

} // namespace rgp


// * * * * * * * * * * * * * * * C ABI 런처 * * * * * * * * * * * * * * * * //

extern "C"
{

int rgpGpuInit(int deviceId)
{
    cudaError_t e = cudaSetDevice(deviceId < 0 ? 0 : deviceId);
    if (e != cudaSuccess) return fail(e, "rgpGpuInit/cudaSetDevice");
    return 0;
}


int rgpGpuDeviceCount(void)
{
    int nd = 0;
    if (cudaGetDeviceCount(&nd) != cudaSuccess) return 0;
    return nd;
}


int rgpGpuUpload(const rgpGpuTables* t)
{
    if (!t || t->nSpecies <= 0 || t->nSpecies > RGP_GPU_MAX_SPECIES)
    {
        snprintf(gErr, sizeof(gErr),
                 "rgpGpuUpload: bad nSpecies (max %d)", RGP_GPU_MAX_SPECIES);
        return -1;
    }

    freeTables();

    const int n = t->nSpecies;
    gTab.nSpecies   = n;
    gTab.stableRoot = t->stableRoot;
    gTab.RR      = t->RR;
    gTab.TlowJ   = t->TlowJ;
    gTab.ThighJ  = t->ThighJ;
    gTab.Tcommon = t->Tcommon;

    struct { double** dst; const double* src; size_t count; } items[] =
    {
        {&gTab.W,          t->W,          (size_t)n},
        {&gTab.BM,         t->BM,         (size_t)n},
        {&gTab.CM,         t->CM,         (size_t)n},
        {&gTab.janafHigh,  t->janafHigh,  (size_t)7*n},
        {&gTab.janafLow,   t->janafLow,   (size_t)7*n},
        {&gTab.COEF1,      t->COEF1,      (size_t)n*n},
        {&gTab.COEF2,      t->COEF2,      (size_t)n*n},
        {&gTab.COEF3,      t->COEF3,      (size_t)n*n},
        {&gTab.SIGMA3M,    t->SIGMA3M,    (size_t)n*n},
        {&gTab.EPSILONKM0, t->EPSILONKM0, (size_t)n*n},
        {&gTab.OMEGAM0,    t->OMEGAM0,    (size_t)n*n},
        {&gTab.MM0,        t->MM0,        (size_t)n*n},
        {&gTab.MIUIM0,     t->MIUIM0,     (size_t)n*n},
        {&gTab.KAPPAIM,    t->KAPPAIM,    (size_t)n*n},
    };

    for (auto& it : items)
    {
        const size_t bytes = it.count*sizeof(double);
        cudaError_t e = cudaMalloc(it.dst, bytes);
        if (e != cudaSuccess) { freeTables(); return fail(e, "upload/malloc"); }
        e = cudaMemcpy(*it.dst, it.src, bytes, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) { freeTables(); return fail(e, "upload/memcpy"); }
    }

    return 0;
}


int rgpGpuEvaluate
(
    int nCells,
    const double* p,
    const double* T,
    const double* Y,
    double* rho,
    double* mu,
    double* kappa
)
{
    if (nCells <= 0) return 0;
    if (gTab.nSpecies <= 0)
    {
        snprintf(gErr, sizeof(gErr), "rgpGpuEvaluate: tables not uploaded");
        return -1;
    }

    const int n = gTab.nSpecies;

    if (nCells > gBuf.capCells)
    {
        freeBuffers();
        const size_t b1 = (size_t)nCells*sizeof(double);
        const size_t bn = (size_t)nCells*n*sizeof(double);
        cudaError_t e;
        if ((e = cudaMalloc(&gBuf.p,     b1)) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.T,     b1)) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.Y,     bn)) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.rho,   b1)) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.mu,    b1)) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.kappa, b1)) != cudaSuccess)
        {
            freeBuffers();
            return fail(e, "evaluate/malloc");
        }
        gBuf.capCells = nCells;
    }

    const size_t b1 = (size_t)nCells*sizeof(double);
    const size_t bn = (size_t)nCells*n*sizeof(double);

    cudaError_t e;
    if ((e = cudaMemcpy(gBuf.p, p, b1, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "evaluate/H2D p");
    if ((e = cudaMemcpy(gBuf.T, T, b1, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "evaluate/H2D T");
    if ((e = cudaMemcpy(gBuf.Y, Y, bn, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "evaluate/H2D Y");

    rgp::KParams kp;
    kp.n = n;
    kp.stableRoot = gTab.stableRoot;
    kp.RR = gTab.RR;
    kp.TlowJ = gTab.TlowJ;
    kp.ThighJ = gTab.ThighJ;
    kp.Tcommon = gTab.Tcommon;
    kp.W = gTab.W;  kp.BM = gTab.BM;  kp.CM = gTab.CM;
    kp.jH = gTab.janafHigh;  kp.jL = gTab.janafLow;
    kp.C1 = gTab.COEF1;  kp.C2 = gTab.COEF2;  kp.C3 = gTab.COEF3;
    kp.S3 = gTab.SIGMA3M;  kp.EK0 = gTab.EPSILONKM0;
    kp.OM0 = gTab.OMEGAM0;  kp.MM0 = gTab.MM0;
    kp.MI0 = gTab.MIUIM0;  kp.KA = gTab.KAPPAIM;

    constexpr int blockSize = 128;
    const int gridSize = (nCells + blockSize - 1)/blockSize;

    rgp::rgpEvaluateKernel<<<gridSize, blockSize>>>
    (
        kp, nCells, gBuf.p, gBuf.T, gBuf.Y, gBuf.rho, gBuf.mu, gBuf.kappa
    );
    if ((e = cudaGetLastError()) != cudaSuccess)
        return fail(e, "evaluate/launch");

    if ((e = cudaMemcpy(rho, gBuf.rho, b1, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "evaluate/D2H rho");
    if ((e = cudaMemcpy(mu, gBuf.mu, b1, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "evaluate/D2H mu");
    if ((e = cudaMemcpy(kappa, gBuf.kappa, b1, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "evaluate/D2H kappa");

    return 0;
}


void rgpGpuFree(void)
{
    freeTables();
    freeBuffers();
}


const char* rgpGpuLastError(void)
{
    return gErr;
}

} // extern "C"

// ************************************************************************* //
