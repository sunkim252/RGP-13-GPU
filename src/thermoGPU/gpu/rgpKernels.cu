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
#include "rgpFgmTypes.H"
#include "rgpStage.H"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//- unified 모드 (rgpStage.H 참조; 모든 nvcc TU가 공유)
int gRgpUnified = 0;
int gRgpCoherentHW = 0;   // rgpGpuInit이 하드웨어 코히런스 감지 시 1

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
        int     capOut = 0;       // p + rho..psi 용량 [cells]
        int     capStage = 0;     // T + Y 스테이징 용량 [cells]
        double* p = nullptr;
        double* T = nullptr;
        double* Y = nullptr;      // capStage * nSpecies
        double* rho = nullptr;
        double* mu = nullptr;
        double* kappa = nullptr;
        double* Cp = nullptr;
        double* Cv = nullptr;
        double* psi = nullptr;

        // fgm 디바이스 체인용: 종→fgm 필드 맵 + 비테이블 종 호스트 Y
        int*    yMap = nullptr;   // [RGP_GPU_MAX_SPECIES]
        int*    cMap = nullptr;   // [13] Tier-2 계수 fgm 필드 인덱스
        double* yH = nullptr;     // [yHCap]
        size_t  yHCap = 0;
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
            &gBuf.p, &gBuf.T, &gBuf.Y, &gBuf.rho, &gBuf.mu, &gBuf.kappa,
            &gBuf.Cp, &gBuf.Cv, &gBuf.psi, &gBuf.yH
        };
        for (auto pp : ptrs) { if (*pp) { cudaFree(*pp); *pp = nullptr; } }
        if (gBuf.yMap) { cudaFree(gBuf.yMap); gBuf.yMap = nullptr; }
        if (gBuf.cMap) { cudaFree(gBuf.cMap); gBuf.cMap = nullptr; }
        gBuf.capOut = 0;
        gBuf.capStage = 0;
        gBuf.yHCap = 0;
    }

    //- p + 출력 6개 (grow-only). 체인은 전체 셀, 플레인은 블록 크기.
    int ensureOut(int cells)
    {
        if (cells <= gBuf.capOut) return 0;
        double** ps[] =
        {
            &gBuf.p, &gBuf.rho, &gBuf.mu, &gBuf.kappa,
            &gBuf.Cp, &gBuf.Cv, &gBuf.psi
        };
        for (auto pp : ps) { if (*pp) { cudaFree(*pp); *pp = nullptr; } }
        gBuf.capOut = 0;
        const size_t b1 = (size_t)cells*sizeof(double);
        for (auto pp : ps)
        {
            cudaError_t e = cudaMalloc(pp, b1);
            if (e != cudaSuccess)
            {
                for (auto q : ps) { if (*q) { cudaFree(*q); *q = nullptr; } }
                cudaGetLastError();
                return fail(e, "ensureOut/malloc");
            }
        }
        gBuf.capOut = cells;
        return 0;
    }

    //- T + Y 스테이징 (grow-only) — 플레인(copy/mapped-폴백) 경로 전용.
    //  체인 경로는 fgm SoA를 직접 소비하므로 이 버퍼가 필요 없다
    //  (구 설계는 여기에 nc×n AoS 사본을 만들어 106종 2.6M에서
    //  2.2GB를 낭비했다 — VRAM 절약의 핵심 지점).
    int ensureStage(int cells)
    {
        if (cells <= gBuf.capStage) return 0;
        const int n = gTab.nSpecies;
        if (gBuf.T) { cudaFree(gBuf.T); gBuf.T = nullptr; }
        if (gBuf.Y) { cudaFree(gBuf.Y); gBuf.Y = nullptr; }
        gBuf.capStage = 0;
        cudaError_t e;
        if ((e = cudaMalloc(&gBuf.T, (size_t)cells*sizeof(double)))
            != cudaSuccess ||
            (e = cudaMalloc(&gBuf.Y, (size_t)cells*n*sizeof(double)))
            != cudaSuccess)
        {
            if (gBuf.T) { cudaFree(gBuf.T); gBuf.T = nullptr; }
            if (gBuf.Y) { cudaFree(gBuf.Y); gBuf.Y = nullptr; }
            cudaGetLastError();
            return fail(e, "ensureStage/malloc");
        }
        gBuf.capStage = cells;
        return 0;
    }

    //- 플레인 경로 스테이징 블록 크기 [cells]: 스테이징 총량을
    //  RGP_GPU_STAGE_MB(기본 512MB)로 캡 — 대격자×다종에서 VRAM 상한
    //  보장 (셀별 독립 커널이라 블록 분할은 비트-동일).
    int stageBlockCells(int nCells)
    {
        if (gRgpUnified == 2) return nCells;   // native: 스테이징 없음
        const char* env = getenv("RGP_GPU_STAGE_MB");
        size_t mb = env ? (size_t)atoll(env) : 512;
        if (mb < 64) mb = 64;
        const size_t perCell = (size_t)(gTab.nSpecies + 9)*sizeof(double);
        size_t b = (mb << 20)/perCell;
        if (b < 65536) b = 65536;
        return (b < (size_t)nCells) ? (int)b : nCells;
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

    const double pf = fmax(p, 1e4);   // EOS p-floor (upstream 34e2c5c 미러)
    const double RT = RR*T;
    const double A = aAlpha*pf/(RT*RT);
    const double B = bM*pf/RT;

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


//- SRK Cp departure + CpMCv [J/(kg K)]: SRKGasI.H Cp(p,T)/CpMCv(p,T)
//  (동일한 A, B, Z, M, N을 공유하므로 한 번에 계산)
__device__ void srkCpDepCpMCv
(
    const double p, const double T,
    const double bM, const double coef1, const double coef2,
    const double coef3, const double Wmix, const double RR,
    const int stableRoot,
    double& cpDep,      // SRKGasI::Cp — EOS departure [J/(kg K)]
    double& cpMCv       // SRKGasI::CpMCv = R_mass*(M-N)^2/(M^2-A(2Z+B))
)
{
    const double pfl = fmax(p, 1e4);   // EOS p-floor (upstream 34e2c5c 미러)
    const double sqrtT = sqrt(T);
    const double aAlpha = coef1 - coef2*sqrtT + coef3*T;
    const double daAlpha = -coef2/(2.0*sqrtT) + coef3;
    const double ddaAlpha = coef2/(4.0*T*sqrtT);

    const double RT = RR*T;
    const double A = aAlpha*pfl/(RT*RT);
    double B = bM*pfl/RT;
    if (B <= 0.0) { B = 1e-16; }

    const double Zv = srkZ(pfl, T, bM, coef1, coef2, coef3, RR, stableRoot);

    const double M = (Zv*Zv + B*Zv)/(Zv - B);
    const double N = daAlpha*B/(bM*RR);
    const double MmN = M - N;
    const double denom = M*M - A*(2.0*Zv + B);

    cpDep =
    (
        (T/bM)*ddaAlpha*log((Zv + B)/Zv)
      + RR*MmN*MmN/denom
      - RR
    )/Wmix;

    cpMCv = (RR/Wmix)*MmN*MmN/denom;
}


//- SRK psi = (drho/dp)_T [s^2/m^2]: SRKGasI.H psi(p,T) — 전방 FD
__device__ double srkPsi
(
    double p, const double T,
    const double bM, const double coef1, const double coef2,
    const double coef3, const double cM,
    const double Wmix, const double RR, const int stableRoot
)
{
    p = fmax(p, 1e4);                        // EOS 내부 압력 floor
    const double dp = 1e-3*p;
    const double rho0 =
        srkRho(p, T, bM, coef1, coef2, coef3, cM, Wmix, RR, stableRoot);
    const double rho1 =
        srkRho(p + dp, T, bM, coef1, coef2, coef3, cM, Wmix, RR, stableRoot);
    const double dRhodP = (rho1 - rho0)/dp;
    return fmax(dRhodP, 1e-3*rho0/p);
}


//- Y 조성 뷰: 커널이 두 레이아웃을 동일 산술로 소비 —
//  ⓐ aos != null: 셀-우선 AoS [nCells*n] (플레인 스테이징 경로)
//  ⓑ aos == null: fgm SoA 직접 소비 (체인 경로 — AoS 사본을 만들지
//     않아 nc×n×8B(106종 2.6M에서 2.2GB)를 절약; 필드-우선이라
//     이웃 스레드=이웃 셀 읽기가 코얼레스드).
//  map[s] >= 0: fgm 필드 인덱스, < 0: yh SoA의 (-1-map[s])행.
struct YView
{
    const double* aos;
    const double* soa;
    const int*    map;
    const double* yh;
    size_t        stride;   // soa/yh 행 스트라이드 (= 체인 nCells)

    // ⓒ SoA-생략(VRAM 다이어트) 모드: tab != null이면 map[s] >= 0의
    //   Y를 fgm 테이블에서 셀 스텐실 재유도로 직접 보간 (fgmKernel과
    //   동일 산술·동일 입력 → 비트-동일). 셀당 스텐실 1회 계산 후
    //   전 종 공유 — rgpFgmKernels.cu의 cellStencilSaved와 1:1 유지.
    const double* tab = nullptr;   // node-major tables
    int tabNF = 0;                 // 테이블 필드 수 (노드 스트라이드)
    int nZ = 0, nG = 0, nC = 0, nK = 0, mode4 = 0;
    double chi0 = 0, hOx = 0, hFuel = 0, Wlo = 0, Whi = 0;
    const double* Zax = nullptr; const double* Gax = nullptr;
    const double* Cax = nullptr; const double* Kax = nullptr;
    const double* Zf = nullptr; const double* Cf = nullptr;
    const double* gZf = nullptr; const double* chif = nullptr;
    const double* hwf = nullptr;
};

//- fMax/fMin·bracket·interp16: rgpFgmKernels.cu와 1:1 (별도 TU라 복제)
__device__ inline double rgpsFMax(double a, double b)
{ return (a > b) ? a : b; }
__device__ inline double rgpsFMin(double a, double b)
{ return (a < b) ? a : b; }

__device__ inline void rgpsBracket
(
    const double* axis, const int n, double v, int& i, double& w
)
{
    constexpr double VSMALL = 1e-300;
    if (n <= 1) { i = 0; w = 0; return; }
    if (v <= axis[0]) { i = 0; w = 0; return; }
    if (v >= axis[n - 1]) { i = n - 2; w = 1; return; }
    const double span = axis[n - 1] - axis[0];
    int j = 1;
    if (span > VSMALL)
    {
        j = int((v - axis[0])/span*(n - 1)) + 1;
        j = max(1, min(j, n - 1));
    }
    while (j < n - 1 && axis[j] < v) { j++; }
    while (j > 1 && axis[j - 1] >= v) { j--; }
    i = j - 1;
    const double d = axis[j] - axis[i];
    w = (d > VSMALL) ? (v - axis[i])/d : 0.0;
}

__device__ inline double rgpsInterp16
(
    const double* __restrict__ tab, const size_t base[16], const int f,
    const double wZ, const double wG, const double wC, const double wK
)
{
    const double c0000 = tab[base[0]  + f];
    const double c1000 = tab[base[1]  + f];
    const double c0100 = tab[base[2]  + f];
    const double c1100 = tab[base[3]  + f];
    const double c0010 = tab[base[4]  + f];
    const double c1010 = tab[base[5]  + f];
    const double c0110 = tab[base[6]  + f];
    const double c1110 = tab[base[7]  + f];
    const double c0001 = tab[base[8]  + f];
    const double c1001 = tab[base[9]  + f];
    const double c0101 = tab[base[10] + f];
    const double c1101 = tab[base[11] + f];
    const double c0011 = tab[base[12] + f];
    const double c1011 = tab[base[13] + f];
    const double c0111 = tab[base[14] + f];
    const double c1111 = tab[base[15] + f];
    const double a00 = c0000*(1 - wZ) + c1000*wZ;
    const double a10 = c0100*(1 - wZ) + c1100*wZ;
    const double a01 = c0010*(1 - wZ) + c1010*wZ;
    const double a11 = c0110*(1 - wZ) + c1110*wZ;
    const double a0  = a00*(1 - wG) + a10*wG;
    const double a1  = a01*(1 - wG) + a11*wG;
    const double A   = a0*(1 - wC) + a1*wC;
    const double b00 = c0001*(1 - wZ) + c1001*wZ;
    const double b10 = c0101*(1 - wZ) + c1101*wZ;
    const double b01 = c0011*(1 - wZ) + c1011*wZ;
    const double b11 = c0111*(1 - wZ) + c1111*wZ;
    const double b0  = b00*(1 - wG) + b10*wG;
    const double b1  = b01*(1 - wG) + b11*wG;
    const double B   = b0*(1 - wC) + b1*wC;
    return A*(1 - wK) + B*wK;
}

//- 셀 스텐실 재유도 + Yc[] 채움 (ⓒ 모드; cellStencilSaved와 1:1)
__device__ inline void rgpYFillFromTab
(
    const YView& v, const int n, const int celli, double* Yc
)
{
    const double Zcl = rgpsFMax(rgpsFMin(v.Zf[celli], 1.0), 0.0);
    const double Ccl = rgpsFMax(v.Cf[celli], 0.0);
    const double gz  = v.gZf[celli];
    double coord4 = v.chi0;
    if (v.mode4 == 1) { coord4 = v.chif[celli]; }
    else if (v.mode4 == 2)
    {
        coord4 = v.hwf[celli] - ((1.0 - Zcl)*v.hOx + Zcl*v.hFuel);
    }
    else if (v.mode4 == 3)
    {
        coord4 = rgpsFMax(v.Wlo, rgpsFMin(v.Whi, v.hwf[celli]));
    }

    int iZ, iG, iC, iK;
    double wZ, wG, wC, wK;
    rgpsBracket(v.Zax, v.nZ, Zcl, iZ, wZ);
    rgpsBracket(v.Gax, v.nG, gz, iG, wG);
    rgpsBracket(v.Cax, v.nC, Ccl, iC, wC);
    rgpsBracket(v.Kax, v.nK, coord4, iK, wK);
    const int iZp = (v.nZ >= 2) ? (iZ + 1) : iZ;
    const int iGp = (v.nG >= 2) ? (iG + 1) : iG;
    const int iCp = (v.nC >= 2) ? (iC + 1) : iC;
    const int iKp = (v.nK >= 2) ? (iK + 1) : iK;

    size_t base[16];
    int m = 0;
    for (int dK = 0; dK < 2; dK++)
    {
        const int k = dK ? iKp : iK;
        for (int dC2 = 0; dC2 < 2; dC2++)
        {
            const int c = dC2 ? iCp : iC;
            for (int dG = 0; dG < 2; dG++)
            {
                const int g = dG ? iGp : iG;
                for (int dZ = 0; dZ < 2; dZ++)
                {
                    const int z = dZ ? iZp : iZ;
                    base[m++] =
                        ((((size_t)z*v.nG + g)*v.nC + c)*v.nK + k)
                       *(size_t)v.tabNF;
                }
            }
        }
    }

    for (int s = 0; s < n; s++)
    {
        const int ms = v.map[s];
        Yc[s] = (ms >= 0)
          ? rgpsInterp16(v.tab, base, ms, wZ, wG, wC, wK)
          : v.yh[(size_t)(-1 - ms)*v.stride + celli];
    }
}

__device__ inline double rgpYAt
(
    const YView& v, const int n, const int cell, const int s
)
{
    if (v.aos) { return v.aos[(size_t)cell*n + s]; }
    const int m = v.map[s];
    return (m >= 0)
      ? v.soa[(size_t)m*v.stride + cell]
      : v.yh[(size_t)(-1 - m)*v.stride + cell];
}

//- Tier-2 혼합계수 뷰: map != null이면 fgm SoA에서 보간된 13계수
//  [bM,coef1..3,cM,sigmaM,epsilonkM,MM,VcM,TcM,omegaM,miuiM,kappaiM]
//  를 직접 읽어 셀당 O(n^2) 쌍 혼합을 건너뛴다 (CPU Tier-2 lookup 1:1)
struct TabView
{
    const double* soa;
    const int*    map;    // [13] fgm 필드 인덱스 (디바이스)
    size_t        stride;
};

__device__ inline double rgpCoeffAt
(
    const TabView& t, const int k, const int cell
)
{
    return t.soa[(size_t)t.map[k]*t.stride + cell];
}


//- 셀 하나의 ha [J/kg]: janafThermo::ha(질량기준 계수 블렌드) +
//  SRKGas::h(믹스처 departure) — updateManifold의 he 재시드
//  thermo_.he(p,T)의 1:1 포트. EOS 혼합은 rgpEvaluateKernel과 동일.
__global__ void rgpHaKernel
(
    const KParams  kp,
    const int      nCells,
    const double* __restrict__ pF,
    const double* __restrict__ TF,
    const YView    yv,
    const TabView  tv,
    double* __restrict__ haF
)
{
    const int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    const int n = kp.n;
    const double RR = kp.RR;

    // Y를 한 번만 글로벌에서 읽어 로컬로 (레이아웃 무관, 값 동일)
    double Yc[RGP_GPU_MAX_SPECIES];
    if (yv.tab)   // SoA-생략: 셀 스텐실 재유도로 테이블에서 직접 보간
    {
        rgpYFillFromTab(yv, n, celli, Yc);
    }
    else
    for (int i = 0; i < n; i++) { Yc[i] = rgpYAt(yv, n, celli, i); }

    double sumY = 0.0, sumYoW = 0.0;
    for (int i = 0; i < n; i++)
    {
        sumY += Yc[i];
        sumYoW += Yc[i]/kp.W[i];
    }
    const double Wmix = (sumYoW != 0.0) ? sumY/sumYoW : kp.W[0];

    double bM, coef1, coef2, coef3;
    if (tv.map)
    {
        // Tier-2: 매니폴드 보간 계수 직접 소비 (CPU lookup 1:1)
        bM    = rgpCoeffAt(tv, 0, celli);
        coef1 = rgpCoeffAt(tv, 1, celli);
        coef2 = rgpCoeffAt(tv, 2, celli);
        coef3 = rgpCoeffAt(tv, 3, celli);
    }
    else
    {
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

    // ── EOS 쌍 혼합 (ha에는 bM, coef1..3만 필요) ──────────────────────
    bM = 0; coef1 = 0; coef2 = 0; coef3 = 0;
    for (int i = 0; i < n; i++)
    {
        bM += X[i]*kp.BM[i];
        const double Xi2 = X[i]*X[i];
        const int ii = i*n + i;
        coef1 += Xi2*kp.C1[ii];
        coef2 += Xi2*kp.C2[ii];
        coef3 += Xi2*kp.C3[ii];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            const double txx = 2.0*X[i]*X[j];
            const int ij = i*n + j;
            coef1 += txx*kp.C1[ij];
            coef2 += txx*kp.C2[ij];
            coef3 += txx*kp.C3[ij];
        }
    }
    }

    const double p = pF[celli];
    const double T = TF[celli];

    // ── JANAF 질량기준 계수 블렌드 + janafThermo::ha 적분식 ───────────
    double aJ[7];
    {
        const double* tab = (T < kp.Tcommon) ? kp.jL : kp.jH;
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

    const double haJ =
        ((((aJ[4]/5.0*T + aJ[3]/4.0)*T + aJ[2]/3.0)*T + aJ[1]/2.0)*T
       + aJ[0])*T
      + aJ[5];

    // ── SRKGas::h — EOS departure (CPU와 동일: 진입 p-플로어(34e2c5c) +
    //    기존 B floor/분기 유지; srkZ는 내부 플로어) ──
    const double phf = fmax(p, 1e4);
    double B = bM*phf/(RR*T);
    if (B <= 0.0) { B = 1e-16; }
    const double Zv = srkZ(phf, T, bM, coef1, coef2, coef3, RR, kp.stableRoot);

    double hDep;
    if (B == -Zv)
    {
        hDep = (RR*T*(Zv - 1.0))/Wmix;
    }
    else
    {
        const double sqrtT = sqrt(T);
        const double aAlpha = coef1 - coef2*sqrtT + coef3*T;
        const double daAlpha = -coef2/(2.0*sqrtT) + coef3;
        hDep =
        (
            RR*T*(Zv - 1.0)
          + (T*daAlpha - aAlpha)/bM*log((Zv + B)/Zv)
        )/Wmix;
    }

    haF[celli] = haJ + hDep;
}


//- 셀 하나의 rho/mu/kappa (CPU calcMixture → mu/kappa 경로 1:1)
__global__ void rgpEvaluateKernel
(
    const KParams  kp,
    const int      nCells,
    const double* __restrict__ pF,
    const double* __restrict__ TF,
    const YView    yv,
    const TabView  tv,
    double* __restrict__ rhoF,
    double* __restrict__ muF,
    double* __restrict__ kappaF,
    double* __restrict__ CpF,
    double* __restrict__ CvF,
    double* __restrict__ psiF
)
{
    const int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    const int n = kp.n;
    const double RR = kp.RR;

    double Yc[RGP_GPU_MAX_SPECIES];
    if (yv.tab)   // SoA-생략: 셀 스텐실 재유도로 테이블에서 직접 보간
    {
        rgpYFillFromTab(yv, n, celli, Yc);
    }
    else
    for (int i = 0; i < n; i++) { Yc[i] = rgpYAt(yv, n, celli, i); }

    // ── 믹스처 몰질량 (specie 질량 블렌드: Wmix = ΣY / Σ(Y/W)) ─────────
    double sumY = 0.0, sumYoW = 0.0;
    for (int i = 0; i < n; i++)
    {
        sumY += Yc[i];
        sumYoW += Yc[i]/kp.W[i];
    }
    const double Wmix = (sumYoW != 0.0) ? sumY/sumYoW : kp.W[0];
    const double Rmass = RR/Wmix;

    double bM, coef1, coef2, coef3, cM;
    double sigmaM, epsilonkM, VcM, TcM, omegaM, MM, miuiM, kappaiM;
    double sumXmd;

    if (tv.map)
    {
        // ── Tier-2: 매니폴드 보간 계수 직접 소비 (CPU lookup 1:1;
        //    O(n^2) 쌍 혼합·유도 변환 전부 생략, floor도 미적용 —
        //    테이블 노드값이 이미 calculateRealGas 산출물) ──
        bM        = rgpCoeffAt(tv,  0, celli);
        coef1     = rgpCoeffAt(tv,  1, celli);
        coef2     = rgpCoeffAt(tv,  2, celli);
        coef3     = rgpCoeffAt(tv,  3, celli);
        cM        = rgpCoeffAt(tv,  4, celli);
        sigmaM    = rgpCoeffAt(tv,  5, celli);
        epsilonkM = rgpCoeffAt(tv,  6, celli);
        MM        = rgpCoeffAt(tv,  7, celli);
        VcM       = rgpCoeffAt(tv,  8, celli);
        TcM       = rgpCoeffAt(tv,  9, celli);
        omegaM    = rgpCoeffAt(tv, 10, celli);
        miuiM     = rgpCoeffAt(tv, 11, celli);
        kappaiM   = rgpCoeffAt(tv, 12, celli);
        (void)sigmaM;
        // updateTRANS의 Xmd 게이트: 정규화 X라 항상 1 (CPU 동일)
        sumXmd = 1.0;
    }
    else
    {
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

    // ── calculateRealGas: 쌍 혼합 (대각 + 2×상삼각) ────────────────────
    bM = 0; coef1 = 0; coef2 = 0; coef3 = 0; cM = 0;
    double sigma3M = 0, epsilonkM0 = 0, omegaM0 = 0, MM0v = 0, miuiM0 = 0;
    kappaiM = 0;

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
    sigmaM = cbrt(sigma3M);

    epsilonkM = epsilonkM0/sigma3M;
    if (epsilonkM == 0.0) { epsilonkM = 1e-30; }

    const double sigmaOverConst = sigmaM/0.809;
    VcM = sigmaOverConst*sigmaOverConst*sigmaOverConst;
    if (VcM == 0.0) { VcM = 1e-30; }

    TcM = 1.2593*epsilonkM;
    omegaM = omegaM0/sigma3M;

    const double MM0_term = MM0v/(epsilonkM*sigmaM*sigmaM);
    MM = MM0_term*MM0_term;
    if (MM == 0.0) { MM = 1e-30; }

    miuiM = sqrt(sqrt(miuiM0*sigma3M*epsilonkM));

    // ── X 보정 (calcMixture: updateTRANS 직전) ────────────────────────
    double sumXcorr = 0.0;
    for (int i = 0; i < n; i++) { X[i] += 1e-40; sumXcorr += X[i]; }
    sumXmd = 0.0;
    for (int i = 0; i < n; i++) { X[i] /= sumXcorr; sumXmd += X[i]; }
    }

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

    // ── rho (SRK + Peneloux) / psi (FD) ──────────────────────────────
    const double rho =
        srkRho(p, T, bM, coef1, coef2, coef3, cM, Wmix, RR, kp.stableRoot);
    rhoF[celli] = rho;
    psiF[celli] =
        srkPsi(p, T, bM, coef1, coef2, coef3, cM, Wmix, RR, kp.stableRoot);

    // ── Cp / Cv (실제 thermo 체인: JANAF 블렌드 + SRK departure, raw p) ─
    {
        const double CpJanafT =
            ((((aJ[4]*T + aJ[3])*T + aJ[2])*T + aJ[1])*T + aJ[0]);
        double cpDep, cpMCv;
        srkCpDepCpMCv
        (
            p, T, bM, coef1, coef2, coef3, Wmix, RR, kp.stableRoot,
            cpDep, cpMCv
        );
        const double CpVal = CpJanafT + cpDep;
        CpF[celli] = CpVal;
        CvF[celli] = CpVal - cpMCv;          // HtoEthermo.H: Cv = Cp - CpMCv
    }

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
    // Y-CAP (upstream 41b3300 CPU 미러): packing 특이점(Y=1) 부호반전
    // 가드 — off-매니폴드 상태에서만 발동. CPU min(Y,0.9) 1:1
    const double Ych = fmin(rhoMolL*VcM/6.0, 0.9);
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
    // 양수 floor/ceiling (upstream 41b3300 CPU 미러 1:1)
    muF[celli] = fmin(fmax(muOut, 1e-7), 0.05);

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
        double cpDepRef, cpMCvUnused;
        srkCpDepCpMCv
        (
            pRef, T, bM, coef1, coef2, coef3, Wmix, RR, kp.stableRoot,
            cpDepRef, cpMCvUnused
        );
        const double CpRef = CpJanaf + cpDepRef;

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
    // 양수 floor/ceiling (upstream 41b3300 CPU 미러 1:1)
    kappaF[celli] = fmin(fmax(kappaOut, 1e-4), 1.0);
}

} // namespace rgp


// * * * * * * * * * * * * * * * C ABI 런처 * * * * * * * * * * * * * * * * //

extern "C"
{

static void fillKParams(rgp::KParams& kp);
static int chainPrepare
(
    int nCells, const double* p,
    const int* yMap, int nYHost, const double* yHost,
    const int* coeffMap,
    const double** dPOut,
    const double** dTOut, rgp::YView* yvOut, rgp::TabView* tvOut
);

int rgpGpuInit(int deviceId)
{
    const int dev = deviceId < 0 ? 0 : deviceId;
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) return fail(e, "rgpGpuInit/cudaSetDevice");

    // 비동기 UEqn 오버랩용: 동기 대기를 spin(코어 점유) 대신 blocking으로
    // — 워커 스레드의 per-iteration D2H 폴링이 CPU를 잡아먹어 메인의
    // tp pre가 느려지는 경합 제거(21.4→23.2s 역전 실측). 컨텍스트 활성
    // 전에만 유효; 이미 활성이면 에러만 소거하고 무해.
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    cudaGetLastError();

    // 기기 인식: native zero-copy는 "진짜 하드웨어 코히런트"에서만 자동.
    //   pma    = cudaDevAttrPageableMemoryAccess — pageable 호스트 메모리
    //            접근 가능(HMM 소프트웨어 마이그레이션 포함).
    //   pmaHPT = ...UsesHostPageTables — GPU가 호스트 페이지테이블을
    //            직접 쓰는 진짜 하드웨어 코히런스(GH200 NVLink-C2C/ATS).
    // pmaHPT=1: 페이지폴트 없이 호스트 포인터 직접 접근 → native가 빠름.
    // pma=1 & pmaHPT=0: HMM(x86+최신 드라이버) — 정합은 되나 PCIe
    //            페이지폴트로 느림 → 자동은 copy 유지, native는 opt-in만.
    // env RGP_GPU_UNIFIED = 0(강제 copy) / 1(mapped 검증) / 2(강제 native)
    int pma = 0, pmaHPT = 0;
    cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, dev);
    cudaDeviceGetAttribute
    (
        &pmaHPT, cudaDevAttrPageableMemoryAccessUsesHostPageTables, dev
    );
    gRgpCoherentHW = pmaHPT;   // 진단용 (호스트가 Info 출력)
    const char* env = getenv("RGP_GPU_UNIFIED");
    if (env && env[0] == '0')      { gRgpUnified = 0; }
    else if (env && env[0] == '2')
    {
        // 강제 native: 일관 메모리 미지원 기기에서는 첫 커널이
        // pageable 포인터 역참조로 즉사 — mapped 검증 모드로 강등
        if (!pma)
        {
            fprintf(stderr, "rgpGpuInit: RGP_GPU_UNIFIED=2 requested "
                    "but device lacks pageable-memory access -- "
                    "downgrading to mapped mode (1)\n");
            gRgpUnified = 1;
        }
        else { gRgpUnified = 2; }
    }
    else if (env && env[0] == '1') { gRgpUnified = 1; }
    else                           { gRgpUnified = pmaHPT ? 2 : 0; }

    return 0;
}


//- 진단: 감지된 하드웨어 코히런스 여부 (1 = NVLink-C2C/ATS류 진짜
//  코히런트, 0 = discrete/HMM). rgpGpuUnifiedMode와 함께 GH200
//  브링업 로그에 쓴다.
int rgpGpuCoherentHW(void) { return gRgpCoherentHW; }


int rgpGpuUnifiedMode(void)
{
    return gRgpUnified;
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
    // 종수 변경 재업로드 대비: Y 스테이징(capStage×구 n)이 새 n보다
    // 작게 남아 오버플로하는 잠복 결함 차단 — 버퍼도 함께 리셋
    freeBuffers();

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
    double* kappa,
    double* Cp,
    double* Cv,
    double* psi
)
{
    if (nCells <= 0) return 0;
    if (gTab.nSpecies <= 0)
    {
        snprintf(gErr, sizeof(gErr), "rgpGpuEvaluate: tables not uploaded");
        return -1;
    }

    const int n = gTab.nSpecies;

    // 스테이징 블록 캡 (native는 nCells 그대로 — 스테이징 자체가 없음)
    const int B = stageBlockCells(nCells);
    if (gRgpUnified != 2)
    {
        if (ensureOut(B)) return -1;
        if (ensureStage(B)) return -1;
    }

    rgp::KParams kp;
    fillKParams(kp);

    constexpr int blockSize = 128;

    for (int off = 0; off < nCells; off += B)
    {
        const int count =
            (nCells - off < B) ? (nCells - off) : B;

        cudaError_t e;
        const double* dP = rgpInPtr(p + off, gBuf.p, (size_t)count, &e);
        if (e != cudaSuccess) return fail(e, "evaluate/in p");
        const double* dT = rgpInPtr(T + off, gBuf.T, (size_t)count, &e);
        if (e != cudaSuccess) return fail(e, "evaluate/in T");
        const double* dY = rgpInPtr
        (
            Y + (size_t)off*n, gBuf.Y, (size_t)count*n, &e
        );
        if (e != cudaSuccess) return fail(e, "evaluate/in Y");

        double* oRho = rgpOutPtr(rho + off, gBuf.rho);
        double* oMu = rgpOutPtr(mu + off, gBuf.mu);
        double* oKappa = rgpOutPtr(kappa + off, gBuf.kappa);
        double* oCp = rgpOutPtr(Cp + off, gBuf.Cp);
        double* oCv = rgpOutPtr(Cv + off, gBuf.Cv);
        double* oPsi = rgpOutPtr(psi + off, gBuf.psi);

        const rgp::YView yv{dY, nullptr, nullptr, nullptr, 0};
        const rgp::TabView tvN{nullptr, nullptr, 0};

        const int gridSize = (count + blockSize - 1)/blockSize;
        rgp::rgpEvaluateKernel<<<gridSize, blockSize>>>
        (
            kp, count, dP, dT, yv, tvN,
            oRho, oMu, oKappa, oCp, oCv, oPsi
        );
        if ((e = cudaGetLastError()) != cudaSuccess)
            return fail(e, "evaluate/launch");

        struct { double* h; double* o; double* d; } outs[] =
        {
            {rho + off, oRho, gBuf.rho}, {mu + off, oMu, gBuf.mu},
            {kappa + off, oKappa, gBuf.kappa}, {Cp + off, oCp, gBuf.Cp},
            {Cv + off, oCv, gBuf.Cv}, {psi + off, oPsi, gBuf.psi}
        };
        for (auto& o : outs)
        {
            if ((e = rgpOutFinish(o.h, o.o, o.d, (size_t)count))
                != cudaSuccess) return fail(e, "evaluate/out");
        }
    }

    return 0;
}


//- fgm 디바이스 체인 공통부: p/yMap/yHost 업로드 + 출력 용량 확보.
//  T는 fgm SoA의 field 1을 포인터로 직접, Y는 YView(SoA 모드)로
//  커널이 직접 소비 — AoS 사본(nc×n×8B)을 만들지 않는다. 0=성공.
static int chainPrepare
(
    int nCells, const double* p,
    const int* yMap, int nYHost, const double* yHost,
    const int* coeffMap,
    const double** dPOut,
    const double** dTOut, rgp::YView* yvOut, rgp::TabView* tvOut
)
{
    const int n = gTab.nSpecies;

    const double* fgmOut = rgpFgmDevOutPtr();
    if (!fgmOut || rgpFgmDevLastN() != nCells || rgpFgmDevNFields() < 2)
    {
        snprintf(gErr, sizeof(gErr),
                 "chain: fgm device output unavailable or size mismatch "
                 "(lastN %d, want %d)", rgpFgmDevLastN(), nCells);
        return -1;
    }

    if (ensureOut(nCells)) return -1;

    cudaError_t e;
    if (!gBuf.yMap)
    {
        if ((e = cudaMalloc(&gBuf.yMap, RGP_GPU_MAX_SPECIES*sizeof(int)))
            != cudaSuccess) return fail(e, "chain/malloc yMap");
    }

    // p 입력: native 코히런트면 호스트 포인터 그대로(zero-copy),
    // 아니면 gBuf.p로 스테이징 (rgpInPtr 규약)
    const double* dP = rgpInPtr(p, gBuf.p, (size_t)nCells, &e);
    if (e != cudaSuccess) return fail(e, "chain/H2D p");
    *dPOut = dP;
    if ((e = cudaMemcpy(gBuf.yMap, yMap, n*sizeof(int),
                        cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "chain/H2D yMap");

    if (nYHost > 0)
    {
        const size_t need = (size_t)nYHost*nCells;
        if (gBuf.yHCap < need)
        {
            if (gBuf.yH) { cudaFree(gBuf.yH); gBuf.yH = nullptr; }
            gBuf.yHCap = 0;
            if ((e = cudaMalloc(&gBuf.yH, need*sizeof(double)))
                != cudaSuccess) return fail(e, "chain/malloc yH");
            gBuf.yHCap = need;
        }
        if ((e = cudaMemcpy(gBuf.yH, yHost, need*sizeof(double),
                            cudaMemcpyHostToDevice))
            != cudaSuccess) return fail(e, "chain/H2D yH");
    }

    // Tier-2 계수 맵 (nullable)
    tvOut->soa = fgmOut;
    tvOut->map = nullptr;
    tvOut->stride = (size_t)nCells;
    if (coeffMap)
    {
        if (!gBuf.cMap)
        {
            if ((e = cudaMalloc(&gBuf.cMap, 13*sizeof(int)))
                != cudaSuccess) return fail(e, "chain/malloc cMap");
        }
        if ((e = cudaMemcpy(gBuf.cMap, coeffMap, 13*sizeof(int),
                            cudaMemcpyHostToDevice))
            != cudaSuccess) return fail(e, "chain/H2D cMap");
        tvOut->map = gBuf.cMap;
    }

    *dTOut = fgmOut + (size_t)nCells;   // field 1 = T (컴팩트에서도 슬롯 1)
    yvOut->aos = nullptr;
    yvOut->soa = fgmOut;
    yvOut->map = gBuf.yMap;
    yvOut->yh = gBuf.yH;
    yvOut->stride = (size_t)nCells;

    // SoA-생략(VRAM 다이어트): Y는 SoA에 없음 — 스텐실 뷰로 테이블
    // 직접 보간 모드 활성 (map의 값은 그대로 테이블 필드 인덱스)
    if (rgpFgmSoAOmitted() > 0)
    {
        rgpFgmStencilView sv;
        if (rgpFgmGetStencilView(&sv))
        {
            snprintf(gErr, sizeof(gErr),
                     "chain: stencil view unavailable under SoA omit");
            return -1;
        }
        yvOut->soa = nullptr;
        yvOut->tab = sv.tables;
        yvOut->tabNF = sv.nFields;
        yvOut->nZ = sv.nZ; yvOut->nG = sv.nG;
        yvOut->nC = sv.nC; yvOut->nK = sv.nK;
        yvOut->mode4 = sv.mode4;
        yvOut->chi0 = sv.chi0; yvOut->hOx = sv.hOx;
        yvOut->hFuel = sv.hFuel;
        yvOut->Wlo = sv.Wlo; yvOut->Whi = sv.Whi;
        yvOut->Zax = sv.Zax; yvOut->Gax = sv.Gax;
        yvOut->Cax = sv.Cax; yvOut->Kax = sv.Kax;
        yvOut->Zf = sv.Z; yvOut->Cf = sv.C;
        yvOut->gZf = sv.gZ; yvOut->chif = sv.chi;
        yvOut->hwf = sv.hw;
    }

    return 0;
}


static void fillKParams(rgp::KParams& kp)
{
    kp.n = gTab.nSpecies;
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
}


int rgpGpuEvaluateFromFgm
(
    int nCells,
    const double* p,
    const int* yMap,
    int nYHost,
    const double* yHost,
    const int* coeffMap,
    double* rho,
    double* mu,
    double* kappa,
    double* Cp,
    double* Cv,
    double* psi
)
{
    if (nCells <= 0) return 0;
    if (gTab.nSpecies <= 0)
    {
        snprintf(gErr, sizeof(gErr),
                 "rgpGpuEvaluateFromFgm: tables not uploaded");
        return -1;
    }

    const double* dP = nullptr;
    const double* dT = nullptr;
    rgp::YView yv{};
    rgp::TabView tvC{};
    int rc = chainPrepare
    (
        nCells, p, yMap, nYHost, yHost, coeffMap, &dP, &dT, &yv, &tvC
    );
    if (rc) return rc;

    rgp::KParams kp;
    fillKParams(kp);

    constexpr int blockSize = 128;
    const int gridSize = (nCells + blockSize - 1)/blockSize;

    // 출력: native면 커널이 호스트 필드에 직접 씀(zero-copy), 아니면
    // gBuf로 스테이징 후 D2H (rgpOutPtr/rgpOutFinish 규약)
    double* oRho = rgpOutPtr(rho, gBuf.rho);
    double* oMu = rgpOutPtr(mu, gBuf.mu);
    double* oKappa = rgpOutPtr(kappa, gBuf.kappa);
    double* oCp = rgpOutPtr(Cp, gBuf.Cp);
    double* oCv = rgpOutPtr(Cv, gBuf.Cv);
    double* oPsi = rgpOutPtr(psi, gBuf.psi);

    rgp::rgpEvaluateKernel<<<gridSize, blockSize>>>
    (
        kp, nCells, dP, dT, yv, tvC,
        oRho, oMu, oKappa, oCp, oCv, oPsi
    );
    cudaError_t e;
    if ((e = cudaGetLastError()) != cudaSuccess)
        return fail(e, "evaluateFromFgm/launch");

    const size_t n1 = (size_t)nCells;
    struct { double* h; double* o; double* d; } outs[] =
    {
        {rho, oRho, gBuf.rho}, {mu, oMu, gBuf.mu},
        {kappa, oKappa, gBuf.kappa}, {Cp, oCp, gBuf.Cp},
        {Cv, oCv, gBuf.Cv}, {psi, oPsi, gBuf.psi}
    };
    for (auto& o : outs)
    {
        if ((e = rgpOutFinish(o.h, o.o, o.d, n1)) != cudaSuccess)
            return fail(e, "evaluateFromFgm/D2H");
    }

    return 0;
}


int rgpGpuEvaluateHaFromFgm
(
    int nCells,
    const double* p,
    const int* yMap,
    int nYHost,
    const double* yHost,
    const int* coeffMap,
    double* ha
)
{
    if (nCells <= 0) return 0;
    if (gTab.nSpecies <= 0)
    {
        snprintf(gErr, sizeof(gErr),
                 "rgpGpuEvaluateHaFromFgm: tables not uploaded");
        return -1;
    }

    const double* dP = nullptr;
    const double* dT = nullptr;
    rgp::YView yv{};
    rgp::TabView tvC{};
    int rc = chainPrepare
    (
        nCells, p, yMap, nYHost, yHost, coeffMap, &dP, &dT, &yv, &tvC
    );
    if (rc) return rc;

    rgp::KParams kp;
    fillKParams(kp);

    constexpr int blockSize = 128;
    const int gridSize = (nCells + blockSize - 1)/blockSize;

    // 출력: native면 호스트 ha에 직접(zero-copy), 아니면 rho 디바이스
    // 버퍼 스테이징 후 D2H (Cp는 직후 refresh 체인이 쓰므로 피함)
    double* oHa = rgpOutPtr(ha, gBuf.rho);
    rgp::rgpHaKernel<<<gridSize, blockSize>>>
    (
        kp, nCells, dP, dT, yv, tvC, oHa
    );
    cudaError_t e;
    if ((e = cudaGetLastError()) != cudaSuccess)
        return fail(e, "evaluateHaFromFgm/launch");

    if ((e = rgpOutFinish(ha, oHa, gBuf.rho, (size_t)nCells))
        != cudaSuccess)
        return fail(e, "evaluateHaFromFgm/D2H");

    return 0;
}


int rgpGpuEvaluateHa
(
    int nCells,
    const double* p,
    const double* T,
    const double* Y,
    double* ha
)
{
    if (nCells <= 0) return 0;
    if (gTab.nSpecies <= 0)
    {
        snprintf(gErr, sizeof(gErr), "rgpGpuEvaluateHa: tables not uploaded");
        return -1;
    }

    const int n = gTab.nSpecies;

    const int B = stageBlockCells(nCells);
    if (gRgpUnified != 2)
    {
        if (ensureOut(B)) return -1;
        if (ensureStage(B)) return -1;
    }

    rgp::KParams kp;
    fillKParams(kp);

    constexpr int blockSize = 128;

    for (int off = 0; off < nCells; off += B)
    {
        const int count =
            (nCells - off < B) ? (nCells - off) : B;

        cudaError_t e;
        const double* dP = rgpInPtr(p + off, gBuf.p, (size_t)count, &e);
        if (e != cudaSuccess) return fail(e, "evaluateHa/in p");
        const double* dT = rgpInPtr(T + off, gBuf.T, (size_t)count, &e);
        if (e != cudaSuccess) return fail(e, "evaluateHa/in T");
        const double* dY = rgpInPtr
        (
            Y + (size_t)off*n, gBuf.Y, (size_t)count*n, &e
        );
        if (e != cudaSuccess) return fail(e, "evaluateHa/in Y");
        // 출력은 Cp 디바이스 버퍼 재사용 (ha 전용 호출이라 충돌 없음)
        double* oHa = rgpOutPtr(ha + off, gBuf.Cp);

        const rgp::YView yv{dY, nullptr, nullptr, nullptr, 0};
        const rgp::TabView tvN{nullptr, nullptr, 0};

        const int gridSize = (count + blockSize - 1)/blockSize;
        rgp::rgpHaKernel<<<gridSize, blockSize>>>
        (
            kp, count, dP, dT, yv, tvN, oHa
        );
        if ((e = cudaGetLastError()) != cudaSuccess)
            return fail(e, "evaluateHa/launch");

        if ((e = rgpOutFinish(ha + off, oHa, gBuf.Cp, (size_t)count))
            != cudaSuccess) return fail(e, "evaluateHa/out");
    }

    return 0;
}


int rgpPinHost(void* p, size_t bytes)
{
    // Mapped(디바이스 별칭)는 unified mode 1 검증 경로에서만 필요.
    // mode 0에서 Mapped로 등록하면 WSL2에서 AmgX(자체 pinned 풀)와
    // 공존 시 이후의 일반 H2D 복사가 invalid argument로 깨지는
    // 상호작용이 실측됨(amgx+devChain+pin 조합, 2026-07-11) —
    // 전송 가속만 필요한 mode 0은 Portable 등록으로 충분하다.
    unsigned flags = cudaHostRegisterPortable;
    if (gRgpUnified == 1)
    {
        flags |= cudaHostRegisterMapped;
    }
    cudaError_t e = cudaHostRegister(p, bytes, flags);
    if (e != cudaSuccess)
    {
        snprintf(gErr, sizeof(gErr), "pinHost: %s", cudaGetErrorString(e));
        // 등록 실패의 sticky 에러 상태를 소거 — 안 하면 다음 커널 런치의
        // cudaGetLastError 체크가 이 에러를 자기 것으로 오인해 정상
        // evaluate가 FATAL로 죽는다(이중 등록 실측, 2026-07-16)
        cudaGetLastError();
        return int(e);
    }
    return 0;
}


int rgpUnpinHost(void* p)
{
    cudaError_t e = cudaHostUnregister(p);
    if (e != cudaSuccess)
    {
        snprintf(gErr, sizeof(gErr), "unpinHost: %s",
                 cudaGetErrorString(e));
        return int(e);
    }
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
