/*---------------------------------------------------------------------------*\
  RGP-13 chemGPU — 배치 강성 화학 적분 (셀당 1스레드, 등압 단열)

  적분기: Rosenbrock34 (Shampine 계수 — OpenFOAM-13 src/ODE와 동일) +
  FD Jacobian + 부분피벗 LU + 적응 서브스텝. 자율계이므로 d-항(∂f/∂t)=0.
  RHS는 rgpChemRHS.H — CPU 레퍼런스(OF ODESolver)와 동일 소스.

  규칙: OpenFOAM 헤더 include 금지 (thermoGPU와 동일).
\*---------------------------------------------------------------------------*/

#include "rgpChemTypes.H"
#include "rgpChemRHS.H"

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

// * * * * * * * * * * * * * * * 디바이스 상태 * * * * * * * * * * * * * * * //

// 메커니즘은 ~500KB(GRI급)라 __constant__(64KB) 대신 글로벌 메모리 상주.
static rgpChemMech* dMechPtr = nullptr;   // device
static int gNSpecies = 0;                 // host 캐시

namespace
{
    char gErr[256] = "no error";

    int fail(cudaError_t e, const char* where)
    {
        snprintf(gErr, sizeof(gErr), "%s: %s", where, cudaGetErrorString(e));
        return int(e);
    }

    struct Buffers
    {
        int capCells = 0;
        double* p = nullptr;
        double* T = nullptr;
        double* c = nullptr;
        double* dt = nullptr;
        long long* steps = nullptr;

        // warp 균형: 이전 스텝 substep 수 기준 정렬 레이아웃 (지연 할당)
        int* perm = nullptr;
        double *p2 = nullptr, *T2 = nullptr, *c2 = nullptr, *dt2 = nullptr;
        long long* steps2 = nullptr;
        int* flag2 = nullptr;

        // retrieve 캐시 (opt-in): 직전 (입력 → 출력) 쌍 (지연 할당)
        int* flag = nullptr;
        double *cp = nullptr, *cT = nullptr, *cc = nullptr, *cdt = nullptr;
        double *coT = nullptr, *coc = nullptr;
    } gBuf;

    // 균형/캐시 상태 (호스트)
    int gBalanceMode = 1;              // 0=off, 1=auto, 2=force
    double gCacheTol = 0.0;            // 0=off
    bool gCacheValid = false;
    long long gCacheHits = 0;
    std::vector<long long> gHostSteps; // 직전 호출 substep (perm 생성용)
    std::vector<int> gHostPerm;
    int gPermN = -1;                   // gHostPerm이 유효한 nCells
    bool gPermActive = false;

    void freeBuffers()
    {
        double* dp[] = {gBuf.p, gBuf.T, gBuf.c, gBuf.dt,
                        gBuf.p2, gBuf.T2, gBuf.c2, gBuf.dt2,
                        gBuf.cp, gBuf.cT, gBuf.cc, gBuf.cdt,
                        gBuf.coT, gBuf.coc};
        for (auto p : dp) { if (p) cudaFree(p); }
        if (gBuf.steps) cudaFree(gBuf.steps);
        if (gBuf.steps2) cudaFree(gBuf.steps2);
        if (gBuf.perm) cudaFree(gBuf.perm);
        if (gBuf.flag) cudaFree(gBuf.flag);
        if (gBuf.flag2) cudaFree(gBuf.flag2);
        gBuf = Buffers();
        gCacheValid = false;
        gPermN = -1;
        gPermActive = false;
    }
}


namespace rgpchem
{

// Rosenbrock34 계수 — L-안정 세트 (Hairer et al.; OpenFOAM-13
// Rosenbrock34.C에 동봉된 대안 상수). 화학 평형 상태에서 Shampine
// 세트는 안정성 한계로 h가 붕괴하므로 L-안정이 필수.
__device__ __constant__ double
    a21 = 2.0, a31 = 1.867943637803922, a32 = 0.2344449711399156,
    c21 = -7.137615036412310, c31 = 2.580708087951457,
    c32 = 0.6515950076447975,
    c41 = -2.137148994382534, c42 = -0.3214669691237626,
    c43 = -0.6949742501781779,
    b1 = 2.255570073418735, b2 = 0.2870493262186792,
    b3 = 0.435317943184018, b4 = 1.093502252409163,
    e1 = -0.2815431932141155, e2 = -0.0727619912493892,
    e3v = -0.1082196201495311, e4 = -1.093502252409163,
    gam = 0.57282;

#define NEQ RGP_CHEM_NEQ


//- 부분피벗 LU 분해 (n×n, 행우선)
__device__ bool luDecompose(double* Am, int* piv, const int n)
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


__device__ void luSolve
(
    const double* Am, const int* piv, double* b, const int n
)
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


//- retrieve 캐시 체크 (raw 레이아웃): 입력 (p,T,c,dt)가 직전 적분의
//  입력과 tol 내 동일하면 hit — 저장된 출력을 즉시 기록하고 적분 스킵.
//  miss면 현재 입력을 캐시에 저장(출력은 적분 후 chemCacheStore가).
__global__ void chemCacheCheck
(
    const int nCells, const int nsp, const double tol, const int valid,
    const double* __restrict__ pF, double* __restrict__ TF,
    double* __restrict__ cF, const double* __restrict__ dtF,
    double* __restrict__ cp, double* __restrict__ cT,
    double* __restrict__ cc, double* __restrict__ cdt,
    const double* __restrict__ coT, const double* __restrict__ coc,
    int* __restrict__ flag
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nCells) return;

    // 순수 상대 비교 (b=0이면 a==0 요구). 절대 플로어를 두면 점화
    // 유도기의 라디칼 성장(절대값은 미소하나 상대적으론 지수 성장)을
    // '미변화'로 오판해 점화가 동결된다 — 실측으로 확인된 함정.
    auto close = [&](double a, double b) -> bool
    {
        return fabs(a - b) <= tol*fabs(b);
    };

    bool hit = (valid != 0)
        && close(pF[i], cp[i]) && close(TF[i], cT[i])
        && close(dtF[i], cdt[i]);
    if (hit)
    {
        for (int s = 0; s < nsp && hit; s++)
        {
            hit = close(cF[(size_t)i*nsp + s], cc[(size_t)i*nsp + s]);
        }
    }

    if (hit)
    {
        flag[i] = 1;
        TF[i] = coT[i];
        for (int s = 0; s < nsp; s++)
        {
            cF[(size_t)i*nsp + s] = coc[(size_t)i*nsp + s];
        }
    }
    else
    {
        flag[i] = 0;
        cp[i] = pF[i]; cT[i] = TF[i]; cdt[i] = dtF[i];
        for (int s = 0; s < nsp; s++)
        {
            cc[(size_t)i*nsp + s] = cF[(size_t)i*nsp + s];
        }
    }
}

//- retrieve 캐시 출력 저장 (적분 후, miss 셀만)
__global__ void chemCacheStore
(
    const int nCells, const int nsp, const int* __restrict__ flag,
    const double* __restrict__ TF, const double* __restrict__ cF,
    double* __restrict__ coT, double* __restrict__ coc
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nCells || flag[i]) return;
    coT[i] = TF[i];
    for (int s = 0; s < nsp; s++)
    {
        coc[(size_t)i*nsp + s] = cF[(size_t)i*nsp + s];
    }
}

//- warp 균형: perm 정렬 레이아웃으로 게더 (flag는 null 허용)
__global__ void chemGather
(
    const int nCells, const int nsp, const int* __restrict__ perm,
    const double* __restrict__ p, const double* __restrict__ T,
    const double* __restrict__ c, const double* __restrict__ dt,
    const int* __restrict__ flag,
    double* __restrict__ p2, double* __restrict__ T2,
    double* __restrict__ c2, double* __restrict__ dt2,
    int* __restrict__ flag2
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nCells) return;
    const int i = perm[k];
    p2[k] = p[i]; T2[k] = T[i]; dt2[k] = dt[i];
    flag2[k] = flag ? flag[i] : 0;
    for (int s = 0; s < nsp; s++)
    {
        c2[(size_t)k*nsp + s] = c[(size_t)i*nsp + s];
    }
}

//- warp 균형: 결과 스캐터 (raw 레이아웃 복원)
__global__ void chemScatter
(
    const int nCells, const int nsp, const int* __restrict__ perm,
    const double* __restrict__ T2, const double* __restrict__ c2,
    const long long* __restrict__ steps2,
    double* __restrict__ T, double* __restrict__ c,
    long long* __restrict__ steps
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nCells) return;
    const int i = perm[k];
    T[i] = T2[k];
    steps[i] = steps2[k];
    for (int s = 0; s < nsp; s++)
    {
        c[(size_t)i*nsp + s] = c2[(size_t)k*nsp + s];
    }
}

//- 셀 하나 적분: [0, dtTot], 적응 Rosenbrock34
__global__ void chemIntegrateKernel
(
    const rgpChemMech* __restrict__ mechP,
    const int nCells,
    const double* __restrict__ dtF,
    const double relTol,
    const double absTol,
    const double* __restrict__ pF,
    double* __restrict__ TF,
    double* __restrict__ cF,
    long long* __restrict__ stepsF,
    const int* __restrict__ hitF
)
{
    const int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    // retrieve 캐시 hit — 출력은 chemCacheCheck가 이미 기록
    if (hitF && hitF[celli]) { stepsF[celli] = 0; return; }

    const rgpChemMech& dMech = *mechP;
    const int n = dMech.nSpecies;
    const int neq = n + 1;
    const double p = pF[celli];

    double y0[NEQ], y[NEQ];
    double wk[NEQ*(NEQ + 7)];

    for (int s = 0; s < n; s++) { y0[s] = cF[(size_t)celli*n + s]; }
    y0[n] = TF[celli];

    const double dtTot = dtF[celli];
    double t = 0.0;
    double h = dtTot;
    long long nSteps = 0;

    while (t < dtTot)
    {
        if (h > dtTot - t) { h = dtTot - t; }

        const double errNorm =
            rosenbrock34Step(dMech, p, h, relTol, absTol, y0, y, wk);
        nSteps++;

#ifdef RGP_CHEM_DEBUG
        if (celli == 0 && nSteps <= 12)
        {
            printf("step %lld t=%.3e h=%.3e err=%.3e T=%.1f\n",
                   nSteps, t, h, errNorm, y[n]);
        }
#endif

        if (errNorm <= 1.0 && y[n] > 0.0 && !isnan(y[n]))
        {
            t += h;
            for (int i = 0; i < neq; i++) { y0[i] = y[i]; }
            h *= fmin(fmax(0.9*pow(errNorm, -1.0/3.0), 0.2), 5.0);
        }
        else
        {
            const double f = (isnan(errNorm) || isnan(y[n]))
                ? 0.25
                : fmin(fmax(0.9*pow(errNorm, -1.0/3.0), 0.1), 0.5);
            h *= f;
            if (h < 1e-25) { stepsF[celli] = -1; return; }
        }

        if (nSteps > 2000000) { stepsF[celli] = -2; return; }
    }
    for (int i = 0; i < neq; i++) { y[i] = y0[i]; }

    for (int s = 0; s < n; s++)
    {
        cF[(size_t)celli*n + s] = fmax(y[s], 0.0);
    }
    TF[celli] = y[n];
    stepsF[celli] = nSteps;
}

//- 디버그: 단일 스레드로 RHS + FD Jacobian만 계산 (host 대조용)
__global__ void debugJacKernel
(
    const rgpChemMech* __restrict__ mechP,
    const double p,
    const double* __restrict__ yIn,
    double* __restrict__ dy0,
    double* __restrict__ J
)
{
    const rgpChemMech& dMech = *mechP;
    const int n = dMech.nSpecies;
    const int neq = n + 1;
    double y0[NEQ], dy[NEQ];
    for (int i = 0; i < neq; i++) { y0[i] = yIn[i]; }
    chemRHS(dMech, p, y0, dy0);
    for (int j = 0; j < neq; j++)
    {
        const double yj = y0[j];
        const double d = fmax(fabs(yj)*1e-7, 1e-14);
        y0[j] = yj + d;
        chemRHS(dMech, p, y0, dy);
        y0[j] = yj;
        const double dinv = 1.0/d;
        for (int i = 0; i < neq; i++)
        {
            J[i*neq + j] = (dy[i] - dy0[i])*dinv;
        }
    }
}

} // namespace rgpchem


// * * * * * * * * * * * * * * * C ABI * * * * * * * * * * * * * * * * * * * //

extern "C"
{

int rgpChemInit(int deviceId)
{
    cudaError_t e = cudaSetDevice(deviceId < 0 ? 0 : deviceId);
    if (e != cudaSuccess) return fail(e, "rgpChemInit");
    return 0;
}


int rgpChemUpload(const rgpChemMech* mech)
{
    if (!mech || mech->nSpecies <= 0 || mech->nSpecies > RGP_CHEM_MAX_SPECIES
        || mech->nReactions <= 0 || mech->nReactions > RGP_CHEM_MAX_REACTIONS)
    {
        snprintf(gErr, sizeof(gErr), "rgpChemUpload: bad mechanism dims");
        return -1;
    }
    cudaError_t e;
    if (!dMechPtr)
    {
        e = cudaMalloc(&dMechPtr, sizeof(rgpChemMech));
        if (e != cudaSuccess) return fail(e, "rgpChemUpload/malloc");
    }
    e = cudaMemcpy(dMechPtr, mech, sizeof(rgpChemMech),
                   cudaMemcpyHostToDevice);
    if (e != cudaSuccess) return fail(e, "rgpChemUpload/memcpy");
    gNSpecies = mech->nSpecies;
    return 0;
}


int rgpChemIntegrate
(
    int nCells,
    const double* dt,
    double relTol,
    double absTol,
    const double* p,
    double* T,
    double* c,
    long long* stats
)
{
    if (nCells <= 0) return 0;

    cudaError_t e;
    const int n = gNSpecies;
    if (n <= 0 || !dMechPtr)
    {
        snprintf(gErr, sizeof(gErr), "rgpChemIntegrate: mech not uploaded");
        return -1;
    }

    if (nCells > gBuf.capCells)
    {
        freeBuffers();
        if ((e = cudaMalloc(&gBuf.p, nCells*sizeof(double))) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.T, nCells*sizeof(double))) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.c, (size_t)nCells*n*sizeof(double)))
                != cudaSuccess ||
            (e = cudaMalloc(&gBuf.dt, nCells*sizeof(double))) != cudaSuccess ||
            (e = cudaMalloc(&gBuf.steps, nCells*sizeof(long long)))
                != cudaSuccess)
        {
            freeBuffers();
            return fail(e, "integrate/malloc");
        }
        gBuf.capCells = nCells;
    }

    const size_t b1s = nCells*sizeof(double);
    const size_t bns = (size_t)nCells*n*sizeof(double);
    if ((e = cudaMemcpy(gBuf.p, p, b1s, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "integrate/H2D p");
    if ((e = cudaMemcpy(gBuf.T, T, b1s, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "integrate/H2D T");
    if ((e = cudaMemcpy(gBuf.c, c, bns, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "integrate/H2D c");
    if ((e = cudaMemcpy(gBuf.dt, dt, b1s, cudaMemcpyHostToDevice))
        != cudaSuccess) return fail(e, "integrate/H2D dt");

    constexpr int blockSize = 64;
    const int gridSize = (nCells + blockSize - 1)/blockSize;

    // ── retrieve 캐시 (opt-in): 입력 미변화 셀은 저장된 출력으로 스킵 ──
    gCacheHits = 0;
    const bool cacheOn = (gCacheTol > 0.0);
    if (cacheOn)
    {
        // 지연 할당은 capCells 기준 (이후 호출이 nCells ≤ capCells로
        // 커질 수 있음); 셀 수가 바뀌면 셀-매핑이 달라지므로 무효화
        static int cacheN = -1;
        if (nCells != cacheN) { gCacheValid = false; cacheN = nCells; }

        if (!gBuf.flag)
        {
            const size_t B1 = (size_t)gBuf.capCells*sizeof(double);
            const size_t BN = (size_t)gBuf.capCells*n*sizeof(double);
            if ((e = cudaMalloc(&gBuf.flag, gBuf.capCells*sizeof(int)))
                    != cudaSuccess ||
                (e = cudaMalloc(&gBuf.cp, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.cT, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.cdt, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.coT, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.cc, BN)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.coc, BN)) != cudaSuccess)
            {
                return fail(e, "integrate/cache malloc");
            }
            gCacheValid = false;
        }
        rgpchem::chemCacheCheck<<<gridSize, blockSize>>>
        (
            nCells, n, gCacheTol, gCacheValid ? 1 : 0,
            gBuf.p, gBuf.T, gBuf.c, gBuf.dt,
            gBuf.cp, gBuf.cT, gBuf.cc, gBuf.cdt,
            gBuf.coT, gBuf.coc, gBuf.flag
        );
        if ((e = cudaGetLastError()) != cudaSuccess)
            return fail(e, "integrate/cache launch");
    }

    // ── warp 균형: 직전 호출의 substep 수로 정렬(무거운 셀 우선,
    //    로그-버킷) — 게더/스캐터는 값 보존이라 결과 비트-동일 ──
    const bool balance = gPermActive && gPermN == nCells
                      && (gBalanceMode >= 1);
    if (balance)
    {
        if (!gBuf.perm)
        {
            const size_t B1 = (size_t)gBuf.capCells*sizeof(double);
            const size_t BN = (size_t)gBuf.capCells*n*sizeof(double);
            if ((e = cudaMalloc(&gBuf.perm, gBuf.capCells*sizeof(int)))
                    != cudaSuccess ||
                (e = cudaMalloc(&gBuf.flag2, gBuf.capCells*sizeof(int)))
                    != cudaSuccess ||
                (e = cudaMalloc(&gBuf.p2, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.T2, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.dt2, B1)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.c2, BN)) != cudaSuccess ||
                (e = cudaMalloc(&gBuf.steps2,
                                gBuf.capCells*sizeof(long long)))
                    != cudaSuccess)
            {
                return fail(e, "integrate/perm malloc");
            }
        }
        if ((e = cudaMemcpy(gBuf.perm, gHostPerm.data(),
                            nCells*sizeof(int), cudaMemcpyHostToDevice))
            != cudaSuccess) return fail(e, "integrate/H2D perm");

        rgpchem::chemGather<<<gridSize, blockSize>>>
        (
            nCells, n, gBuf.perm, gBuf.p, gBuf.T, gBuf.c, gBuf.dt,
            cacheOn ? gBuf.flag : nullptr,
            gBuf.p2, gBuf.T2, gBuf.c2, gBuf.dt2, gBuf.flag2
        );
        rgpchem::chemIntegrateKernel<<<gridSize, blockSize>>>
        (
            dMechPtr, nCells, gBuf.dt2, relTol, absTol,
            gBuf.p2, gBuf.T2, gBuf.c2, gBuf.steps2,
            cacheOn ? gBuf.flag2 : nullptr
        );
        rgpchem::chemScatter<<<gridSize, blockSize>>>
        (
            nCells, n, gBuf.perm, gBuf.T2, gBuf.c2, gBuf.steps2,
            gBuf.T, gBuf.c, gBuf.steps
        );
    }
    else
    {
        rgpchem::chemIntegrateKernel<<<gridSize, blockSize>>>
        (
            dMechPtr, nCells, gBuf.dt, relTol, absTol,
            gBuf.p, gBuf.T, gBuf.c, gBuf.steps,
            cacheOn ? gBuf.flag : nullptr
        );
    }
    if ((e = cudaGetLastError()) != cudaSuccess)
        return fail(e, "integrate/launch");

    if (cacheOn)
    {
        rgpchem::chemCacheStore<<<gridSize, blockSize>>>
        (
            nCells, n, gBuf.flag, gBuf.T, gBuf.c, gBuf.coT, gBuf.coc
        );
        if ((e = cudaGetLastError()) != cudaSuccess)
            return fail(e, "integrate/cache store");
        gCacheValid = true;
    }

    if ((e = cudaMemcpy(T, gBuf.T, b1s, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "integrate/D2H T");
    if ((e = cudaMemcpy(c, gBuf.c, bns, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "integrate/D2H c");

    // ── substep 통계 회수: stats 보고 + 다음 호출의 균형 perm 생성 ──
    if ((int)gHostSteps.size() < nCells) gHostSteps.resize(nCells);
    if ((e = cudaMemcpy(gHostSteps.data(), gBuf.steps,
                        nCells*sizeof(long long), cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "integrate/D2H steps");

    long long tot = 0, mx = 0, nHit = 0;
    for (int i = 0; i < nCells; i++)
    {
        const long long si = gHostSteps[i];
        if (si < 0)
        {
            snprintf(gErr, sizeof(gErr),
                     "integrate: cell %d failed (code %lld)", i, si);
            return -2;
        }
        tot += si;
        if (si > mx) mx = si;
        if (si == 0) nHit++;
    }
    if (cacheOn) gCacheHits = nHit;
    if (stats)
    {
        stats[0] = tot;
        stats[1] = mx;
    }

    // 불균형(max > 8×mean)일 때만 다음 호출에서 정렬 발동 (mode 2=항상)
    gPermActive = false;
    if (gBalanceMode == 2
     || (gBalanceMode == 1 && nCells >= 4096
      && mx*(long long)nCells > 8*tot))
    {
        // 로그-버킷 카운팅 정렬 (내림차순) — O(n), 버킷 내 자연순 유지
        constexpr int NB = 44;
        int cnt[NB] = {0};
        auto bucketOf = [](long long s) -> int
        {
            int b = 0;
            while (s > 0 && b < NB - 1) { s >>= 1; b++; }
            return b;
        };
        for (int i = 0; i < nCells; i++)
        {
            cnt[bucketOf(gHostSteps[i])]++;
        }
        int start[NB];   // 내림차순: 큰 버킷이 앞
        {
            int acc = 0;
            for (int b = NB - 1; b >= 0; b--) { start[b] = acc; acc += cnt[b]; }
        }
        if ((int)gHostPerm.size() < nCells) gHostPerm.resize(nCells);
        for (int i = 0; i < nCells; i++)
        {
            gHostPerm[start[bucketOf(gHostSteps[i])]++] = i;
        }
        gPermN = nCells;
        gPermActive = true;
    }

    return 0;
}


int rgpChemSetBalance(int mode)
{
    gBalanceMode = (mode < 0) ? 0 : (mode > 2 ? 2 : mode);
    return 0;
}


int rgpChemSetCacheTol(double relTol)
{
    gCacheTol = (relTol > 0.0) ? relTol : 0.0;
    gCacheValid = false;
    return 0;
}


long long rgpChemCacheHits(void)
{
    return gCacheHits;
}


int rgpChemDebugJac(double p, const double* y, double* dy0, double* J)
{
    const int n = gNSpecies;
    const int neq = n + 1;

    double *dY, *dDy, *dJ;
    cudaMalloc(&dY, neq*sizeof(double));
    cudaMalloc(&dDy, neq*sizeof(double));
    cudaMalloc(&dJ, neq*neq*sizeof(double));
    cudaMemcpy(dY, y, neq*sizeof(double), cudaMemcpyHostToDevice);

    rgpchem::debugJacKernel<<<1, 1>>>(dMechPtr, p, dY, dDy, dJ);
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();

    cudaMemcpy(dy0, dDy, neq*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(J, dJ, neq*neq*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dY); cudaFree(dDy); cudaFree(dJ);

    if (e != cudaSuccess) return fail(e, "debugJac");
    return 0;
}


void rgpChemFree(void)
{
    freeBuffers();
    if (dMechPtr) { cudaFree(dMechPtr); dMechPtr = nullptr; }
    gNSpecies = 0;
}

const char* rgpChemLastError(void) { return gErr; }

} // extern "C"

// ************************************************************************* //
