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

// * * * * * * * * * * * * * * * 디바이스 상태 * * * * * * * * * * * * * * * //

__constant__ rgpChemMech dMech;

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
        long long* steps = nullptr;
    } gBuf;

    void freeBuffers()
    {
        if (gBuf.p) cudaFree(gBuf.p);
        if (gBuf.T) cudaFree(gBuf.T);
        if (gBuf.c) cudaFree(gBuf.c);
        if (gBuf.steps) cudaFree(gBuf.steps);
        gBuf = Buffers();
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


//- 셀 하나 적분: [0, dtTot], 적응 Rosenbrock34
__global__ void chemIntegrateKernel
(
    const int nCells,
    const double dtTot,
    const double relTol,
    const double absTol,
    const double* __restrict__ pF,
    double* __restrict__ TF,
    double* __restrict__ cF,
    long long* __restrict__ stepsF
)
{
    const int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    const int n = dMech.nSpecies;
    const int neq = n + 1;
    const double p = pF[celli];

    double y[NEQ], y0[NEQ], dy0[NEQ], dy[NEQ], err[NEQ];
    double k1[NEQ], k2[NEQ], k3[NEQ], k4[NEQ];
    double J[NEQ*NEQ], LU[NEQ*NEQ];
    int piv[NEQ];

    for (int s = 0; s < n; s++) { y[s] = cF[(size_t)celli*n + s]; }
    y[n] = TF[celli];

    double t = 0.0;
    double h = dtTot;            // 초기 스텝: 전 구간 시도 (거부되면 축소)
    long long nSteps = 0;

    while (t < dtTot)
    {
        if (h > dtTot - t) { h = dtTot - t; }

        // RHS + FD Jacobian (y0 기준)
        for (int i = 0; i < neq; i++) { y0[i] = y[i]; }
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

        bool accepted = false;
        while (!accepted)
        {
            // A = I/(γh) - J
            const double diag = 1.0/(gam*h);
            for (int i = 0; i < neq*neq; i++) { LU[i] = -J[i]; }
            for (int i = 0; i < neq; i++) { LU[i*neq + i] += diag; }

            if (!luDecompose(LU, piv, neq))
            {
                h *= 0.25;
                if (h < 1e-25) { stepsF[celli] = -1; return; }
                continue;
            }

            // stage 1
            for (int i = 0; i < neq; i++) { k1[i] = dy0[i]; }
            luSolve(LU, piv, k1, neq);

            // stage 2
            for (int i = 0; i < neq; i++) { y[i] = y0[i] + a21*k1[i]; }
            chemRHS(dMech, p, y, dy);
            for (int i = 0; i < neq; i++) { k2[i] = dy[i] + c21*k1[i]/h; }
            luSolve(LU, piv, k2, neq);

            // stage 3
            for (int i = 0; i < neq; i++)
            {
                y[i] = y0[i] + a31*k1[i] + a32*k2[i];
            }
            chemRHS(dMech, p, y, dy);
            for (int i = 0; i < neq; i++)
            {
                k3[i] = dy[i] + (c31*k1[i] + c32*k2[i])/h;
            }
            luSolve(LU, piv, k3, neq);

            // stage 4 (같은 y에서 평가 — Shampine)
            for (int i = 0; i < neq; i++)
            {
                k4[i] = dy[i] + (c41*k1[i] + c42*k2[i] + c43*k3[i])/h;
            }
            luSolve(LU, piv, k4, neq);

            // 해 + 오차
            double errNorm = 0.0;
            for (int i = 0; i < neq; i++)
            {
                y[i] = y0[i] + b1*k1[i] + b2*k2[i] + b3*k3[i] + b4*k4[i];
                err[i] = e1*k1[i] + e2*k2[i] + e3v*k3[i] + e4*k4[i];
                const double sc =
                    absTol + relTol*fmax(fabs(y0[i]), fabs(y[i]));
                const double r = err[i]/sc;
                errNorm += r*r;
            }
            errNorm = sqrt(errNorm/neq);
            nSteps++;

#ifdef RGP_CHEM_DEBUG
            if (celli == 0 && nSteps <= 12)
            {
                printf("step %lld t=%.3e h=%.3e err=%.3e T=%.1f k1T=%.3e\n",
                       nSteps, t, h, errNorm, y[n], k1[n]);
            }
#endif

            if (errNorm <= 1.0 && y[n] > 0.0 && !isnan(y[n]))
            {
                accepted = true;
                t += h;
                const double f =
                    fmin(fmax(0.9*pow(errNorm, -1.0/3.0), 0.2), 5.0);
                h *= f;
            }
            else
            {
                const double f = isnan(errNorm) || isnan(y[n])
                    ? 0.25
                    : fmin(fmax(0.9*pow(errNorm, -1.0/3.0), 0.1), 0.5);
                h *= f;
                if (h < 1e-25) { stepsF[celli] = -1; return; }
                for (int i = 0; i < neq; i++) { y[i] = y0[i]; }
            }

            if (nSteps > 2000000) { stepsF[celli] = -2; return; }
        }
    }

    for (int s = 0; s < n; s++)
    {
        cF[(size_t)celli*n + s] = fmax(y[s], 0.0);
    }
    TF[celli] = y[n];
    stepsF[celli] = nSteps;
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
    cudaError_t e = cudaMemcpyToSymbol(dMech, mech, sizeof(rgpChemMech));
    if (e != cudaSuccess) return fail(e, "rgpChemUpload/memcpyToSymbol");
    return 0;
}


int rgpChemIntegrate
(
    int nCells,
    double dt,
    double relTol,
    double absTol,
    const double* p,
    double* T,
    double* c,
    long long* stats
)
{
    if (nCells <= 0) return 0;

    rgpChemMech host;
    cudaError_t e = cudaMemcpyFromSymbol(&host, dMech, sizeof(int)*2);
    // (nSpecies만 필요 — 구조체 앞머리 2 int)
    const int n = host.nSpecies;
    if (n <= 0)
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

    constexpr int blockSize = 64;
    const int gridSize = (nCells + blockSize - 1)/blockSize;
    rgpchem::chemIntegrateKernel<<<gridSize, blockSize>>>
    (
        nCells, dt, relTol, absTol, gBuf.p, gBuf.T, gBuf.c, gBuf.steps
    );
    if ((e = cudaGetLastError()) != cudaSuccess)
        return fail(e, "integrate/launch");

    if ((e = cudaMemcpy(T, gBuf.T, b1s, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "integrate/D2H T");
    if ((e = cudaMemcpy(c, gBuf.c, bns, cudaMemcpyDeviceToHost))
        != cudaSuccess) return fail(e, "integrate/D2H c");

    if (stats)
    {
        static long long* hostSteps = nullptr;
        static int hostCap = 0;
        if (nCells > hostCap)
        {
            free(hostSteps);
            hostSteps = (long long*)malloc(nCells*sizeof(long long));
            hostCap = nCells;
        }
        if ((e = cudaMemcpy(hostSteps, gBuf.steps, nCells*sizeof(long long),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return fail(e, "integrate/D2H steps");
        long long tot = 0, mx = 0;
        for (int i = 0; i < nCells; i++)
        {
            if (hostSteps[i] < 0)
            {
                snprintf(gErr, sizeof(gErr),
                         "integrate: cell %d failed (code %lld)",
                         i, hostSteps[i]);
                return -2;
            }
            tot += hostSteps[i];
            if (hostSteps[i] > mx) mx = hostSteps[i];
        }
        stats[0] = tot;
        stats[1] = mx;
    }

    return 0;
}


void rgpChemFree(void) { freeBuffers(); }

const char* rgpChemLastError(void) { return gErr; }

} // extern "C"

// ************************************************************************* //
