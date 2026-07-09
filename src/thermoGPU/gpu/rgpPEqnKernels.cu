/*---------------------------------------------------------------------------*\
  RGP-13 pEqnGPU — 압력 방정식의 디바이스 조립 + Jacobi-PCG + 플럭스

  fvm 의미론 1:1 (fvSchemes 'Gauss linear orthogonal' 한정):
    laplacian(rAUf, p):  upper_f = +Γ_f, negSumDiag  →  pEqn(−laplacian):
    upper_f = −Γ_f, diag_c += Σ_f Γ_f.
  경계 기여는 호스트가 fvPatchField gradientInternal/BoundaryCoeffs로
  정확 계산해 내린 bDiag/bSrc를 faceCells에 산입.
  솔버는 lduMatrix PCG의 잔차 규약(normFactor, 수렴 판정)을 그대로 따르고
  전처리기만 DIC → Jacobi(대각).
  규칙: OpenFOAM 헤더 include 금지.
\*---------------------------------------------------------------------------*/

#include "rgpPEqnTypes.H"

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

namespace
{
    char gPErr[256] = "no error";

    int pfail(cudaError_t e, const char* where)
    {
        snprintf(gPErr, sizeof(gPErr), "%s: %s",
                 where, cudaGetErrorString(e));
        return int(e);
    }

    struct PMesh
    {
        int nCells = 0, nIntFaces = 0, nBFaces = 0;
        int    *own = nullptr, *nei = nullptr, *bfc = nullptr;
        double *gg = nullptr, *V = nullptr;
    } gM;

    struct PBuf
    {
        // 행렬/벡터 (모두 [nCells] 또는 [nIntFaces])
        double *diag = nullptr, *upper = nullptr, *b = nullptr;
        double *x = nullptr, *rA = nullptr, *pA = nullptr, *wA = nullptr;
        double *rowSum = nullptr;
        // 조립 입력
        double *rAUf = nullptr, *psis = nullptr, *pOld = nullptr,
               *phiInt = nullptr, *phiB = nullptr,
               *bDiag = nullptr, *bSrc = nullptr;
        double *flux = nullptr;
        double *red = nullptr;   // 리덕션 스칼라 [4]
    } gB;

    void pFreeAll()
    {
        int* ip[] = {gM.own, gM.nei, gM.bfc};
        for (auto p : ip) { if (p) cudaFree(p); }
        gM.own = gM.nei = gM.bfc = nullptr;

        double** dp[] =
        {
            &gM.gg, &gM.V,
            &gB.diag, &gB.upper, &gB.b, &gB.x, &gB.rA, &gB.pA, &gB.wA,
            &gB.rowSum, &gB.rAUf, &gB.psis, &gB.pOld, &gB.phiInt,
            &gB.phiB, &gB.bDiag, &gB.bSrc, &gB.flux, &gB.red
        };
        for (auto p : dp) { if (*p) { cudaFree(*p); *p = nullptr; } }
        gM.nCells = gM.nIntFaces = gM.nBFaces = 0;
    }
}

namespace rgppeqn
{

constexpr int BS = 128;

__global__ void cellAssemble
(
    const int n, const double dtInv,
    const double* __restrict__ psis, const double* __restrict__ V,
    const double* __restrict__ pOld,
    double* __restrict__ diag, double* __restrict__ b
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double a = psis[i]*V[i]*dtInv;
    diag[i] = a;
    b[i] = a*pOld[i];
}

__global__ void faceAssemble
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ gg, const double* __restrict__ rAUf,
    const double* __restrict__ phiInt,
    double* __restrict__ diag, double* __restrict__ upper,
    double* __restrict__ b
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const double G = rAUf[f]*gg[f];
    upper[f] = -G;
    atomicAdd(&diag[own[f]], G);
    atomicAdd(&diag[nei[f]], G);
    atomicAdd(&b[own[f]], -phiInt[f]);
    atomicAdd(&b[nei[f]],  phiInt[f]);
}

__global__ void bFaceAssemble
(
    const int nbf, const int* __restrict__ bfc,
    const double* __restrict__ bDiag, const double* __restrict__ bSrc,
    const double* __restrict__ phiB,
    double* __restrict__ diag, double* __restrict__ b
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    atomicAdd(&diag[bfc[k]], bDiag[k]);
    atomicAdd(&b[bfc[k]], bSrc[k] - phiB[k]);
}

__global__ void setRef
(
    const int refCell, const double refValue,
    double* __restrict__ diag, double* __restrict__ b
)
{
    // OF fvMatrix::setReference: source += diag*value; diag += diag
    b[refCell] += diag[refCell]*refValue;
    diag[refCell] += diag[refCell];
}

//- y = diag*x (셀) — 면 기여는 faceSpmv가 누적
__global__ void diagSpmv
(
    const int n, const double* __restrict__ diag,
    const double* __restrict__ x, double* __restrict__ y
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = diag[i]*x[i];
}

__global__ void faceSpmv
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ upper,
    const double* __restrict__ x, double* __restrict__ y
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    atomicAdd(&y[own[f]], upper[f]*x[nei[f]]);
    atomicAdd(&y[nei[f]], upper[f]*x[own[f]]);
}

__global__ void faceRowSum
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ upper, double* __restrict__ rs
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    atomicAdd(&rs[own[f]], upper[f]);
    atomicAdd(&rs[nei[f]], upper[f]);
}

//- 리덕션: mode 0=Σa, 1=Σ|a|, 2=Σ a*b, 3=Σ|a-c*rs|(normFactor용)
__global__ void reduceK
(
    const int n, const int mode,
    const double* __restrict__ a, const double* __restrict__ bb,
    const double c, const double* __restrict__ rs,
    double* __restrict__ out
)
{
    __shared__ double sh[BS];
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    double v = 0.0;
    if (i < n)
    {
        if      (mode == 0) { v = a[i]; }
        else if (mode == 1) { v = fabs(a[i]); }
        else if (mode == 2) { v = a[i]*bb[i]; }
        else                { v = fabs(a[i] - c*rs[i]); }
    }
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) { sh[threadIdx.x] += sh[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(out, sh[0]); }
}

//- r = b - y;  (조립 직후 초기 잔차)
__global__ void residualInit
(
    const int n, const double* __restrict__ b,
    const double* __restrict__ y, double* __restrict__ r
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    r[i] = b[i] - y[i];
}

//- Jacobi 전처리 + 방향 갱신: pA = r/diag + beta*pA (beta=0이면 초기화)
__global__ void precondDir
(
    const int n, const double* __restrict__ r,
    const double* __restrict__ diag, const double beta,
    double* __restrict__ pA
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double w = r[i]/diag[i];
    pA[i] = (beta == 0.0) ? w : w + beta*pA[i];
}

//- x += alpha*pA; r -= alpha*wA
__global__ void cgUpdate
(
    const int n, const double alpha,
    const double* __restrict__ pA, const double* __restrict__ wA,
    double* __restrict__ x, double* __restrict__ r
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += alpha*pA[i];
    r[i] -= alpha*wA[i];
}

//- 내부면 플럭스: faceH = upper*x_n - lower*x_o = -upper*(x_o - x_n)...
//  대칭(lower=upper)이므로 faceH_f = upper_f*(x_n - x_o)
__global__ void faceFlux
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ upper, const double* __restrict__ x,
    double* __restrict__ flux
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    flux[f] = upper[f]*(x[nei[f]] - x[own[f]]);
}

} // namespace rgppeqn


// * * * * * * * * * * * * * * * 호스트 헬퍼 * * * * * * * * * * * * * * * * //

namespace
{
    constexpr int BS = rgppeqn::BS;
    inline int nb(int n) { return (n + BS - 1)/BS; }

    //- 디바이스 리덕션 실행 → 호스트 스칼라
    int reduceHost
    (
        int n, int mode, const double* a, const double* b,
        double c, const double* rs, double& out
    )
    {
        cudaError_t e;
        if ((e = cudaMemset(gB.red, 0, sizeof(double))) != cudaSuccess)
            return pfail(e, "peqn/red memset");
        rgppeqn::reduceK<<<nb(n), BS>>>(n, mode, a, b, c, rs, gB.red);
        if ((e = cudaGetLastError()) != cudaSuccess)
            return pfail(e, "peqn/red launch");
        if ((e = cudaMemcpy(&out, gB.red, sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "peqn/red D2H");
        return 0;
    }

    //- 조립 공통부 (업로드 + diag/upper/b 완성). 0=성공.
    int assemble
    (
        double dtInv,
        const double* rAUfInt, const double* psis, const double* pOld,
        const double* phiInt, const double* phiB,
        const double* bDiag, const double* bSrc,
        int needRef, int refCell, double refValue
    )
    {
        const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
        if (nc <= 0)
        {
            snprintf(gPErr, sizeof(gPErr), "peqn: mesh not uploaded");
            return -1;
        }

        cudaError_t e;
        struct { double* d; const double* s; size_t n; } up[] =
        {
            {gB.rAUf, rAUfInt, (size_t)nf}, {gB.psis, psis, (size_t)nc},
            {gB.pOld, pOld, (size_t)nc}, {gB.phiInt, phiInt, (size_t)nf},
            {gB.phiB, phiB, (size_t)nbf}, {gB.bDiag, bDiag, (size_t)nbf},
            {gB.bSrc, bSrc, (size_t)nbf}
        };
        for (auto& u : up)
        {
            if (u.n == 0) continue;
            if ((e = cudaMemcpy(u.d, u.s, u.n*sizeof(double),
                                cudaMemcpyHostToDevice)) != cudaSuccess)
                return pfail(e, "peqn/H2D");
        }

        rgppeqn::cellAssemble<<<nb(nc), BS>>>
            (nc, dtInv, gB.psis, gM.V, gB.pOld, gB.diag, gB.b);
        rgppeqn::faceAssemble<<<nb(nf), BS>>>
            (nf, gM.own, gM.nei, gM.gg, gB.rAUf, gB.phiInt,
             gB.diag, gB.upper, gB.b);
        if (nbf > 0)
        {
            rgppeqn::bFaceAssemble<<<nb(nbf), BS>>>
                (nbf, gM.bfc, gB.bDiag, gB.bSrc, gB.phiB, gB.diag, gB.b);
        }
        if (needRef)
        {
            rgppeqn::setRef<<<1, 1>>>(refCell, refValue, gB.diag, gB.b);
        }
        if ((e = cudaGetLastError()) != cudaSuccess)
            return pfail(e, "peqn/assemble launch");
        return 0;
    }
}


// * * * * * * * * * * * * * * * C ABI * * * * * * * * * * * * * * * * * * * //

extern "C"
{

int rgpPEqnMeshUpload
(
    int nCells, int nIntFaces,
    const int* owner, const int* neigh,
    const double* gammaGeom,
    const double* V,
    int nBFaces, const int* bFaceCells
)
{
    pFreeAll();
    gM.nCells = nCells; gM.nIntFaces = nIntFaces; gM.nBFaces = nBFaces;

    cudaError_t e;
    struct { void** d; const void* s; size_t bytes; } it[] =
    {
        {(void**)&gM.own, owner, (size_t)nIntFaces*sizeof(int)},
        {(void**)&gM.nei, neigh, (size_t)nIntFaces*sizeof(int)},
        {(void**)&gM.gg,  gammaGeom, (size_t)nIntFaces*sizeof(double)},
        {(void**)&gM.V,   V, (size_t)nCells*sizeof(double)},
        {(void**)&gM.bfc, bFaceCells, (size_t)nBFaces*sizeof(int)}
    };
    for (auto& i : it)
    {
        if (i.bytes == 0) continue;
        if ((e = cudaMalloc(i.d, i.bytes)) != cudaSuccess)
        { pFreeAll(); return pfail(e, "peqn/mesh malloc"); }
        if ((e = cudaMemcpy(*i.d, i.s, i.bytes, cudaMemcpyHostToDevice))
            != cudaSuccess)
        { pFreeAll(); return pfail(e, "peqn/mesh H2D"); }
    }

    const size_t bc = (size_t)nCells*sizeof(double);
    const size_t bf = (size_t)nIntFaces*sizeof(double);
    const size_t bb = (size_t)(nBFaces > 0 ? nBFaces : 1)*sizeof(double);
    struct { double** d; size_t bytes; } al[] =
    {
        {&gB.diag, bc}, {&gB.b, bc}, {&gB.x, bc}, {&gB.rA, bc},
        {&gB.pA, bc}, {&gB.wA, bc}, {&gB.rowSum, bc}, {&gB.psis, bc},
        {&gB.pOld, bc},
        {&gB.upper, bf}, {&gB.rAUf, bf}, {&gB.phiInt, bf}, {&gB.flux, bf},
        {&gB.phiB, bb}, {&gB.bDiag, bb}, {&gB.bSrc, bb},
        {&gB.red, 4*sizeof(double)}
    };
    for (auto& a : al)
    {
        if ((e = cudaMalloc(a.d, a.bytes)) != cudaSuccess)
        { pFreeAll(); return pfail(e, "peqn/buf malloc"); }
    }
    return 0;
}


int rgpPEqnSolve
(
    double dtInv,
    const double* rAUfInt,
    const double* psis, const double* pOld, const double* p0,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double tol, double relTol, int maxIter,
    double* pOut, double* fluxInt,
    double* initRes, double* finalRes, int* nIters
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;

    int rc = assemble(dtInv, rAUfInt, psis, pOld, phiInt, phiB,
                      bDiag, bSrc, needRef, refCell, refValue);
    if (rc) return rc;

    cudaError_t e;
    if ((e = cudaMemcpy(gB.x, p0, (size_t)nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "peqn/H2D x0");

    // rowSum (normFactor용): rs = diag + Σ upper 기여
    rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, gB.diag, gB.x, gB.wA);
    // rowSum은 diag 복사 후 면 누적 — diagSpmv를 x=1로 쓰는 대신 직접
    if ((e = cudaMemcpy(gB.rowSum, gB.diag, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToDevice)) != cudaSuccess)
        return pfail(e, "peqn/rowSum D2D");
    rgppeqn::faceRowSum<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                        gB.rowSum);

    // wA = A x0
    rgppeqn::faceSpmv<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                      gB.x, gB.wA);
    rgppeqn::residualInit<<<nb(nc), BS>>>(nc, gB.b, gB.wA, gB.rA);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "peqn/init launch");

    // OF normFactor: xRef = gAverage(x);
    // normFactor = Σ|Ax - rs*xRef| + Σ|b - rs*xRef| + SMALL
    double sumX = 0;
    if ((rc = reduceHost(nc, 0, gB.x, nullptr, 0, nullptr, sumX))) return rc;
    const double xRef = sumX/nc;
    double nf1 = 0, nf2 = 0;
    if ((rc = reduceHost(nc, 3, gB.wA, nullptr, xRef, gB.rowSum, nf1)))
        return rc;
    if ((rc = reduceHost(nc, 3, gB.b, nullptr, xRef, gB.rowSum, nf2)))
        return rc;
    const double normFactor = nf1 + nf2 + 1e-20;

    double res = 0;
    if ((rc = reduceHost(nc, 1, gB.rA, nullptr, 0, nullptr, res))) return rc;
    res /= normFactor;
    *initRes = res;

    int iter = 0;
    double rArD = 0, rArDold = 0;

    auto converged = [&](double r) -> bool
    {
        return (r < tol) || (relTol > 0 && r <= relTol*(*initRes));
    };

    if (!converged(res))
    {
        while (iter < maxIter)
        {
            rArDold = rArD;

            // 전처리 방향: pA = rA/diag + beta*pA
            double beta = 0.0;
            if (iter > 0)
            {
                // rArD(new)는 아래에서 계산 — 표준 PCG 순서 유지 위해
                // 먼저 z=r/D와 r·z를 구한 뒤 방향 갱신
            }

            // z·r (z = rA/diag)를 위해: reduce mode 2 with b = rA/diag —
            // 별도 z 버퍼 대신 wA를 z로 재사용
            rgppeqn::precondDir<<<nb(nc), BS>>>(nc, gB.rA, gB.diag, 0.0,
                                                gB.wA);   // wA = z
            if ((rc = reduceHost(nc, 2, gB.wA, gB.rA, 0, nullptr, rArD)))
                return rc;

            beta = (iter > 0) ? rArD/rArDold : 0.0;
            rgppeqn::precondDir<<<nb(nc), BS>>>(nc, gB.rA, gB.diag, beta,
                                                gB.pA);   // pA = z + beta*pA

            // wA = A pA
            rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, gB.diag, gB.pA, gB.wA);
            rgppeqn::faceSpmv<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                              gB.pA, gB.wA);

            double pAwA = 0;
            if ((rc = reduceHost(nc, 2, gB.pA, gB.wA, 0, nullptr, pAwA)))
                return rc;
            if (pAwA == 0) break;   // 특이 — 안전 탈출

            const double alpha = rArD/pAwA;
            rgppeqn::cgUpdate<<<nb(nc), BS>>>(nc, alpha, gB.pA, gB.wA,
                                              gB.x, gB.rA);

            iter++;
            if ((rc = reduceHost(nc, 1, gB.rA, nullptr, 0, nullptr, res)))
                return rc;
            res /= normFactor;
            if (converged(res)) break;
        }
    }

    *finalRes = res;
    *nIters = iter;

    // 플럭스 + 결과 회수
    rgppeqn::faceFlux<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                      gB.x, gB.flux);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "peqn/flux launch");

    if ((e = cudaMemcpy(pOut, gB.x, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/D2H p");
    if ((e = cudaMemcpy(fluxInt, gB.flux, (size_t)nf*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/D2H flux");

    return 0;
}


int rgpPEqnAssembleDump
(
    double dtInv,
    const double* rAUfInt,
    const double* psis, const double* pOld,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double* diagOut, double* upperOut, double* bOut
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;
    int rc = assemble(dtInv, rAUfInt, psis, pOld, phiInt, phiB,
                      bDiag, bSrc, needRef, refCell, refValue);
    if (rc) return rc;

    cudaError_t e;
    if ((e = cudaMemcpy(diagOut, gB.diag, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/dump diag");
    if ((e = cudaMemcpy(upperOut, gB.upper, (size_t)nf*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/dump upper");
    if ((e = cudaMemcpy(bOut, gB.b, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/dump b");
    return 0;
}


void rgpPEqnFree(void) { pFreeAll(); }

const char* rgpPEqnLastError(void) { return gPErr; }

} // extern "C"

// ************************************************************************* //
