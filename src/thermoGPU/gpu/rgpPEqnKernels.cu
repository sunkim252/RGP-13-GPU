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
#include "rgpStage.H"

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <vector>

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

        // CSR (AmgX 직결용): 구조는 호스트 보관, 슬롯맵·값은 디바이스
        std::vector<int> hOwn, hNei, hRowPtr, hColInd;
        int nnz = 0;
        int *dDiagSlot = nullptr, *dUpSlot = nullptr, *dLoSlot = nullptr;
        int *dRowPtr = nullptr, *dColInd = nullptr;
        double *dVals = nullptr;
    } gM;

    //- 스칼라 수송(Z/C) 확장: limitedLinear div + laplacian + ddt/Sp
    struct STMesh
    {
        double *wLin = nullptr;              // 내부면 linear 가중치
        double *sf = nullptr, *dvec = nullptr;   // [3*nf] Sf, C_n-C_o
        double *bSf = nullptr;               // [3*nbf] 경계면 Sf
    } gSTM;

    struct STBuf
    {
        double *rho = nullptr, *rhoOld = nullptr, *psiOld = nullptr,
               *gamma = nullptr, *sp = nullptr, *src = nullptr;
        double *phi = nullptr, *wLim = nullptr, *lower = nullptr;
        double *grad = nullptr;              // [3*nc]
        double *bPsi = nullptr;
        double *rA0 = nullptr, *sA = nullptr, *yz = nullptr,
               *AyA = nullptr, *AzA = nullptr;
    } gST;

    //- UEqn(벡터, 성분 공유 LDU) 퍼시스턴트 버퍼 — pCorr의 A()/H()가
    //  스텝 내내 참조하므로 ST/pEqn 버퍼와 분리
    struct UBuf
    {
        double *diag = nullptr, *upper = nullptr, *lower = nullptr;
        double *b3 = nullptr, *U3 = nullptr;          // [3*nc]
        double *bDiag3 = nullptr, *bSrc3 = nullptr;   // [3*nbf]
        double *workDiag = nullptr, *workB = nullptr;
        double *H = nullptr;                          // [nc] 성분 작업
        double *grad9 = nullptr;                      // [9*nc] grad(U)
        double *src3 = nullptr, *gradP = nullptr;     // [3*nc]
        double *UB3 = nullptr, *bfx3 = nullptr;       // [3*nbf]
        double *bg9 = nullptr;                        // [9*nbf]
        double *rAU = nullptr, *HbyA3 = nullptr;      // [nc], [3*nc]
        bool assembled = false;   // rgpUEqnSolve 조립 완료 여부 (AH 가드)
    } gU;

    //- pCorr 준비체인 디바이스 버퍼 (rAUf/phiHbyAv/psis/rhof 상주)
    struct PCBuf
    {
        double *rAUf = nullptr, *phiH = nullptr, *rhof = nullptr;
        double *psis = nullptr;
        double *rhoOld = nullptr, *Uold3 = nullptr, *phiOld = nullptr;
    } gPC;

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
        double *rdt = nullptr;   // LTS rDeltaT [nc] (없으면 균일 dtInv)
    } gB;

    void pFreeAll()
    {
        int* ip[] = {gM.own, gM.nei, gM.bfc,
                     gM.dDiagSlot, gM.dUpSlot, gM.dLoSlot,
                     gM.dRowPtr, gM.dColInd};
        for (auto p : ip) { if (p) cudaFree(p); }
        gM.own = gM.nei = gM.bfc = nullptr;
        gM.dDiagSlot = gM.dUpSlot = gM.dLoSlot = nullptr;
        gM.dRowPtr = gM.dColInd = nullptr;
        if (gM.dVals) { cudaFree(gM.dVals); gM.dVals = nullptr; }
        gM.hOwn.clear(); gM.hNei.clear();
        gM.hRowPtr.clear(); gM.hColInd.clear();
        gM.nnz = 0;

        double** dp[] =
        {
            &gM.gg, &gM.V,
            &gB.diag, &gB.upper, &gB.b, &gB.x, &gB.rA, &gB.pA, &gB.wA,
            &gB.rowSum, &gB.rAUf, &gB.psis, &gB.pOld, &gB.phiInt,
            &gB.phiB, &gB.bDiag, &gB.bSrc, &gB.flux, &gB.red, &gB.rdt,
            &gSTM.wLin, &gSTM.sf, &gSTM.dvec, &gSTM.bSf,
            &gST.rho, &gST.rhoOld, &gST.psiOld, &gST.gamma, &gST.sp,
            &gST.src, &gST.phi, &gST.wLim, &gST.lower, &gST.grad,
            &gST.bPsi, &gST.rA0, &gST.sA, &gST.yz, &gST.AyA, &gST.AzA,
            &gU.diag, &gU.upper, &gU.lower, &gU.b3, &gU.U3,
            &gU.bDiag3, &gU.bSrc3, &gU.workDiag, &gU.workB, &gU.H,
            &gU.grad9, &gU.src3, &gU.gradP, &gU.UB3, &gU.bfx3, &gU.bg9,
            &gU.rAU, &gU.HbyA3,
            &gPC.rAUf, &gPC.phiH, &gPC.rhof, &gPC.psis,
            &gPC.rhoOld, &gPC.Uold3, &gPC.phiOld
        };
        for (auto p : dp) { if (*p) { cudaFree(*p); *p = nullptr; } }
        gU.assembled = false;
        gM.nCells = gM.nIntFaces = gM.nBFaces = 0;
    }
}

namespace rgppeqn
{

constexpr int BS = 128;

__global__ void cellAssemble
(
    const int n, const double dtInv, const double* __restrict__ rdt,
    const double* __restrict__ psis, const double* __restrict__ V,
    const double* __restrict__ pOld,
    double* __restrict__ diag, double* __restrict__ b
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double a = psis[i]*V[i]*(rdt ? rdt[i] : dtInv);
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

//- LDU → CSR 값 산포 (구조는 정적, 슬롯맵으로 O(1) 기록)
__global__ void csrScatterDiag
(
    const int n, const int* __restrict__ slot,
    const double* __restrict__ diag, double* __restrict__ vals
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    vals[slot[i]] = diag[i];
}

__global__ void csrScatterFace
(
    const int nf,
    const int* __restrict__ upSlot, const int* __restrict__ loSlot,
    const double* __restrict__ upper, double* __restrict__ vals
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    vals[upSlot[f]] = upper[f];   // 대칭: lower == upper
    vals[loSlot[f]] = upper[f];
}


//- 무-atomics CSR SpMV (행당 1스레드, 7-포인트)
__global__ void csrSpmv
(
    const int nc, const int* __restrict__ rowPtr,
    const int* __restrict__ colInd, const double* __restrict__ vals,
    const double* __restrict__ x, double* __restrict__ y
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    double s = 0.0;
    const int e = rowPtr[i + 1];
    for (int k = rowPtr[i]; k < e; k++)
    {
        s += vals[k]*x[colInd[k]];
    }
    y[i] = s;
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

// ── 스칼라 수송(Z/C): Gauss linear grad + limitedLinear(k=1) + 조립 ──

//- Gauss linear 셀 그래디언트: g[k*nc+c] = (1/V) Σ ±Sf_k ψ_f
__global__ void stGradFace
(
    const int nf, const int nc,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ wLin, const double* __restrict__ sf,
    const double* __restrict__ x, double* __restrict__ g
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const double xf = wLin[f]*x[own[f]] + (1.0 - wLin[f])*x[nei[f]];
    for (int k = 0; k < 3; k++)
    {
        const double c = sf[(size_t)k*nf + f]*xf;
        atomicAdd(&g[(size_t)k*nc + own[f]],  c);
        atomicAdd(&g[(size_t)k*nc + nei[f]], -c);
    }
}

__global__ void stGradBFace
(
    const int nbf, const int nc,
    const int* __restrict__ bfc, const double* __restrict__ bSf,
    const double* __restrict__ bPsi, double* __restrict__ g
)
{
    const int k2 = blockIdx.x*blockDim.x + threadIdx.x;
    if (k2 >= nbf) return;
    for (int k = 0; k < 3; k++)
    {
        atomicAdd
        (
            &g[(size_t)k*nc + bfc[k2]],
            bSf[(size_t)k*nbf + k2]*bPsi[k2]
        );
    }
}

__global__ void stGradDivV
(
    const int nc, const double* __restrict__ V, double* __restrict__ g
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    for (int k = 0; k < 3; k++) { g[(size_t)k*nc + i] /= V[i]; }
}

//- limitedLinear(k=1) 면 limiter를 min-누적 (multivariate 규약:
//  fields 테이블의 각 필드 limiter의 min). OF NVDTVD::r 1:1.
__global__ void stLimField
(
    const int nf, const int nc,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ dvec,
    const double* __restrict__ phi, const double* __restrict__ x,
    const double* __restrict__ g, double* __restrict__ lim
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;

    const int o = own[f], n = nei[f];
    const double gradf = x[n] - x[o];
    const int c = (phi[f] > 0.0) ? o : n;
    const double gradcf =
        dvec[f]*g[c]
      + dvec[(size_t)nf + f]*g[(size_t)nc + c]
      + dvec[(size_t)2*nf + f]*g[(size_t)2*nc + c];

    double r;
    if (fabs(gradcf) >= 1000.0*fabs(gradf))
    {
        const double sg = (gradcf >= 0.0) ? 1.0 : -1.0;
        const double sf2 = (gradf >= 0.0) ? 1.0 : -1.0;
        r = 2.0*1000.0*sg*sf2 - 1.0;
    }
    else
    {
        r = 2.0*(gradcf/gradf) - 1.0;
    }

    lim[f] = fmin(lim[f], fmax(fmin(2.0*r, 1.0), 0.0));
}

__global__ void stLimInit(const int nf, double* __restrict__ lim)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    lim[f] = 1.0;
}

//- w = lim*wLin + (1-lim)*upwind (in-place: lim 버퍼가 w가 됨)
__global__ void stWeightsEnd
(
    const int nf, const double* __restrict__ wLin,
    const double* __restrict__ phi, double* __restrict__ limW
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const double lim = limW[f];
    limW[f] = lim*wLin[f] + (1.0 - lim)*((phi[f] >= 0.0) ? 1.0 : 0.0);
}

//- 셀 항: fvm::ddt(rho,ψ) − fvm::Sp(sp,ψ) (+ src)
__global__ void stCellAssemble
(
    const int n, const double dtInv, const double* __restrict__ rdt,
    const int hasSrc,
    const double* __restrict__ rho, const double* __restrict__ rhoOld,
    const double* __restrict__ psiOld, const double* __restrict__ sp,
    const double* __restrict__ src, const double* __restrict__ V,
    double* __restrict__ diag, double* __restrict__ b
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double r = rdt ? rdt[i] : dtInv;
    diag[i] = rho[i]*V[i]*r - sp[i]*V[i];
    b[i] = rhoOld[i]*V[i]*r*psiOld[i] + (hasSrc ? src[i]*V[i] : 0.0);
}

//- 면 항: fvm::div(phi,ψ)[limited w] − fvm::laplacian(γ,ψ) + negSumDiag
__global__ void stFaceAssemble
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ phi, const double* __restrict__ w,
    const double* __restrict__ wLin, const double* __restrict__ gamma,
    const double* __restrict__ gg,
    double* __restrict__ diag, double* __restrict__ upper,
    double* __restrict__ lower
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const int o = own[f], n = nei[f];
    const double G = (wLin[f]*gamma[o] + (1.0 - wLin[f])*gamma[n])*gg[f];
    const double lo = -w[f]*phi[f] - G;
    const double up = (1.0 - w[f])*phi[f] - G;
    lower[f] = lo;
    upper[f] = up;
    atomicAdd(&diag[o], -lo);
    atomicAdd(&diag[n], -up);
}

//- 경계 기여 (phiB 차감 없는 순수 diag/source 산입)
__global__ void stBFaceAssemble
(
    const int nbf, const int* __restrict__ bfc,
    const double* __restrict__ bDiag, const double* __restrict__ bSrc,
    double* __restrict__ diag, double* __restrict__ b
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    atomicAdd(&diag[bfc[k]], bDiag[k]);
    atomicAdd(&b[bfc[k]], bSrc[k]);
}

//- 비대칭 SpMV 면 기여 / rowSum
__global__ void stFaceSpmv
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ upper, const double* __restrict__ lower,
    const double* __restrict__ x, double* __restrict__ y
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    atomicAdd(&y[own[f]], upper[f]*x[nei[f]]);
    atomicAdd(&y[nei[f]], lower[f]*x[own[f]]);
}

__global__ void stFaceRowSum
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ upper, const double* __restrict__ lower,
    double* __restrict__ rs
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    atomicAdd(&rs[own[f]], upper[f]);
    atomicAdd(&rs[nei[f]], lower[f]);
}

//- BiCGStab 벡터 연산들
__global__ void stPA
(
    const int n, const double beta, const double omega,
    const double* __restrict__ rA, const double* __restrict__ AyA,
    double* __restrict__ pA
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    pA[i] = rA[i] + beta*(pA[i] - omega*AyA[i]);
}

__global__ void stAxpy
(
    const int n, const double a,
    const double* __restrict__ x, const double* __restrict__ y,
    double* __restrict__ out
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = y[i] + a*x[i];
}

__global__ void stX2
(
    const int n, const double alpha, const double omega,
    const double* __restrict__ yA, const double* __restrict__ zA,
    double* __restrict__ x
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += alpha*yA[i] + omega*zA[i];
}

// ── UEqn(벡터, 성분 공유 LDU): 셀 항 + A()/H() 산출 ──────────────────

//- fvm::ddt(rho,U): 공유 diag + 성분별 b (명시 소스 srcExp3는 per-volume)
__global__ void uCellAssemble
(
    const int nc, const double dtInv, const double* __restrict__ rdt,
    const double* __restrict__ rho, const double* __restrict__ rhoOld,
    const double* __restrict__ V,
    const double* __restrict__ U3old, const double* srcExp3,
    double* __restrict__ diag, double* b3
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    const double r = rdt ? rdt[i] : dtInv;
    diag[i] = rho[i]*V[i]*r;
    const double ao = rhoOld[i]*V[i]*r;
    for (int k = 0; k < 3; k++)
    {
        b3[(size_t)k*nc + i] =
            ao*U3old[(size_t)k*nc + i] + srcExp3[(size_t)k*nc + i]*V[i];
    }
}

//- fvMatrix::D()의 addCmptAvBoundaryDiag: diag[fc] += cmptAv(iC)
__global__ void uAvIC
(
    const int nbf, const int* __restrict__ bfc,
    const double* __restrict__ bDiag3, double* __restrict__ workDiag
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    const double avIC =
        (bDiag3[k] + bDiag3[(size_t)nbf + k] + bDiag3[(size_t)2*nbf + k])
       /3.0;
    atomicAdd(&workDiag[bfc[k]], avIC);
}

//- rAU = 1/A() = V/D
__global__ void uRAU
(
    const int nc, const double* __restrict__ V,
    const double* __restrict__ workDiag, double* __restrict__ out
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    out[i] = V[i]/workDiag[i];
}

//- lduMatrix::H: H[o] -= upper*x[n]; H[n] -= lower*x[o]
__global__ void uHFace
(
    const int nf,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ upper, const double* __restrict__ lower,
    const double* __restrict__ xc, double* __restrict__ H
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    atomicAdd(&H[own[f]], -upper[f]*xc[nei[f]]);
    atomicAdd(&H[nei[f]], -lower[f]*xc[own[f]]);
}

//- fvMatrix::H 경계: (cmptAv(iC) - iC_c)*U_c[fc] + bSrc_c
__global__ void uHBnd
(
    const int nbf, const int nc, const int cmpt,
    const int* __restrict__ bfc,
    const double* __restrict__ bDiag3, const double* __restrict__ bSrc3,
    const double* __restrict__ U3, double* __restrict__ H
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    const int c = bfc[k];
    const double avIC =
        (bDiag3[k] + bDiag3[(size_t)nbf + k] + bDiag3[(size_t)2*nbf + k])
       /3.0;
    atomicAdd
    (
        &H[c],
        (avIC - bDiag3[(size_t)cmpt*nbf + k])*U3[(size_t)cmpt*nc + c]
      + bSrc3[(size_t)cmpt*nbf + k]
    );
}

//- 벡터장 Gauss linear 그래디언트: g9[(3i+j)*nc+c] += Sf_i * Uf_j
__global__ void uVecGradFace
(
    const int nf, const int nc,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ wLin, const double* __restrict__ sf,
    const double* __restrict__ U3, double* __restrict__ g9
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const int o = own[f], n = nei[f];
    const double w = wLin[f];
    for (int j = 0; j < 3; j++)
    {
        const double uf =
            w*U3[(size_t)j*nc + o] + (1.0 - w)*U3[(size_t)j*nc + n];
        for (int i = 0; i < 3; i++)
        {
            const double c = sf[(size_t)i*nf + f]*uf;
            atomicAdd(&g9[(size_t)(3*i + j)*nc + o],  c);
            atomicAdd(&g9[(size_t)(3*i + j)*nc + n], -c);
        }
    }
}

__global__ void uVecGradBnd
(
    const int nbf, const int nc,
    const int* __restrict__ bfc, const double* __restrict__ bSf,
    const double* __restrict__ UB3, double* __restrict__ g9
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    for (int j = 0; j < 3; j++)
    {
        const double ub = UB3[(size_t)j*nbf + k];
        for (int i = 0; i < 3; i++)
        {
            atomicAdd
            (
                &g9[(size_t)(3*i + j)*nc + bfc[k]],
                bSf[(size_t)i*nbf + k]*ub
            );
        }
    }
}

__global__ void uGrad9DivV
(
    const int nc, const double* __restrict__ V, double* __restrict__ g9
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    for (int m = 0; m < 9; m++) { g9[(size_t)m*nc + i] /= V[i]; }
}

//- 경계 셀의 gradU 채집 (호스트가 경계 보정/명시항 X_b 계산용)
__global__ void uGradGather
(
    const int nbf, const int nc, const int* __restrict__ bfc,
    const double* __restrict__ g9, double* __restrict__ bg9
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    for (int m = 0; m < 9; m++)
    {
        bg9[(size_t)m*nbf + k] = g9[(size_t)m*nc + bfc[k]];
    }
}

//- limitedLinearV(k=1) 가중치 (OF NVDVTVDV::r 1:1 — guard 방향 주의)
__global__ void uLimVWeights
(
    const int nf, const int nc,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ wLin, const double* __restrict__ dvec,
    const double* __restrict__ phi, const double* __restrict__ U3,
    const double* __restrict__ g9, double* __restrict__ w
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const int o = own[f], n = nei[f];

    double gfv[3];
    double gradf = 0.0;
    for (int j = 0; j < 3; j++)
    {
        gfv[j] = U3[(size_t)j*nc + n] - U3[(size_t)j*nc + o];
        gradf += gfv[j]*gfv[j];
    }

    const int c = (phi[f] > 0.0) ? o : n;
    double gradcf = 0.0;
    for (int j = 0; j < 3; j++)
    {
        double gcfj = 0.0;
        for (int i = 0; i < 3; i++)
        {
            gcfj += dvec[(size_t)i*nf + f]*g9[(size_t)(3*i + j)*nc + c];
        }
        gradcf += gfv[j]*gcfj;
    }

    double r;
    if (fabs(gradcf) >= 1000.0*fabs(gradf))
    {
        const double sg = (gradcf >= 0.0) ? 1.0 : -1.0;
        const double sf2 = (gradf >= 0.0) ? 1.0 : -1.0;
        r = 2.0*1000.0*sg*sf2 - 1.0;
    }
    else
    {
        r = 2.0*(gradcf/gradf) - 1.0;
    }

    const double lim = fmax(fmin(2.0*r, 1.0), 0.0);
    w[f] = lim*wLin[f] + (1.0 - lim)*((phi[f] >= 0.0) ? 1.0 : 0.0);
}

//- 명시항 div(mu*dev2(T(gradU))) 내부면 플럭스: flux_j = Sf_i Xf_ij,
//  X_ij = mu*(gradU_ji - (2/3) delta_ij tr(gradU)), Xf = 선형보간
__global__ void uDevTauFlux
(
    const int nf, const int nc,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ wLin, const double* __restrict__ sf,
    const double* __restrict__ mu, const double* __restrict__ g9,
    double* __restrict__ src3
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const int o = own[f], n = nei[f];
    const double w = wLin[f];

    double Xo[9], Xn[9];
    const double tro =
        g9[(size_t)0*nc + o] + g9[(size_t)4*nc + o] + g9[(size_t)8*nc + o];
    const double trn =
        g9[(size_t)0*nc + n] + g9[(size_t)4*nc + n] + g9[(size_t)8*nc + n];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            const double to = g9[(size_t)(3*j + i)*nc + o]
                - ((i == j) ? (2.0/3.0)*tro : 0.0);
            const double tn = g9[(size_t)(3*j + i)*nc + n]
                - ((i == j) ? (2.0/3.0)*trn : 0.0);
            Xo[3*i + j] = mu[o]*to;
            Xn[3*i + j] = mu[n]*tn;
        }
    }

    for (int j = 0; j < 3; j++)
    {
        double flux = 0.0;
        for (int i = 0; i < 3; i++)
        {
            flux += sf[(size_t)i*nf + f]
                   *(w*Xo[3*i + j] + (1.0 - w)*Xn[3*i + j]);
        }
        atomicAdd(&src3[(size_t)j*nc + o],  flux);
        atomicAdd(&src3[(size_t)j*nc + n], -flux);
    }
}

//- 명시항 경계 기여 (호스트 계산 [3*nbf]) 산입
__global__ void uBndSrcAdd
(
    const int nbf, const int nc, const int* __restrict__ bfc,
    const double* __restrict__ bfx3, double* __restrict__ src3
)
{
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nbf) return;
    for (int j = 0; j < 3; j++)
    {
        atomicAdd
        (
            &src3[(size_t)j*nc + bfc[k]], bfx3[(size_t)j*nbf + k]
        );
    }
}

__global__ void uSrc3DivV
(
    const int nc, const double* __restrict__ V, double* __restrict__ s3
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    for (int j = 0; j < 3; j++) { s3[(size_t)j*nc + i] /= V[i]; }
}

//- HbyA = rAU*H (성분별 — AH에서 디바이스 상주 사본 유지용)
__global__ void uHbyA
(
    const int nc, const int cmpt,
    const double* __restrict__ rAU, const double* __restrict__ H,
    double* __restrict__ HbyA3
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    HbyA3[(size_t)cmpt*nc + i] = rAU[i]*H[i];
}

//- pCorr 준비 면 커널: rhof/rAUf(조화)/phiHbyAv(= flux(HbyA) +
//  rcDdtScale*rhorAUf*coeff*rDeltaT*phiCorr/rhof; coeff = OF
//  fvcDdtPhiCoeff flux-normalised, 경계 0은 호스트 처리)
__global__ void pcPrepFace
(
    const int nf, const int nc,
    const int* __restrict__ own, const int* __restrict__ nei,
    const double* __restrict__ wLin, const double* __restrict__ sf,
    const double* __restrict__ rho, const double* __restrict__ rAU,
    const double* __restrict__ HbyA3,
    const double* __restrict__ rhoOld, const double* __restrict__ Uold3,
    const double* __restrict__ phiOld,
    const double rDeltaT, const double* __restrict__ rdt,
    const double rcDdtScale,
    double* __restrict__ rhof, double* __restrict__ rAUf,
    double* __restrict__ phiH
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    const int o = own[f], n = nei[f];
    const double w = wLin[f];

    const double rf = w*rho[o] + (1.0 - w)*rho[n];
    rhof[f] = rf;
    rAUf[f] = 1.0/(w*(1.0/rAU[o]) + (1.0 - w)*(1.0/rAU[n]));
    const double rrAUf = w*rho[o]*rAU[o] + (1.0 - w)*rho[n]*rAU[n];

    double fluxH = 0.0, phiCorr = 0.0;
    for (int k = 0; k < 3; k++)
    {
        const double sfk = sf[(size_t)k*nf + f];
        fluxH += sfk*(w*HbyA3[(size_t)k*nc + o]
                    + (1.0 - w)*HbyA3[(size_t)k*nc + n]);
        // rhoU0 = rho.old*U.old의 선형보간 · Sf
        phiCorr -= sfk*(w*rhoOld[o]*Uold3[(size_t)k*nc + o]
                      + (1.0 - w)*rhoOld[n]*Uold3[(size_t)k*nc + n]);
    }
    phiCorr += phiOld[f];

    const double coeff =
        1.0 - fmin(fabs(phiCorr)/(fabs(phiOld[f]) + 1e-15), 1.0);

    // localEuler: rDeltaT_f = 선형보간 (fvc::interpolate(localRDeltaT))
    const double rdtf =
        rdt ? (w*rdt[o] + (1.0 - w)*rdt[n]) : rDeltaT;

    phiH[f] = fluxH + rcDdtScale*rrAUf*coeff*rdtf*phiCorr/rf;
}

//- psis = psi/rho
__global__ void pcPsis
(
    const int nc, const double* __restrict__ psi,
    const double* __restrict__ rho, double* __restrict__ psis
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    psis[i] = psi[i]/rho[i];
}

//- phi 재구성: phi = rhof*(phiHbyAv + flux)
__global__ void pcPhiRecon
(
    const int nf, const double* __restrict__ rhof,
    const double* __restrict__ phiH, const double* flux,
    double* phiOut
)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nf) return;
    phiOut[f] = rhof[f]*(phiH[f] + flux[f]);
}

//- U = HbyA − rAU*grad(p) (성분별 SoA out)
__global__ void pcUUpdate
(
    const int nc,
    const double* __restrict__ HbyA3, const double* __restrict__ rAU,
    const double* __restrict__ gradP, double* __restrict__ U3
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    for (int k = 0; k < 3; k++)
    {
        U3[(size_t)k*nc + i] =
            HbyA3[(size_t)k*nc + i] - rAU[i]*gradP[(size_t)k*nc + i];
    }
}

//- 솔브 전용 소스: workB += s*g*V (grad p는 저장 소스가 아니라
//  solve(UEqn == -grad p)의 임시 소스 — H()에 포함되면 안 됨)
__global__ void uAddVolSrc
(
    const int nc, const double s,
    const double* __restrict__ g, const double* __restrict__ V,
    double* __restrict__ b
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    b[i] += s*g[i]*V[i];
}

//- H /= V
__global__ void uDivV
(
    const int nc, const double* __restrict__ V, double* __restrict__ H
)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nc) return;
    H[i] /= V[i];
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

    //- LTS rDeltaT 스테이징: null이면 null 유지(균일 dtInv)
    const double* stageRdt(const double* rdtH, cudaError_t* e)
    {
        *e = cudaSuccess;
        if (!rdtH) return nullptr;
        return rgpInPtr(rdtH, gB.rdt, (size_t)gM.nCells, e);
    }

    //- 조립 공통부 (업로드 + diag/upper/b 완성). 0=성공.
    int assemble
    (
        double dtInv, const double* rdtH,
        const double* rAUfInt, const double* psis, const double* pOld,
        const double* phiInt, const double* phiB,
        const double* bDiag, const double* bSrc,
        const double* srcExtra,
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
        // null 입력 → rgpPCorrPrep 디바이스 상주본(rAUf/psis/phiHbyAv)
        const double* dRAUf = rAUfInt ? gB.rAUf : gPC.rAUf;
        const double* dPsis = psis ? gB.psis : gPC.psis;
        const double* dPhi = phiInt ? gB.phiInt : gPC.phiH;
        struct { double* d; const double* s; size_t n; } up[] =
        {
            {gB.rAUf, rAUfInt, rAUfInt ? (size_t)nf : 0},
            {gB.psis, psis, psis ? (size_t)nc : 0},
            {gB.pOld, pOld, (size_t)nc},
            {gB.phiInt, phiInt, phiInt ? (size_t)nf : 0},
            {gB.phiB, phiB, (size_t)nbf}, {gB.bDiag, bDiag, (size_t)nbf},
            {gB.bSrc, bSrc, (size_t)nbf}
        };
        for (auto& u : up)
        {
            if (u.n == 0 || !u.s) continue;
            if ((e = cudaMemcpy(u.d, u.s, u.n*sizeof(double),
                                cudaMemcpyHostToDevice)) != cudaSuccess)
                return pfail(e, "peqn/H2D");
        }

        cudaError_t er;
        const double* dRdt = stageRdt(rdtH, &er);
        if (er != cudaSuccess) return pfail(er, "peqn/rdt");
        rgppeqn::cellAssemble<<<nb(nc), BS>>>
            (nc, dtInv, dRdt, dPsis, gM.V, gB.pOld, gB.diag, gB.b);
        if (srcExtra)   // AMD 등 명시항 (저장 소스: b += extra*V)
        {
            const double* dX = rgpInPtr(srcExtra, gB.pA, (size_t)nc, &er);
            if (er != cudaSuccess) return pfail(er, "peqn/srcExtra");
            rgppeqn::uAddVolSrc<<<nb(nc), BS>>>(nc, 1.0, dX, gM.V, gB.b);
        }
        rgppeqn::faceAssemble<<<nb(nf), BS>>>
            (nf, gM.own, gM.nei, gM.gg, dRAUf, dPhi,
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
        // CSR 구조가 준비돼 있으면 값 산포 (무-atomics SpMV 경로)
        if (gM.dVals && gM.dRowPtr)
        {
            rgppeqn::csrScatterDiag<<<nb(nc), BS>>>
                (nc, gM.dDiagSlot, gB.diag, gM.dVals);
            rgppeqn::csrScatterFace<<<nb(nf), BS>>>
                (nf, gM.dUpSlot, gM.dLoSlot, gB.upper, gM.dVals);
        }
        if ((e = cudaGetLastError()) != cudaSuccess)
            return pfail(e, "peqn/assemble launch");
        return 0;
    }

    //- pEqn용 A·x: CSR 준비 시 무-atomics, 아니면 LDU+atomics
    void pSpmv(const int nc, const int nf, const double* x, double* y)
    {
        if (gM.dVals && gM.dRowPtr)
        {
            rgppeqn::csrSpmv<<<nb(nc), BS>>>
                (nc, gM.dRowPtr, gM.dColInd, gM.dVals, x, y);
        }
        else
        {
            rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, gB.diag, x, y);
            rgppeqn::faceSpmv<<<nb(nf), BS>>>
                (nf, gM.own, gM.nei, gB.upper, x, y);
        }
    }
    //- Jacobi-BiCGStab (OF PBiCGStab 구조/잔차 규약) — 임의 디바이스
    //  포인터의 (diag, upper, lower, b, x)에 대해. 스크래치는 gB/gST 공유.
    int bicgSolve
    (
        const int nc, const int nf,
        const double* diag, const double* upper, const double* lower,
        const double* b, double* x,
        const double tol, const double relTol, const int maxIter,
        double* initRes, double* finalRes, int* nIters
    )
    {
        cudaError_t e;
        int rc;

        if ((e = cudaMemcpy(gB.rowSum, diag, (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "bicg/rowSum D2D");
        rgppeqn::stFaceRowSum<<<nb(nf), BS>>>
            (nf, gM.own, gM.nei, upper, lower, gB.rowSum);

        rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, diag, x, gB.wA);
        rgppeqn::stFaceSpmv<<<nb(nf), BS>>>
            (nf, gM.own, gM.nei, upper, lower, x, gB.wA);
        rgppeqn::residualInit<<<nb(nc), BS>>>(nc, b, gB.wA, gB.rA);
        if ((e = cudaGetLastError()) != cudaSuccess)
            return pfail(e, "bicg/init launch");

        double sumX = 0;
        if ((rc = reduceHost(nc, 0, x, nullptr, 0, nullptr, sumX)))
            return rc;
        const double xRef = sumX/nc;
        double nf1 = 0, nf2 = 0;
        if ((rc = reduceHost(nc, 3, gB.wA, nullptr, xRef, gB.rowSum, nf1)))
            return rc;
        if ((rc = reduceHost(nc, 3, b, nullptr, xRef, gB.rowSum, nf2)))
            return rc;
        const double normFactor = nf1 + nf2 + 1e-20;

        double res = 0;
        if ((rc = reduceHost(nc, 1, gB.rA, nullptr, 0, nullptr, res)))
            return rc;
        res /= normFactor;
        *initRes = res;

        auto converged = [&](double r) -> bool
        {
            return (r < tol) || (relTol > 0 && r <= relTol*(*initRes));
        };

        int iter = 0;
        if (!converged(res))
        {
            if ((e = cudaMemcpy(gST.rA0, gB.rA, (size_t)nc*sizeof(double),
                                cudaMemcpyDeviceToDevice)) != cudaSuccess)
                return pfail(e, "bicg/rA0");

            double rA0rA = 0, alpha = 0, omega = 0;

            while (iter < maxIter)
            {
                const double rA0rAold = rA0rA;
                if ((rc = reduceHost(nc, 2, gST.rA0, gB.rA, 0, nullptr,
                                     rA0rA))) return rc;

                if (iter == 0)
                {
                    if ((e = cudaMemcpy(gB.pA, gB.rA,
                                        (size_t)nc*sizeof(double),
                                        cudaMemcpyDeviceToDevice))
                        != cudaSuccess) return pfail(e, "bicg/pA0");
                }
                else
                {
                    if (rA0rAold == 0 || omega == 0) break;
                    const double beta = (rA0rA/rA0rAold)*(alpha/omega);
                    rgppeqn::stPA<<<nb(nc), BS>>>
                        (nc, beta, omega, gB.rA, gST.AyA, gB.pA);
                }

                rgppeqn::precondDir<<<nb(nc), BS>>>
                    (nc, gB.pA, diag, 0.0, gST.yz);
                rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, diag, gST.yz,
                                                  gST.AyA);
                rgppeqn::stFaceSpmv<<<nb(nf), BS>>>
                    (nf, gM.own, gM.nei, upper, lower, gST.yz, gST.AyA);

                double rA0AyA = 0;
                if ((rc = reduceHost(nc, 2, gST.rA0, gST.AyA, 0, nullptr,
                                     rA0AyA))) return rc;
                if (rA0AyA == 0) break;
                alpha = rA0rA/rA0AyA;

                rgppeqn::stAxpy<<<nb(nc), BS>>>
                    (nc, -alpha, gST.AyA, gB.rA, gST.sA);

                double resS = 0;
                if ((rc = reduceHost(nc, 1, gST.sA, nullptr, 0, nullptr,
                                     resS))) return rc;
                resS /= normFactor;
                if (converged(resS))
                {
                    rgppeqn::stX2<<<nb(nc), BS>>>
                        (nc, alpha, 0.0, gST.yz, gST.yz, x);
                    iter++;
                    res = resS;
                    break;
                }

                rgppeqn::precondDir<<<nb(nc), BS>>>
                    (nc, gST.sA, diag, 0.0, gB.wA);
                rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, diag, gB.wA,
                                                  gST.AzA);
                rgppeqn::stFaceSpmv<<<nb(nf), BS>>>
                    (nf, gM.own, gM.nei, upper, lower, gB.wA, gST.AzA);

                double tS = 0, tT = 0;
                if ((rc = reduceHost(nc, 2, gST.AzA, gST.sA, 0, nullptr,
                                     tS))) return rc;
                if ((rc = reduceHost(nc, 2, gST.AzA, gST.AzA, 0, nullptr,
                                     tT))) return rc;
                if (tT == 0) break;
                omega = tS/tT;

                rgppeqn::stX2<<<nb(nc), BS>>>
                    (nc, alpha, omega, gST.yz, gB.wA, x);
                rgppeqn::stAxpy<<<nb(nc), BS>>>
                    (nc, -omega, gST.AzA, gST.sA, gB.rA);

                iter++;
                if ((rc = reduceHost(nc, 1, gB.rA, nullptr, 0, nullptr,
                                     res))) return rc;
                res /= normFactor;
                if (converged(res)) break;
            }
        }

        *finalRes = res;
        *nIters = iter;
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
    gM.hOwn.assign(owner, owner + nIntFaces);
    gM.hNei.assign(neigh, neigh + nIntFaces);

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
        {&gB.pOld, bc}, {&gB.rdt, bc},
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
    double dtInv, const double* rdtCell,
    const double* srcCellExtra,
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

    int rc = assemble(dtInv, rdtCell, rAUfInt, psis, pOld, phiInt, phiB,
                      bDiag, bSrc, srcCellExtra, needRef, refCell,
                      refValue);
    if (rc) return rc;

    cudaError_t e;
    if ((e = cudaMemcpy(gB.x, p0, (size_t)nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "peqn/H2D x0");

    // rowSum (normFactor용)
    if ((e = cudaMemcpy(gB.rowSum, gB.diag, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToDevice)) != cudaSuccess)
        return pfail(e, "peqn/rowSum D2D");
    rgppeqn::faceRowSum<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                        gB.rowSum);

    // wA = A x0
    pSpmv(nc, nf, gB.x, gB.wA);
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
            pSpmv(nc, nf, gB.pA, gB.wA);

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

    // 플럭스 + 결과 회수 (null이면 스킵 — devChain은 Finish2가 회수)
    if (fluxInt)
    {
        rgppeqn::faceFlux<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                          gB.x, gB.flux);
        if ((e = cudaGetLastError()) != cudaSuccess)
            return pfail(e, "peqn/flux launch");
        if ((e = cudaMemcpy(fluxInt, gB.flux, (size_t)nf*sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "peqn/D2H flux");
    }
    if (pOut)
    {
        if ((e = cudaMemcpy(pOut, gB.x, (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "peqn/D2H p");
    }

    return 0;
}


int rgpPEqnAssembleDump
(
    double dtInv, const double* rdtCell,
    const double* srcCellExtra,
    const double* rAUfInt,
    const double* psis, const double* pOld,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double* diagOut, double* upperOut, double* bOut
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;
    int rc = assemble(dtInv, rdtCell, rAUfInt, psis, pOld, phiInt, phiB,
                      bDiag, bSrc, srcCellExtra, needRef, refCell,
                      refValue);
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


int rgpPEqnCsrPrepare(int* nnzOut)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;
    if (nc <= 0)
    {
        snprintf(gPErr, sizeof(gPErr), "peqn/csr: mesh not uploaded");
        return -1;
    }

    // 구조: row 당 [lower cols(<row)] [diag] [upper cols(>row)] —
    // amgxSolver.C와 동일한 삽입 순서 (lduAddr 정렬 승계)
    const int nnz = nc + 2*nf;
    gM.hRowPtr.assign(nc + 1, 0);
    for (int f = 0; f < nf; f++)
    {
        gM.hRowPtr[gM.hNei[f] + 1]++;
        gM.hRowPtr[gM.hOwn[f] + 1]++;
    }
    for (int r = 0; r < nc; r++)
    {
        gM.hRowPtr[r + 1] += gM.hRowPtr[r] + 1;
    }

    gM.hColInd.assign(nnz, -1);
    std::vector<int> diagSlot(nc), upSlot(nf), loSlot(nf), fill(nc, 0);

    for (int f = 0; f < nf; f++)      // lower: row=nei, col=own
    {
        const int r = gM.hNei[f];
        const int k = gM.hRowPtr[r] + fill[r]++;
        gM.hColInd[k] = gM.hOwn[f];
        loSlot[f] = k;
    }
    for (int r = 0; r < nc; r++)      // diag
    {
        const int k = gM.hRowPtr[r] + fill[r]++;
        gM.hColInd[k] = r;
        diagSlot[r] = k;
    }
    for (int f = 0; f < nf; f++)      // upper: row=own, col=nei
    {
        const int r = gM.hOwn[f];
        const int k = gM.hRowPtr[r] + fill[r]++;
        gM.hColInd[k] = gM.hNei[f];
        upSlot[f] = k;
    }

    cudaError_t e;
    struct { int** d; const int* s; size_t n; } it[] =
    {
        {&gM.dDiagSlot, diagSlot.data(), (size_t)nc},
        {&gM.dUpSlot, upSlot.data(), (size_t)nf},
        {&gM.dLoSlot, loSlot.data(), (size_t)nf}
    };
    for (auto& i : it)
    {
        if (!*i.d)
        {
            if ((e = cudaMalloc(i.d, i.n*sizeof(int))) != cudaSuccess)
                return pfail(e, "peqn/csr malloc");
        }
        if ((e = cudaMemcpy(*i.d, i.s, i.n*sizeof(int),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "peqn/csr H2D");
    }
    if (!gM.dVals)
    {
        if ((e = cudaMalloc(&gM.dVals, (size_t)nnz*sizeof(double)))
            != cudaSuccess) return pfail(e, "peqn/csr vals malloc");
    }
    // CSR 구조 디바이스 사본 (무-atomics SpMV용)
    struct { int** d; const int* s; size_t n; } cs[] =
    {
        {&gM.dRowPtr, gM.hRowPtr.data(), (size_t)(nc + 1)},
        {&gM.dColInd, gM.hColInd.data(), (size_t)nnz}
    };
    for (auto& c : cs)
    {
        if (!*c.d)
        {
            if ((e = cudaMalloc(c.d, c.n*sizeof(int))) != cudaSuccess)
                return pfail(e, "peqn/csr struct malloc");
        }
        if ((e = cudaMemcpy(*c.d, c.s, c.n*sizeof(int),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "peqn/csr struct H2D");
    }

    gM.nnz = nnz;
    *nnzOut = nnz;
    return 0;
}


const int* rgpPEqnCsrRowPtr(void) { return gM.hRowPtr.data(); }
const int* rgpPEqnCsrColInd(void) { return gM.hColInd.data(); }
void* rgpPEqnDevValues(void) { return gM.dVals; }
void* rgpPEqnDevB(void) { return gB.b; }
void* rgpPEqnDevX(void) { return gB.x; }


int rgpPEqnAssembleCsr
(
    double dtInv, const double* rdtCell,
    const double* srcCellExtra,
    const double* rAUfInt,
    const double* psis, const double* pOld, const double* p0,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double* normFactor, double* initRes
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;
    if (gM.nnz <= 0)
    {
        snprintf(gPErr, sizeof(gPErr), "peqn/csr: not prepared");
        return -1;
    }

    int rc = assemble(dtInv, rdtCell, rAUfInt, psis, pOld, phiInt, phiB,
                      bDiag, bSrc, srcCellExtra, needRef, refCell,
                      refValue);
    if (rc) return rc;

    // LDU → CSR 값 산포
    rgppeqn::csrScatterDiag<<<nb(nc), BS>>>(nc, gM.dDiagSlot, gB.diag,
                                            gM.dVals);
    rgppeqn::csrScatterFace<<<nb(nf), BS>>>(nf, gM.dUpSlot, gM.dLoSlot,
                                            gB.upper, gM.dVals);

    cudaError_t e;
    if ((e = cudaMemcpy(gB.x, p0, (size_t)nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "peqn/csr H2D x0");

    // OF normFactor + 초기 잔차 (rgpPEqnSolve와 동일 규약)
    if ((e = cudaMemcpy(gB.rowSum, gB.diag, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToDevice)) != cudaSuccess)
        return pfail(e, "peqn/csr rowSum D2D");
    rgppeqn::faceRowSum<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                        gB.rowSum);

    rgppeqn::diagSpmv<<<nb(nc), BS>>>(nc, gB.diag, gB.x, gB.wA);
    rgppeqn::faceSpmv<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                      gB.x, gB.wA);
    rgppeqn::residualInit<<<nb(nc), BS>>>(nc, gB.b, gB.wA, gB.rA);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "peqn/csr launch");

    double sumX = 0;
    if ((rc = reduceHost(nc, 0, gB.x, nullptr, 0, nullptr, sumX))) return rc;
    const double xRef = sumX/nc;
    double nf1 = 0, nf2 = 0;
    if ((rc = reduceHost(nc, 3, gB.wA, nullptr, xRef, gB.rowSum, nf1)))
        return rc;
    if ((rc = reduceHost(nc, 3, gB.b, nullptr, xRef, gB.rowSum, nf2)))
        return rc;
    *normFactor = nf1 + nf2 + 1e-20;

    double res = 0;
    if ((rc = reduceHost(nc, 1, gB.rA, nullptr, 0, nullptr, res))) return rc;
    *initRes = res/(*normFactor);

    return 0;
}


int rgpPEqnResidual(double normFactor, double* res)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;

    pSpmv(nc, nf, gB.x, gB.wA);
    rgppeqn::residualInit<<<nb(nc), BS>>>(nc, gB.b, gB.wA, gB.rA);
    cudaError_t e;
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "peqn/res launch");

    double r = 0;
    int rc;
    if ((rc = reduceHost(nc, 1, gB.rA, nullptr, 0, nullptr, r))) return rc;
    *res = r/normFactor;
    return 0;
}


int rgpPEqnFinish(double* pOut, double* fluxInt)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;

    rgppeqn::faceFlux<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                      gB.x, gB.flux);
    cudaError_t e;
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "peqn/finish launch");

    if ((e = cudaMemcpy(pOut, gB.x, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/finish D2H p");
    if ((e = cudaMemcpy(fluxInt, gB.flux, (size_t)nf*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "peqn/finish D2H flux");
    return 0;
}


int rgpSTEqnMeshUpload
(
    const double* wLin, const double* Sf, const double* dVec,
    const double* bSf
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
    if (nc <= 0)
    {
        snprintf(gPErr, sizeof(gPErr), "steqn: mesh not uploaded");
        return -1;
    }

    cudaError_t e;
    struct { double** d; const double* s; size_t n; } it[] =
    {
        {&gSTM.wLin, wLin, (size_t)nf},
        {&gSTM.sf, Sf, (size_t)3*nf},
        {&gSTM.dvec, dVec, (size_t)3*nf},
        {&gSTM.bSf, bSf, (size_t)3*(nbf > 0 ? nbf : 1)}
    };
    for (auto& i : it)
    {
        if (!*i.d)
        {
            if ((e = cudaMalloc(i.d, i.n*sizeof(double))) != cudaSuccess)
                return pfail(e, "steqn/mesh malloc");
        }
        if (i.s)
        {
            if ((e = cudaMemcpy(*i.d, i.s, i.n*sizeof(double),
                                cudaMemcpyHostToDevice)) != cudaSuccess)
                return pfail(e, "steqn/mesh H2D");
        }
    }

    const size_t bc = (size_t)nc*sizeof(double);
    const size_t bf = (size_t)nf*sizeof(double);
    const size_t bb = (size_t)(nbf > 0 ? nbf : 1)*sizeof(double);
    struct { double** d; size_t bytes; } al[] =
    {
        {&gST.rho, bc}, {&gST.rhoOld, bc}, {&gST.psiOld, bc},
        {&gST.gamma, bc}, {&gST.sp, bc}, {&gST.src, bc},
        {&gST.phi, bf}, {&gST.wLim, bf}, {&gST.lower, bf},
        {&gST.grad, 3*bc}, {&gST.bPsi, bb},
        {&gST.rA0, bc}, {&gST.sA, bc}, {&gST.yz, bc},
        {&gST.AyA, bc}, {&gST.AzA, bc}
    };
    for (auto& a : al)
    {
        if (!*a.d)
        {
            if ((e = cudaMalloc(a.d, a.bytes)) != cudaSuccess)
                return pfail(e, "steqn/buf malloc");
        }
    }
    return 0;
}


int rgpSTWeightsBegin(const double* phiInt)
{
    const int nf = gM.nIntFaces;
    if (!gSTM.wLin)
    {
        snprintf(gPErr, sizeof(gPErr), "steqn: ST mesh not uploaded");
        return -1;
    }
    cudaError_t e;
    if ((e = cudaMemcpy(gST.phi, phiInt, (size_t)nf*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "stw/H2D phi");
    rgppeqn::stLimInit<<<nb(nf), BS>>>(nf, gST.wLim);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "stw/init launch");
    return 0;
}


int rgpSTWeightsField(const double* psi, const double* bPsi)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
    cudaError_t e;
    if ((e = cudaMemcpy(gB.x, psi, (size_t)nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "stw/H2D psi");
    if (nbf > 0)
    {
        if ((e = cudaMemcpy(gST.bPsi, bPsi, (size_t)nbf*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "stw/H2D bPsi");
    }

    if ((e = cudaMemset(gST.grad, 0, (size_t)3*nc*sizeof(double)))
        != cudaSuccess) return pfail(e, "stw/grad memset");
    rgppeqn::stGradFace<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.sf, gB.x, gST.grad);
    if (nbf > 0)
    {
        rgppeqn::stGradBFace<<<nb(nbf), BS>>>
            (nbf, nc, gM.bfc, gSTM.bSf, gST.bPsi, gST.grad);
    }
    rgppeqn::stGradDivV<<<nb(nc), BS>>>(nc, gM.V, gST.grad);
    rgppeqn::stLimField<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.dvec, gST.phi, gB.x, gST.grad,
         gST.wLim);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "stw/field launch");
    return 0;
}


int rgpSTWeightsEnd(void)
{
    const int nf = gM.nIntFaces;
    rgppeqn::stWeightsEnd<<<nb(nf), BS>>>(nf, gSTM.wLin, gST.phi,
                                          gST.wLim);
    cudaError_t e;
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "stw/end launch");
    return 0;
}


int rgpSTEqnSolve
(
    double dtInv, const double* rdtCell, int hasSrc,
    const double* rho, const double* rhoOld, const double* psiOld,
    const double* p0,
    const double* phiInt, const double* gammaCell, const double* spCell,
    const double* srcCell, const double* bPsi,
    const double* bDiag, const double* bSrc,
    double tol, double relTol, int maxIter,
    double* psiOut,
    double* initRes, double* finalRes, int* nIters
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
    if (!gSTM.wLin)
    {
        snprintf(gPErr, sizeof(gPErr), "steqn: ST mesh not uploaded");
        return -1;
    }

    cudaError_t e;
    struct { double* d; const double* s; size_t n; } up[] =
    {
        {gST.rho, rho, (size_t)nc}, {gST.rhoOld, rhoOld, (size_t)nc},
        {gST.psiOld, psiOld, (size_t)nc}, {gB.x, p0, (size_t)nc},
        {gST.phi, phiInt, (size_t)nf}, {gST.gamma, gammaCell, (size_t)nc},
        {gST.sp, spCell, (size_t)nc},
        {gST.src, hasSrc ? srcCell : nullptr, (size_t)nc},
        {gST.bPsi, bPsi, (size_t)nbf},
        {gB.bDiag, bDiag, (size_t)nbf}, {gB.bSrc, bSrc, (size_t)nbf}
    };
    for (auto& u : up)
    {
        if (u.n == 0 || !u.s) continue;
        if ((e = cudaMemcpy(u.d, u.s, u.n*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "steqn/H2D");
    }

    // (가중치 gST.wLim은 rgpSTWeightsBegin/Field/End가 사전 준비)

    // ── 조립 ─────────────────────────────────────────────────────────
    const double* dRdt = stageRdt(rdtCell, &e);
    if (e != cudaSuccess) return pfail(e, "steqn/rdt");
    rgppeqn::stCellAssemble<<<nb(nc), BS>>>
        (nc, dtInv, dRdt, hasSrc, gST.rho, gST.rhoOld, gST.psiOld,
         gST.sp, gST.src, gM.V, gB.diag, gB.b);
    rgppeqn::stFaceAssemble<<<nb(nf), BS>>>
        (nf, gM.own, gM.nei, gST.phi, gST.wLim, gSTM.wLin, gST.gamma,
         gM.gg, gB.diag, gB.upper, gST.lower);
    if (nbf > 0)
    {
        rgppeqn::stBFaceAssemble<<<nb(nbf), BS>>>
            (nbf, gM.bfc, gB.bDiag, gB.bSrc, gB.diag, gB.b);
    }
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "steqn/assemble launch");

    int rc2 = bicgSolve
    (
        nc, nf, gB.diag, gB.upper, gST.lower, gB.b, gB.x,
        tol, relTol, maxIter, initRes, finalRes, nIters
    );
    if (rc2) return rc2;

    if ((e = cudaMemcpy(psiOut, gB.x, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "steqn/D2H psi");

    return 0;
}


int rgpSTEqnDump
(
    double* diagOut, double* upperOut, double* lowerOut, double* bOut,
    double* wOut
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;
    cudaError_t e;
    struct { double* h; const double* d; size_t n; } outs[] =
    {
        {diagOut, gB.diag, (size_t)nc}, {upperOut, gB.upper, (size_t)nf},
        {lowerOut, gST.lower, (size_t)nf}, {bOut, gB.b, (size_t)nc},
        {wOut, gST.wLim, (size_t)nf}
    };
    for (auto& o : outs)
    {
        if (!o.h) continue;
        if ((e = cudaMemcpy(o.h, o.d, o.n*sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "steqn/dump");
    }
    return 0;
}


//- UEqn prep 1/2: grad(U) 디바이스 계산 + 경계셀 gradU 채집(D2H).
//  UBuf 지연 할당도 여기서 (Solve보다 먼저 불리므로).
int rgpUEqnGrad
(
    const double* U3, const double* UB3, double* bGrad9
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
    if (!gSTM.wLin)
    {
        snprintf(gPErr, sizeof(gPErr), "ueqn: ST mesh not uploaded");
        return -1;
    }

    cudaError_t e;
    {
        const size_t bc = (size_t)nc*sizeof(double);
        const size_t bf = (size_t)nf*sizeof(double);
        const size_t nb1 = (size_t)(nbf > 0 ? nbf : 1);
        struct { double** d; size_t bytes; } al[] =
        {
            {&gU.diag, bc}, {&gU.upper, bf}, {&gU.lower, bf},
            {&gU.b3, 3*bc}, {&gU.U3, 3*bc},
            {&gU.bDiag3, 3*nb1*sizeof(double)},
            {&gU.bSrc3, 3*nb1*sizeof(double)},
            {&gU.workDiag, bc}, {&gU.workB, bc}, {&gU.H, bc},
            {&gU.grad9, 9*bc}, {&gU.src3, 3*bc}, {&gU.gradP, 3*bc},
            {&gU.UB3, 3*nb1*sizeof(double)},
            {&gU.bfx3, 3*nb1*sizeof(double)},
            {&gU.bg9, 9*nb1*sizeof(double)}
        };
        for (auto& a : al)
        {
            if (*a.d) continue;
            if ((e = cudaMalloc(a.d, a.bytes)) != cudaSuccess)
                return pfail(e, "ueqn/malloc");
            // Grad가 먼저 할당해도 (Prep2 없는) 독립 rgpUEqnSolve 호출이
            // 쓰레기 src3/gradP를 읽지 않도록 여기서도 0-초기화
            if ((e = cudaMemset(*a.d, 0, a.bytes)) != cudaSuccess)
                return pfail(e, "ueqn/memset");
        }
    }

    if ((e = cudaMemcpy(gU.U3, U3, (size_t)3*nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "ueqnGrad/H2D U");
    if (nbf > 0)
    {
        if ((e = cudaMemcpy(gU.UB3, UB3, (size_t)3*nbf*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "ueqnGrad/H2D UB");
    }

    if ((e = cudaMemset(gU.grad9, 0, (size_t)9*nc*sizeof(double)))
        != cudaSuccess) return pfail(e, "ueqnGrad/memset");
    rgppeqn::uVecGradFace<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.sf, gU.U3, gU.grad9);
    if (nbf > 0)
    {
        rgppeqn::uVecGradBnd<<<nb(nbf), BS>>>
            (nbf, nc, gM.bfc, gSTM.bSf, gU.UB3, gU.grad9);
    }
    rgppeqn::uGrad9DivV<<<nb(nc), BS>>>(nc, gM.V, gU.grad9);
    if (nbf > 0)
    {
        rgppeqn::uGradGather<<<nb(nbf), BS>>>
            (nbf, nc, gM.bfc, gU.grad9, gU.bg9);
    }
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "ueqnGrad/launch");

    if (nbf > 0)
    {
        if ((e = cudaMemcpy(bGrad9, gU.bg9,
                            (size_t)9*nbf*sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "ueqnGrad/D2H bg9");
    }
    return 0;
}


//- UEqn prep 2/2: limitedLinearV 가중치 + 명시항(div dev2, 경계 기여
//  bndFlux3 포함, per-volume) + grad(p) — 모두 디바이스 상주.
int rgpUEqnPrep2
(
    const double* pHost, const double* pB,
    const double* mu, const double* phiInt,
    const double* bndFlux3
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;

    cudaError_t e;
    struct { double* d; const double* s; size_t n; } up[] =
    {
        {gB.pOld, pHost, (size_t)nc}, {gST.bPsi, pB, (size_t)nbf},
        {gST.gamma, mu, (size_t)nc}, {gST.phi, phiInt, (size_t)nf},
        {gU.bfx3, bndFlux3, (size_t)3*nbf}
    };
    for (auto& u : up)
    {
        if (u.n == 0 || !u.s) continue;
        if ((e = cudaMemcpy(u.d, u.s, u.n*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "ueqnPrep/H2D");
    }

    // 가중치 (내부면; 경계 가중치는 BC 계수에서 미사용 — 호스트 참조)
    rgppeqn::uLimVWeights<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.dvec, gST.phi,
         gU.U3, gU.grad9, gST.wLim);

    // 명시항: div(mu*dev2(T(gradU))) — 내부 플럭스 + 경계 기여 + /V
    if ((e = cudaMemset(gU.src3, 0, (size_t)3*nc*sizeof(double)))
        != cudaSuccess) return pfail(e, "ueqnPrep/memset src");
    rgppeqn::uDevTauFlux<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.sf, gST.gamma,
         gU.grad9, gU.src3);
    if (nbf > 0 && bndFlux3)
    {
        rgppeqn::uBndSrcAdd<<<nb(nbf), BS>>>
            (nbf, nc, gM.bfc, gU.bfx3, gU.src3);
    }
    rgppeqn::uSrc3DivV<<<nb(nc), BS>>>(nc, gM.V, gU.src3);

    // grad(p): 스칼라 Gauss linear (경계 값 포함)
    if ((e = cudaMemset(gU.gradP, 0, (size_t)3*nc*sizeof(double)))
        != cudaSuccess) return pfail(e, "ueqnPrep/memset gp");
    rgppeqn::stGradFace<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.sf, gB.pOld, gU.gradP);
    if (nbf > 0)
    {
        rgppeqn::stGradBFace<<<nb(nbf), BS>>>
            (nbf, nc, gM.bfc, gSTM.bSf, gST.bPsi, gU.gradP);
    }
    rgppeqn::stGradDivV<<<nb(nc), BS>>>(nc, gM.V, gU.gradP);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "ueqnPrep/launch");

    return 0;
}


int rgpUEqnSolve
(
    double dtInv, const double* rdtCell,
    const double* rho, const double* rhoOld,
    const double* U3old, const double* U3,
    const double* phiInt, const double* w, const double* mu,
    const double* srcExp3, const double* srcExtra3,
    const double* gradP3,
    const double* bDiag3, const double* bSrc3,
    const int* solveCmpt,
    double tol, double relTol, int maxIter,
    double* U3out, double* initRes3, double* finalRes3, int* iters3
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
    if (!gSTM.wLin)
    {
        snprintf(gPErr, sizeof(gPErr), "ueqn: ST mesh not uploaded");
        return -1;
    }

    cudaError_t e;

    // 지연 할당 (UEqn 퍼시스턴트) — 독립 호출(Grad/Prep2 없이)도
    // 안전하도록 src3/gradP까지 포함하고 0으로 초기화
    if (!gU.diag)
    {
        const size_t bc = (size_t)nc*sizeof(double);
        const size_t bf = (size_t)nf*sizeof(double);
        const size_t bb = (size_t)3*(nbf > 0 ? nbf : 1)*sizeof(double);
        struct { double** d; size_t bytes; } al[] =
        {
            {&gU.diag, bc}, {&gU.upper, bf}, {&gU.lower, bf},
            {&gU.b3, 3*bc}, {&gU.U3, 3*bc},
            {&gU.bDiag3, bb}, {&gU.bSrc3, bb},
            {&gU.workDiag, bc}, {&gU.workB, bc}, {&gU.H, bc},
            {&gU.src3, 3*bc}, {&gU.gradP, 3*bc}
        };
        for (auto& a : al)
        {
            if (*a.d) continue;
            if ((e = cudaMalloc(a.d, a.bytes)) != cudaSuccess)
                return pfail(e, "ueqn/malloc");
            if ((e = cudaMemset(*a.d, 0, a.bytes)) != cudaSuccess)
                return pfail(e, "ueqn/memset");
        }
    }

    // 입력 업로드. w/srcExp3/gradP3가 null이면 rgpUEqnGrad/Prep2가
    // 준비한 디바이스 상주본(gST.wLim/gU.src3/gU.gradP) 사용.
    struct { double* d; const double* s; size_t n; } up[] =
    {
        {gST.rho, rho, (size_t)nc}, {gST.rhoOld, rhoOld, (size_t)nc},
        {gST.phi, phiInt, phiInt ? (size_t)nf : 0},
        {gST.wLim, w, w ? (size_t)nf : 0},
        {gST.gamma, mu, mu ? (size_t)nc : 0},
        {gU.U3, U3, U3 ? (size_t)3*nc : 0},
        {gU.bDiag3, bDiag3, (size_t)3*nbf},
        {gU.bSrc3, bSrc3, (size_t)3*nbf}
    };
    for (auto& u : up)
    {
        if (u.n == 0 || !u.s) continue;
        if ((e = cudaMemcpy(u.d, u.s, u.n*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/H2D");
    }
    // U3old는 gST.grad 스테이징, srcExp3는 gU.b3(셀 커널이 in-place로
    // b3 완성 — 셀별 독립이라 안전); null이면 gU.src3 → gU.b3 복사.
    if ((e = cudaMemcpy(gST.grad, U3old, (size_t)3*nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "ueqn/H2D U3old");
    if (srcExp3)
    {
        if ((e = cudaMemcpy(gU.b3, srcExp3, (size_t)3*nc*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/H2D src");
    }
    else
    {
        if ((e = cudaMemcpy(gU.b3, gU.src3, (size_t)3*nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/D2D src");
    }

    // ── 조립 ─────────────────────────────────────────────────────────
    const double* dRdt = stageRdt(rdtCell, &e);
    if (e != cudaSuccess) return pfail(e, "ueqn/rdt");
    rgppeqn::uCellAssemble<<<nb(nc), BS>>>
        (nc, dtInv, dRdt, gST.rho, gST.rhoOld, gM.V, gST.grad, gU.b3,
         gU.diag, gU.b3);
    if (srcExtra3)   // LAD-bulk 등 명시항 (저장 소스 — H()에 포함)
    {
        // gST.grad 재사용: 위 uCellAssemble(gST.grad=U3old 읽음)와 이
        // H2D가 모두 legacy NULL 스트림이라 스트림 순서가 보장된다 —
        // 커널을 비-디폴트 스트림으로 옮기면 이 재사용부터 깨진다
        const double* dX =
            rgpInPtr(srcExtra3, gST.grad, (size_t)3*nc, &e);
        if (e != cudaSuccess) return pfail(e, "ueqn/srcExtra");
        for (int c = 0; c < 3; c++)
        {
            rgppeqn::uAddVolSrc<<<nb(nc), BS>>>
                (nc, 1.0, dX + (size_t)c*nc, gM.V,
                 gU.b3 + (size_t)c*nc);
        }
    }
    rgppeqn::stFaceAssemble<<<nb(nf), BS>>>
        (nf, gM.own, gM.nei, gST.phi, gST.wLim, gSTM.wLin, gST.gamma,
         gM.gg, gU.diag, gU.upper, gU.lower);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "ueqn/assemble launch");
    gU.assembled = true;

    // grad p (솔브 전용 소스) — null이면 Prep2의 디바이스 상주본 사용
    if (gradP3)
    {
        if ((e = cudaMemcpy(gU.gradP, gradP3,
                            (size_t)3*nc*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/H2D gradP");
    }

    // ── 성분별 솔브 ──────────────────────────────────────────────────
    for (int c = 0; c < 3; c++)
    {
        initRes3[c] = 0; finalRes3[c] = 0; iters3[c] = 0;
        if (!solveCmpt[c]) continue;

        if ((e = cudaMemcpy(gU.workDiag, gU.diag,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/workDiag");
        if ((e = cudaMemcpy(gU.workB, gU.b3 + (size_t)c*nc,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/workB");
        if (nbf > 0)
        {
            rgppeqn::stBFaceAssemble<<<nb(nbf), BS>>>
                (nbf, gM.bfc, gU.bDiag3 + (size_t)c*nbf,
                 gU.bSrc3 + (size_t)c*nbf, gU.workDiag, gU.workB);
        }
        rgppeqn::uAddVolSrc<<<nb(nc), BS>>>
            (nc, -1.0, gU.gradP + (size_t)c*nc, gM.V, gU.workB);

        if ((e = cudaMemcpy(gB.x, gU.U3 + (size_t)c*nc,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/x0");

        int rc = bicgSolve
        (
            nc, nf, gU.workDiag, gU.upper, gU.lower, gU.workB, gB.x,
            tol, relTol, maxIter,
            &initRes3[c], &finalRes3[c], &iters3[c]
        );
        if (rc) return rc;

        if ((e = cudaMemcpy(U3out + (size_t)c*nc, gB.x,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "ueqn/D2H U");
        // 솔브 결과를 gU.U3에도 반영 (H() 계산은 최신 U 사용이 원칙이나
        // pCorr가 매번 U를 새로 업로드하므로 여기선 참고용)
        if ((e = cudaMemcpy(gU.U3 + (size_t)c*nc, gB.x,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "ueqn/U3 sync");
    }

    return 0;
}


int rgpUEqnAH(const double* U3, double* rAU, double* H3)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;
    if (!gU.diag || !gU.assembled)
    {
        snprintf(gPErr, sizeof(gPErr), "ueqn: not assembled");
        return -1;
    }

    cudaError_t e;
    if ((e = cudaMemcpy(gU.U3, U3, (size_t)3*nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "ueqnAH/H2D U");

    // rAU = V/D, D = diag + cmptAv(iC)
    if ((e = cudaMemcpy(gU.workDiag, gU.diag, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToDevice)) != cudaSuccess)
        return pfail(e, "ueqnAH/D2D");
    if (nbf > 0)
    {
        rgppeqn::uAvIC<<<nb(nbf), BS>>>(nbf, gM.bfc, gU.bDiag3,
                                        gU.workDiag);
    }
    rgppeqn::uRAU<<<nb(nc), BS>>>(nc, gM.V, gU.workDiag, gU.workB);
    if ((e = cudaMemcpy(rAU, gU.workB, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "ueqnAH/D2H rAU");
    if (!gU.rAU)
    {
        if ((e = cudaMalloc(&gU.rAU, (size_t)nc*sizeof(double)))
            != cudaSuccess) return pfail(e, "ueqnAH/malloc rAU");
        if ((e = cudaMalloc(&gU.HbyA3, (size_t)3*nc*sizeof(double)))
            != cudaSuccess) return pfail(e, "ueqnAH/malloc HbyA");
    }
    if ((e = cudaMemcpy(gU.rAU, gU.workB, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToDevice)) != cudaSuccess)
        return pfail(e, "ueqnAH/rAU D2D");

    // H(U) 성분별
    for (int c = 0; c < 3; c++)
    {
        if ((e = cudaMemcpy(gU.H, gU.b3 + (size_t)c*nc,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToDevice)) != cudaSuccess)
            return pfail(e, "ueqnAH/H init");
        rgppeqn::uHFace<<<nb(nf), BS>>>
            (nf, gM.own, gM.nei, gU.upper, gU.lower,
             gU.U3 + (size_t)c*nc, gU.H);
        if (nbf > 0)
        {
            rgppeqn::uHBnd<<<nb(nbf), BS>>>
                (nbf, nc, c, gM.bfc, gU.bDiag3, gU.bSrc3, gU.U3, gU.H);
        }
        rgppeqn::uDivV<<<nb(nc), BS>>>(nc, gM.V, gU.H);
        rgppeqn::uHbyA<<<nb(nc), BS>>>(nc, c, gU.rAU, gU.H, gU.HbyA3);
        if ((e = cudaGetLastError()) != cudaSuccess)
            return pfail(e, "ueqnAH/launch");
        if ((e = cudaMemcpy(H3 + (size_t)c*nc, gU.H,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyDeviceToHost)) != cudaSuccess)
            return pfail(e, "ueqnAH/D2H H");
    }
    return 0;
}


//- pCorr 준비체인: rhof/rAUf(조화)/phiHbyAv/psis 디바이스 상주 생성.
//  rAU/HbyA는 직전 rgpUEqnAH의 디바이스 사본 사용. ddtCorr는 OF
//  EulerDdt fvcDdtPhiCorr(rho,U,phi) 1:1 (경계 기여 0은 호스트 담당).
int rgpPCorrPrep
(
    const double* rho, const double* rhoOld,
    const double* Uold3, const double* phiOld, const double* psi,
    const double* psisOverride,
    double rDeltaT, const double* rdtCell, double rcDdtScale
)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;
    if (!gU.rAU)
    {
        snprintf(gPErr, sizeof(gPErr), "pcorr: rgpUEqnAH not run");
        return -1;
    }

    cudaError_t e;
    if (!gPC.rAUf)
    {
        const size_t bc = (size_t)nc*sizeof(double);
        const size_t bf = (size_t)nf*sizeof(double);
        struct { double** d; size_t bytes; } al[] =
        {
            {&gPC.rAUf, bf}, {&gPC.phiH, bf}, {&gPC.rhof, bf},
            {&gPC.psis, bc}, {&gPC.rhoOld, bc}, {&gPC.Uold3, 3*bc},
            {&gPC.phiOld, bf}
        };
        for (auto& a : al)
        {
            if ((e = cudaMalloc(a.d, a.bytes)) != cudaSuccess)
                return pfail(e, "pcorr/malloc");
        }
    }

    struct { double* d; const double* s; size_t n; } up[] =
    {
        {gST.rho, rho, (size_t)nc}, {gPC.rhoOld, rhoOld, (size_t)nc},
        {gPC.Uold3, Uold3, (size_t)3*nc}, {gPC.phiOld, phiOld, (size_t)nf},
        {gST.gamma, psi, (size_t)nc}
    };
    for (auto& u : up)
    {
        if ((e = cudaMemcpy(u.d, u.s, u.n*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "pcorr/H2D");
    }

    const double* dRdt = stageRdt(rdtCell, &e);
    if (e != cudaSuccess) return pfail(e, "pcorr/rdt");
    rgppeqn::pcPrepFace<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.sf,
         gST.rho, gU.rAU, gU.HbyA3, gPC.rhoOld, gPC.Uold3, gPC.phiOld,
         rDeltaT, dRdt, rcDdtScale, gPC.rhof, gPC.rAUf, gPC.phiH);
    if (psisOverride)
    {
        // psisCapRatio/psisIsentropic: 호스트 계산본 업로드 (정확 경로)
        if ((e = cudaMemcpy(gPC.psis, psisOverride,
                            (size_t)nc*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "pcorr/psis override");
    }
    else
    {
        rgppeqn::pcPsis<<<nb(nc), BS>>>(nc, gST.gamma, gST.rho,
                                        gPC.psis);
    }
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "pcorr/launch");
    return 0;
}


//- 솔브 후처리 v2: 플럭스 → phi = rhof*(phiHbyAv + flux)까지 디바이스
int rgpPEqnFinish2(double* pOut, double* phiOut)
{
    const int nc = gM.nCells, nf = gM.nIntFaces;

    rgppeqn::faceFlux<<<nb(nf), BS>>>(nf, gM.own, gM.nei, gB.upper,
                                      gB.x, gB.flux);
    rgppeqn::pcPhiRecon<<<nb(nf), BS>>>(nf, gPC.rhof, gPC.phiH,
                                        gB.flux, gB.flux);
    cudaError_t e;
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "pcorr/finish launch");

    if ((e = cudaMemcpy(pOut, gB.x, (size_t)nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "pcorr/D2H p");
    if ((e = cudaMemcpy(phiOut, gB.flux, (size_t)nf*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "pcorr/D2H phi");
    return 0;
}


//- 후처리: U = HbyA − rAU*grad(p_new) (grad p 디바이스 계산)
int rgpPCorrU(const double* pNew, const double* pB, double* U3out)
{
    const int nc = gM.nCells, nf = gM.nIntFaces, nbf = gM.nBFaces;

    cudaError_t e;
    if ((e = cudaMemcpy(gB.pOld, pNew, (size_t)nc*sizeof(double),
                        cudaMemcpyHostToDevice)) != cudaSuccess)
        return pfail(e, "pcorrU/H2D p");
    if (nbf > 0)
    {
        if ((e = cudaMemcpy(gST.bPsi, pB, (size_t)nbf*sizeof(double),
                            cudaMemcpyHostToDevice)) != cudaSuccess)
            return pfail(e, "pcorrU/H2D pB");
    }

    if ((e = cudaMemset(gU.gradP, 0, (size_t)3*nc*sizeof(double)))
        != cudaSuccess) return pfail(e, "pcorrU/memset");
    rgppeqn::stGradFace<<<nb(nf), BS>>>
        (nf, nc, gM.own, gM.nei, gSTM.wLin, gSTM.sf, gB.pOld, gU.gradP);
    if (nbf > 0)
    {
        rgppeqn::stGradBFace<<<nb(nbf), BS>>>
            (nbf, nc, gM.bfc, gSTM.bSf, gST.bPsi, gU.gradP);
    }
    rgppeqn::stGradDivV<<<nb(nc), BS>>>(nc, gM.V, gU.gradP);

    rgppeqn::pcUUpdate<<<nb(nc), BS>>>
        (nc, gU.HbyA3, gU.rAU, gU.gradP, gU.src3);
    if ((e = cudaGetLastError()) != cudaSuccess)
        return pfail(e, "pcorrU/launch");

    if ((e = cudaMemcpy(U3out, gU.src3, (size_t)3*nc*sizeof(double),
                        cudaMemcpyDeviceToHost)) != cudaSuccess)
        return pfail(e, "pcorrU/D2H U");
    return 0;
}


void rgpPEqnFree(void) { pFreeAll(); }

const char* rgpPEqnLastError(void) { return gPErr; }

} // extern "C"

// ************************************************************************* //
