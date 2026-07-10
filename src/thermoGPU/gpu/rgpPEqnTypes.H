/*---------------------------------------------------------------------------*\
  RGP-13 pEqnGPU — host/device shared interface (pure C ABI)

  fgmFluid correctPressurePEP의 압력 방정식
      psis*ddt(p) + div(phiHbyAv) - laplacian(rAUf, p) = 0
  을 fvMatrix를 우회해 GPU에서 조립·솔브(Jacobi-PCG, OF 잔차 규약)·
  플럭스 추출까지 수행한다. 직교(orthogonal) laplacian 전용 —
  'Gauss linear orthogonal' + 'snGrad orthogonal' 스킴과 1:1.
  경계 기여(bDiag/bSrc)는 호스트가 fvPatchField API로 정확 계산해
  패치-연접 배열로 내린다. 규칙: OpenFOAM/CUDA 헤더 include 금지.
\*---------------------------------------------------------------------------*/

#ifndef rgpPEqnTypes_H
#define rgpPEqnTypes_H

#ifdef __cplusplus
extern "C" {
#endif

/*  정적 메시 업로드(1회): 내부면 owner/neighbour, 면 기하계수
    gammaGeom = magSf*deltaCoeffs, 셀 부피 V, 경계면 faceCells(패치 연접).
    재호출 시 교체(AMR 대응).                                          */
int rgpPEqnMeshUpload
(
    int nCells, int nIntFaces,
    const int* owner, const int* neigh,
    const double* gammaGeom,
    const double* V,
    int nBFaces, const int* bFaceCells
);

/*  조립+솔브+플럭스 (모두 디바이스 상주):
      diag_c = psis_c V_c dtInv + Σ_f Γ_f + bDiag,  Γ_f = rAUfInt_f gammaGeom_f
      upper_f = -Γ_f (대칭)
      b_c = psis_c V_c dtInv pOld_c - Σ_f ± phiInt_f - Σ_bf phiB + bSrc
      needRef: OF setReference 규약 (b[ref] += diag[ref]*refValue; diag[ref] *= 2)
    솔버: Jacobi-PCG, OF normFactor/수렴 규약(res<tol || res<=relTol*res0).
    출력: pOut[nCells], fluxInt[nIntFaces] = Γ_f (p_o - p_n).
    경계 플럭스는 호스트에서 iC*p[fc] - bC 로 계산할 것.               */
int rgpPEqnSolve
(
    double dtInv, const double* rdtCell,
    const double* rAUfInt,
    const double* psis, const double* pOld, const double* p0,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double tol, double relTol, int maxIter,
    double* pOut, double* fluxInt,
    double* initRes, double* finalRes, int* nIters
);

/*  검증용: 조립만 수행하고 diag(경계 포함)/upper/b(경계 포함)를 D2H.
    CPU fvMatrix 계수와의 비트-수준 대조에 사용.                       */
int rgpPEqnAssembleDump
(
    double dtInv, const double* rdtCell,
    const double* rAUfInt,
    const double* psis, const double* pOld,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double* diagOut, double* upperOut, double* bOut
);

/*  AmgX 직결 (③): CSR 구조를 1회 생성하고, 조립 결과를 디바이스 CSR
    값으로 산포. 솔브는 외부(libRGP13amgx의 rgpPEqnAmgxSolve)가 디바이스
    포인터로 수행하고, 잔차 확인·플럭스·회수는 여기서.                 */
int rgpPEqnCsrPrepare(int* nnzOut);
const int* rgpPEqnCsrRowPtr(void);   /* [nCells+1] 호스트 */
const int* rgpPEqnCsrColInd(void);   /* [nnz] 호스트 */
void* rgpPEqnDevValues(void);        /* [nnz] 디바이스 */
void* rgpPEqnDevB(void);             /* [nCells] 디바이스 */
void* rgpPEqnDevX(void);             /* [nCells] 디바이스 (x0=p, 솔브 후 p) */

int rgpPEqnAssembleCsr
(
    double dtInv, const double* rdtCell,
    const double* rAUfInt,
    const double* psis, const double* pOld, const double* p0,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double* normFactor, double* initRes
);

/*  현재 x의 OF 규약 잔차 Σ|b-Ax|/normFactor */
int rgpPEqnResidual(double normFactor, double* res);

/*  솔브 후처리: faceH 플럭스 + p/flux D2H */
int rgpPEqnFinish(double* pOut, double* fluxInt);

/*  스칼라 수송(Z/C) GPU: fvm::ddt(rho,ψ) + fvm::div(phi,ψ)[Gauss
    limitedLinear 1] − fvm::Sp(sp,ψ) − fvm::laplacian(γ,ψ) [+ src]
    를 디바이스 조립하고 Jacobi-BiCGStab(OF PBiCGStab 구조/잔차 규약)
    으로 솔브. 경계 기여(bDiag/bSrc = div의 vIC/vBC + −laplacian의
    gic/gbc 합산)는 호스트가 fvPatchField API로 계산.
    rgpPEqnMeshUpload 이후 1회 rgpSTEqnMeshUpload(선형가중치, Sf,
    d = C_n−C_o, 경계 Sf; 모두 SoA [3][n]) 필요.                       */
int rgpSTEqnMeshUpload
(
    const double* wLin, const double* Sf, const double* dVec,
    const double* bSf
);

/*  multivariate limitedLinear 가중치: Begin(phi) → Field(각 필드 —
    fields 테이블의 he, Z, C[, h, W] 순서로 limiter min-누적) → End.
    mvConvection 생성 시점(manifold 갱신 전) 필드 값으로 호출할 것.     */
int rgpSTWeightsBegin(const double* phiInt);
int rgpSTWeightsField(const double* psi, const double* bPsi);
int rgpSTWeightsEnd(void);

/*  UEqn(벡터, 성분 공유 LDU): ddt(rho,U) + div(phi,U;w) − laplacian(mu,U)
    를 디바이스 조립(가중치 w·mu·명시소스 srcExp3=(divDevTau 명시항−grad p)
    는 호스트가 OF 스킴으로 정확 계산해 업로드), 유효 성분별
    Jacobi-BiCGStab 솔브. 행렬은 스텝 내내 디바이스 상주 —
    rgpUEqnAH가 pCorr용 rAU=1/A()와 H(U)를 산출(fvMatrix::D/H 1:1).
    SoA 규약: [3*n], 경계 배열 [3*nbf].                                */
/*  UEqn prep (v2): 물리 필드도 GPU —
    Grad: U 업로드 → grad(U) 디바이스 + 경계셀 gradU 채집(D2H, 호스트가
    경계 보정·명시항 X_b 계산용). Prep2: limitedLinearV 가중치(NVDVTVDV
    1:1) + div(mu*dev2(T(gradU)))(경계 기여 bndFlux3 산입) + grad(p).
    이후 rgpUEqnSolve에 w/mu/srcExp3/gradP3/U3/phiInt를 null로 넘기면
    디바이스 상주본 사용.                                              */
int rgpUEqnGrad(const double* U3, const double* UB3, double* bGrad9);
int rgpUEqnPrep2
(
    const double* pHost, const double* pB,
    const double* mu, const double* phiInt,
    const double* bndFlux3
);

int rgpUEqnSolve
(
    double dtInv, const double* rdtCell,
    const double* rho, const double* rhoOld,
    const double* U3old, const double* U3,
    const double* phiInt, const double* w, const double* mu,
    const double* srcExp3, const double* gradP3,
    const double* bDiag3, const double* bSrc3,
    const int* solveCmpt,
    double tol, double relTol, int maxIter,
    double* U3out, double* initRes3, double* finalRes3, int* iters3
);

int rgpUEqnAH(const double* U3, double* rAU, double* H3);

/*  pCorr 준비체인 (gpuUEqn+gpuPEqn 디바이스 체인 전용):
    Prep — rhof/rAUf(조화)/phiHbyAv(flux(HbyA)+Euler ddtCorr)/psis를
    디바이스 상주 생성(rAU/HbyA는 직전 rgpUEqnAH 사본). 이후
    rgpPEqnSolve/AssembleCsr에 rAUfInt/psis/phiInt를 null로 넘기면
    상주본 사용. Finish2 — 플럭스+phi 재구성까지 디바이스.
    U — U = HbyA − rAU∇p(신규 p) 후처리.                                */
int rgpPCorrPrep
(
    const double* rho, const double* rhoOld,
    const double* Uold3, const double* phiOld, const double* psi,
    double rDeltaT, const double* rdtCell, double rcDdtScale
);
int rgpPEqnFinish2(double* pOut, double* phiOut);
int rgpPCorrU(const double* pNew, const double* pB, double* U3out);

/*  검증용: 마지막 rgpSTEqnSolve의 조립 계수/가중치 D2H (null 허용) */
int rgpSTEqnDump
(
    double* diagOut, double* upperOut, double* lowerOut, double* bOut,
    double* wOut
);

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
);

void rgpPEqnFree(void);
const char* rgpPEqnLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
