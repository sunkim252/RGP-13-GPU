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
    double dtInv,
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
    double dtInv,
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
    double dtInv,
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

void rgpPEqnFree(void);
const char* rgpPEqnLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
