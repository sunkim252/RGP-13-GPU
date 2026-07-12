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
    const double* srcCellExtra,
    const double* rAUfInt,
    const double* psis, const double* pOld, const double* p0,
    const double* phiInt, const double* phiB,
    const double* bDiag, const double* bSrc,
    int needRef, int refCell, double refValue,
    double tol, double relTol, int maxIter,
    double* pOut, double* fluxInt,
    double* initRes, double* finalRes, int* nIters
);

/*  병렬 pEqn (pcg 전용): processor 패치의 halo 교환 + 전역 리덕션.
    ParArm은 메시 아밍 시 1회 — nbrRank/nFaces[nProcPatches],
    faceCells는 패치-연접, globalCells는 전역 셀 수(xRef용).
    이 랭크에 processor 패치가 없어도(0, NULL 허용) 호출해야
    전역 리덕션 collective가 성립한다.
    ParCoeffs는 솔브마다 — 인터페이스 계수 bC(=+pGamma*gbc, OF
    boundaryCoeffs 규약; SpMV가 y[fc] -= bC*x_nbr).
    경계 concat(bDiag/bSrc/phiB)에서 processor 패치는 bDiag=-pGamma*gic,
    bSrc=0, phiB=phiHbyAv_b 로 채울 것.                                */
int rgpPEqnParArm
(
    int nProcPatches, const int* nbrRank, const int* nFaces,
    const int* faceCells, double globalCells
);
int rgpPEqnParCoeffs(const double* bCoeffs);

/*  pEqn PCG 전처리기 선택: 0=Jacobi(기본), 1=멀티컬러 DIC.
    DIC 구조(그리디 셀 컬러링 + 면 fwd/bwd 그룹)는 첫 솔브에서 지연
    아밍되고 메시 재아밍 시 해제·재구축된다. rD는 솔브마다 재계산.
    rgpPEqnPreconColors()는 아밍된 컬러 수(미아밍 0)를 반환.           */
int rgpPEqnSetPrecon(int mode);
int rgpPEqnPreconColors(void);

/*  검증용: 조립만 수행하고 diag(경계 포함)/upper/b(경계 포함)를 D2H.
    CPU fvMatrix 계수와의 비트-수준 대조에 사용.                       */
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
    const double* srcCellExtra,
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

/*  병렬: 직전 rgpSTWeightsField의 셀 grad를 processor 면 셀에서 게더해
    D2H (out3[3*totF] SoA). 호스트가 이웃 랭크와 교환해 coupled 경계
    리미터(LimitedScheme.C의 coupled 분기 1:1)를 계산하는 데 쓴다.
    병렬 미아밍/proc 면 0이면 no-op. ZC 솔브의 인터페이스 계수는
    rgpPEqnParCoeffs로 (방정식마다 gamma가 달라 솔브 직전 갱신).       */
int rgpSTGradAtPar(double* out3);
int rgpSTGradDump(double* out3);   /* 진단: 셀 grad [3*nc] D2H */
int rgpSTWeightsSet(const double* w);   /* 호스트 가중치 직업로드 [nf] */

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

/*  relaxAlpha > 0 이면 fvMatrix::relax(alpha) 1:1을 디바이스에서 수행
    (조립 후·솔브 전): D=max(|D+Σadd_b|, Σ|off|+Σsoff_b)/α − Σrem_b,
    S += (D−D0)·ψ현재. bRelax3 = [add|rem|soff] 각 [nbf] 연접 —
    uncoupled: add=cmptMax(cmptMag(iC)), rem=cmptMin(iC), soff=0;
    coupled(processor): add=rem=iC성분0, soff=|bC|.                   */
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
    double relaxAlpha, const double* bRelax3,
    double tol, double relTol, int maxIter,
    double* U3out, double* initRes3, double* finalRes3, int* iters3
);

/*  병렬: UEqn 인터페이스 계수 업로드 (성분 공유 스칼라, proc-면 연접,
    momentumPredictor가 조립 직전 호출 — 별도 디바이스 버퍼라 pEqn/ZC의
    rgpPEqnParCoeffs와 독립). rgpUEqnAH의 UNbr3는 fvMatrix::H()의
    coupled addBoundarySource(H += bC*U_nbr)용 이웃 U (SoA [3*totF],
    직렬이면 NULL).                                                    */
int rgpUEqnParCoeffs(const double* bCoeffs);
int rgpUEqnAH(const double* U3, const double* UNbr3,
              double* rAU, double* H3);

/*  pCorr 준비체인 (gpuUEqn+gpuPEqn 디바이스 체인 전용):
    Prep — rhof/rAUf(조화)/phiHbyAv(flux(HbyA)+Euler ddtCorr)/psis를
    디바이스 상주 생성(rAU/HbyA는 직전 rgpUEqnAH 사본). 이후
    rgpPEqnSolve/AssembleCsr에 rAUfInt/psis/phiInt를 null로 넘기면
    상주본 사용. Finish2 — 플럭스+phi 재구성까지 디바이스.
    U — U = HbyA − rAU∇p(신규 p) 후처리.                                */
/*  devChain 버퍼 프리플라이트 — 실패 시 부분 할당 회수+오류 소거.
    호스트는 실패 시 devChain을 강등(CPU-prep 경로)할 수 있다.        */
int rgpPCorrEnsure(void);

int rgpPCorrPrep
(
    const double* rho, const double* rhoOld,
    const double* Uold3, const double* phiOld, const double* psi,
    const double* psisOverride,
    double rDeltaT, const double* rdtCell, double rcDdtScale
);
int rgpPEqnFinish2(double* pOut, double* phiOut);
int rgpPCorrU(const double* pNew, const double* pB, double* U3out);

/*  fvc::div(phi, q) — 직전 rgpSTWeights*가 준비한 multivariate
    가중치로 내부면·경계면 flux-div, 결과는 per-volume [nc] (D2H).
    mvConvection->fvcDiv(phi, q)와 1:1 (결합 min-리미터는 임의 필드에
    적용됨 — EEqn he==e의 pressureWork용). coupled(proc) 면은 bQ/bPhi
    슬롯을 0으로 두고 호스트가 procW로 직접 가산할 것.               */
int rgpSTFluxDiv
(
    const double* q, const double* bQ, const double* phiInt,
    const double* bPhi, double* divOut
);

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
    const double* phiInt, const double* gammaCell,
    const double* gammaFace,   /* [nf] Fickian류: CPU-추출 면 계수
                                  Γ_f(=-divj.upper) 직접 주입; null이면
                                  gammaCell 셀 필드의 선형 면 보간 */
    const double* spCell,
    const double* srcCell, const double* bPsi,
    const double* bDiag, const double* bSrc,
    double relaxAlpha,         /* >0 이면 fvMatrix::relax(alpha) 1:1을
                                  경계 폴드 전(=CPU의 iC 분리 상태)에
                                  디바이스 수행. bRelaxA = [add|rem|soff]
                                  각 [nbf] 연접 — uncoupled: |iC|/iC/0,
                                  coupled: iC/iC/|bC| (호스트 계산).
                                  S += (D−D0)·ψ현재(p0).               */
    const double* bRelaxA,
    double tol, double relTol, int maxIter,
    double* psiOut,
    double* initRes, double* finalRes, int* nIters
);

/*  비직교 보정 소스 디바이스화 (limited/corrected snGrad 전용, 직렬 v1;
    호출자는 스킴 파싱·경계 corr_b 계산·강등 폴백을 담당):
    Arm: dcsNO=[nf] nonOrthDeltaCoeffs 상주. mdK >= 0 이면 grad(p)에
         cellMDLimited(k=mdK) 리미터를 결정론 CSR 재생으로 적용 —
         C3=[3*nc]·CfB3=[3*nbf] 상주 (CPU 식 dcf=Cf−C 그대로 — 파생식은
         ±ulp로 이산 분기 플립). mdK < 0 또는 k<small → 순수 Gauss
         linear. kf3/Cf3 [3*nf]는 상주하지 않고 Prep가 패스별 스테이징
         (UEqn grad9 별칭 — VRAM 순증 ~수십 MB).
         VRAM 부족 시 -2 (호스트 경로 강등).
    Prep: p 셀값·경계 면값(linear interp 규약)·gMsf=rAUf*magSf 업로드
         → grad(결정론 게더)·면 보정 플럭스 디바이스 계산, 경계셀
         grad [3*nbf] D2H (호스트가 gaussGrad 경계보정 + corr_b/limiter_b
         + bCorr=A_b·lim_b·corr_b 계산).
    Finish: bCorr=[nbf] 합류 → src=div(A·lim·corr)/V [nc] D2H,
         corrFOut=[nf] 면 보정 플럭스(플럭스 재구성 차감용) D2H.
    limitCoeff < 0 → corrected(무리미터).                              */
int rgpPEqnNOArm
(
    const double* dcsNO, double mdK,
    const double* C3, const double* CfB3
);

/*  은퇴한 디바이스 U-리미터의 dvec [3*nf] 해제 (호스트 가중치 상시
    전달 후 VRAM 회수). 해제 후 rgpUEqnSolve는 w 필수.               */
int rgpSTDvecRelease(void);
int rgpPEqnNOCorrPrep
(
    const double* pCell, const double* pBFace, const double* gMsf,
    const double* kf3,          /* [3*nf] 비직교 보정 벡터 (매 패스) */
    const double* Cf3,          /* [3*nf] 면 중심 (mdK>=0일 때 필수) */
    double limitCoeff, double* bGrad3Out
);
int rgpPEqnNOCorrFinish
(
    const double* bCorr, double* srcOut, double* corrFOut
);

void rgpPEqnFree(void);
const char* rgpPEqnLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
