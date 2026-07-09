/*---------------------------------------------------------------------------*\
  RGP-13 fgmGPU — host/device shared interface (pure C ABI)

  rgpKernelTypes.H와 같은 규칙: OpenFOAM/CUDA 헤더 include 금지,
  플레인 배열로만 통신. 구현은 gpu/rgpFgmKernels.cu.
\*---------------------------------------------------------------------------*/

#ifndef rgpFgmTypes_H
#define rgpFgmTypes_H

#ifdef __cplusplus
extern "C" {
#endif

/*  4-D FGM 테이블 업로드. tables는 [nFields * nZ*nG*nC*nK] 연접 버퍼
    (FGMTable flatIndex 순서 ((iZ*nG+iG)*nC+iC)*nK+iK), field 0 =
    sourcePV. 재호출 시 기존 테이블은 해제 후 교체.                   */
int rgpFgmUpload
(
    int nZ, int nG, int nC, int nK,
    const double* Zax, const double* Gax,
    const double* Cax, const double* Kax,
    int nFields,
    const double* tables
);

/*  셀별 gZ 대수 클로저 + chi_st(Pitsch-Steiner) + 16-코너 보간.
    mode4: 0=chi0 고정, 1=chi_st, 2=엔탈피 결손(h-hAd), 3=W 클램프
    hOrW: mode4 2/3에서 h 또는 W 필드 (그 외 nullptr 허용)
    fieldsOut: [nFields*nCells] SoA — field f의 셀 i는 [f*nCells+i];
    field 0(sourcePV)은 srcScale*rho가 곱해진 체적량.                 */
int rgpFgmEvaluate
(
    int nCells, int mode4,
    double Cv, double shapeZst, double chiMin, double chiMax,
    double srcScale, double chi0, double hOx, double hFuel,
    double Wlo, double Whi,
    const double* Z, const double* C, const double* rho,
    const double* Lsqr, const double* magSqrGradZ, const double* DeffZ,
    const double* hOrW,
    double* gZ, double* chiSt,
    double* fieldsOut
);

/*  디바이스 체인용: 마지막 rgpFgmEvaluate의 SoA 출력(디바이스 포인터)
    과 레이아웃. field 1 = T, field 2+k = k번째 테이블 종 Y.           */
const double* rgpFgmDevOutPtr(void);
int rgpFgmDevNFields(void);
int rgpFgmDevLastN(void);

void rgpFgmFree(void);
const char* rgpFgmLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
