/*---------------------------------------------------------------------------*\
  RGP-13 thermoGPU — host/device shared interface (pure C ABI)

  이 헤더는 nvcc(.cu)와 g++(Foam 브리지/유틸) 양쪽에서 include된다.
  절대 OpenFOAM 헤더나 CUDA 헤더를 include하지 말 것 — OF Scalar.H의
  Bessel 선언(j0/j1/y0/y1)이 CUDA 내장과 충돌하므로 디바이스 계층은
  플레인 배열 + POD 구조체로만 통신한다.
\*---------------------------------------------------------------------------*/

#ifndef rgpKernelTypes_H
#define rgpKernelTypes_H

#include <stddef.h>

/* 스레드-로컬 스택 배열 상한 (FPV 케이스: 4-7종, 여유 포함) */
#define RGP_GPU_MAX_SPECIES 40

#ifdef __cplusplus
extern "C" {
#endif

/*  종-상수 테이블 (호스트 포인터; rgpGpuUpload가 디바이스로 복사)
    쌍 행렬은 [n*n] row-major. SRKchungTakaMixture 생성자가 만든
    행렬을 그대로 사용한다 — k_ij는 COEF1..3에 이미 반영돼 있음.       */
typedef struct
{
    int    nSpecies;
    int    stableRoot;   /* SRK 근 선택: 0 = 최대근(legacy), 1 = 최소 fugacity */
    double RR;           /* 몰 기체상수 [J/(kmol K)] — Foam 상수를 그대로 전달 */

    /* 믹스처 JANAF 온도 범위 (OF 관례: 종-0의 범위가 믹스처에 승계됨) */
    double TlowJ, ThighJ, Tcommon;

    /* [n] */
    const double* W;         /* 몰질량 [kg/kmol]                 */
    const double* BM;        /* SRK co-volume b_i                */
    const double* CM;        /* Peneloux 상수 시프트 c_i          */
    const double* janafHigh; /* [7n] 질량기준 high-T Cp 계수      */
    const double* janafLow;  /* [7n] 질량기준 low-T Cp 계수       */

    /* [n*n] 쌍 혼합 행렬 (SRK aAlpha 3항 + Chung 조합칙) */
    const double* COEF1;
    const double* COEF2;
    const double* COEF3;
    const double* SIGMA3M;
    const double* EPSILONKM0;
    const double* OMEGAM0;
    const double* MM0;
    const double* MIUIM0;
    const double* KAPPAIM;
} rgpGpuTables;

/* 모든 함수는 0=성공, 비0=cudaError_t 코드를 반환 */
int  rgpGpuInit(int deviceId);              /* 음수면 device 0 */
int  rgpGpuDeviceCount(void);               /* 실패 시 0 */
int  rgpGpuUpload(const rgpGpuTables* tables);

/*  p[nCells], T[nCells], Y[nCells*nSpecies] (셀-우선 AoS: Y[i*n+s])
    → rho/mu/kappa/Cp/Cv/psi[nCells] — RhoFluidThermo::calculate()가
    셀마다 채우는 물성 세트 전체. 내부 디바이스 버퍼는 grow-only 재사용. */
int  rgpGpuEvaluate
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
);

/*  ha [J/kg] 일괄 평가: janafThermo::ha(질량 블렌드) + SRK 믹스처
    departure — updateManifold의 he 재시드 thermo_.he(p,T) 대체.
    rgpGpuUpload로 올린 동일 테이블 사용.                              */
int  rgpGpuEvaluateHa
(
    int nCells,
    const double* p,
    const double* T,
    const double* Y,
    double* ha
);

/*  fgm 디바이스 체인 변형: (T, 테이블 종 Y)를 마지막 rgpFgmEvaluate의
    디바이스 SoA 출력에서 직접 취해 호스트 왕복을 제거한다. p만 업로드.
    yMap[s] >= 0: fgm 필드 인덱스(2+k), < 0: yHost SoA의 (-1-yMap[s])행
    (비테이블 종). 셀 수는 마지막 fgm evaluate와 같아야 한다.          */
int  rgpGpuEvaluateFromFgm
(
    int nCells,
    const double* p,
    const int* yMap,
    int nYHost,
    const double* yHost,
    double* rho,
    double* mu,
    double* kappa,
    double* Cp,
    double* Cv,
    double* psi
);

int  rgpGpuEvaluateHaFromFgm
(
    int nCells,
    const double* p,
    const int* yMap,
    int nYHost,
    const double* yHost,
    double* ha
);

/*  호스트 버퍼 page-lock 등록/해제 (pageable 전송 병목 완화 +
    unified mapped 모드의 디바이스 별칭).                              */
int rgpPinHost(void* p, size_t bytes);
int rgpUnpinHost(void* p);

/*  unified 메모리 모드: 0=copy(디스크리트 기본), 1=mapped(등록 버퍼
    zero-copy — 검증용), 2=native(GH200류 일관 메모리, 자동 감지).
    env RGP_GPU_UNIFIED=0/1/2 로 override.                             */
int rgpGpuUnifiedMode(void);

void rgpGpuFree(void);
const char* rgpGpuLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
