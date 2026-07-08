/*---------------------------------------------------------------------------*\
  RGP-13 thermoGPU — host/device shared interface (pure C ABI)

  이 헤더는 nvcc(.cu)와 g++(Foam 브리지/유틸) 양쪽에서 include된다.
  절대 OpenFOAM 헤더나 CUDA 헤더를 include하지 말 것 — OF Scalar.H의
  Bessel 선언(j0/j1/y0/y1)이 CUDA 내장과 충돌하므로 디바이스 계층은
  플레인 배열 + POD 구조체로만 통신한다.
\*---------------------------------------------------------------------------*/

#ifndef rgpKernelTypes_H
#define rgpKernelTypes_H

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
    → rho/mu/kappa[nCells]. 내부 디바이스 버퍼는 grow-only 재사용.    */
int  rgpGpuEvaluate
(
    int nCells,
    const double* p,
    const double* T,
    const double* Y,
    double* rho,
    double* mu,
    double* kappa
);

void rgpGpuFree(void);
const char* rgpGpuLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
