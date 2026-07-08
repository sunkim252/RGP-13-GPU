/*---------------------------------------------------------------------------*\
  RGP-13 chemGPU — 상세화학 배치 적분 C ABI (호스트/디바이스 공유)

  thermoGPU와 같은 규칙: 이 헤더는 OpenFOAM/CUDA 헤더를 include하지 않는다.
  메커니즘은 고정 크기 POD로 정의되어 __constant__ 메모리에 통째로 올라간다.
\*---------------------------------------------------------------------------*/

#ifndef rgpChemTypes_H
#define rgpChemTypes_H

#define RGP_CHEM_MAX_SPECIES   16
#define RGP_CHEM_MAX_REACTIONS 32
#define RGP_CHEM_NEQ (RGP_CHEM_MAX_SPECIES + 1)   /* c[] + T */

/* 3체/폴오프 유형 */
#define RGP_TB_NONE   0
#define RGP_TB_M      1   /* +M 3체 */
#define RGP_TB_LIND   2   /* (+M) Lindemann */
#define RGP_TB_TROE   3   /* (+M) Troe */

#ifdef __cplusplus
extern "C" {
#endif

/*  단위: SI-kmol (OpenFOAM 관례) — A는 (m3/kmol)^(order-1)/s, Ta[K].
    janaf 계수는 무차원(NASA7, Cp/R 기준), Kc 산출에 그대로 사용.        */
typedef struct
{
    int    nSpecies;
    int    nReactions;
    double RR;        /* 몰 기체상수 [J/(kmol K)] */
    double pRef;      /* Kp→Kc 기준압 [Pa] (101325) */

    double W[RGP_CHEM_MAX_SPECIES];
    double Tcommon[RGP_CHEM_MAX_SPECIES];
    double janafHigh[RGP_CHEM_MAX_SPECIES][7];
    double janafLow[RGP_CHEM_MAX_SPECIES][7];

    double A[RGP_CHEM_MAX_REACTIONS];
    double beta[RGP_CHEM_MAX_REACTIONS];
    double Ta[RGP_CHEM_MAX_REACTIONS];
    int    reversible[RGP_CHEM_MAX_REACTIONS];

    /* 정수 양론계수 (희소성 무시, n이 작아 dense가 단순·빠름) */
    double nuL[RGP_CHEM_MAX_REACTIONS][RGP_CHEM_MAX_SPECIES];
    double nuR[RGP_CHEM_MAX_REACTIONS][RGP_CHEM_MAX_SPECIES];

    int    tbType[RGP_CHEM_MAX_REACTIONS];
    double eff[RGP_CHEM_MAX_REACTIONS][RGP_CHEM_MAX_SPECIES];
    double A0[RGP_CHEM_MAX_REACTIONS];      /* 저압 한계 (폴오프) */
    double beta0[RGP_CHEM_MAX_REACTIONS];
    double Ta0[RGP_CHEM_MAX_REACTIONS];
    double troe[RGP_CHEM_MAX_REACTIONS][4]; /* alpha, T3, T1, T2(0=3파라미터) */
} rgpChemMech;

int  rgpChemInit(int deviceId);
int  rgpChemUpload(const rgpChemMech* mech);

/*  등압 단열 반응기 배치: 각 셀 상태 (p, T, c[kmol/m3])를 자체 시간
    [0, dt]로 적분 (Rosenbrock34 적응 서브스텝). T, c는 in/out.
    반환 0=성공. stats에 총 서브스텝/최대 서브스텝 기록(NULL 허용).       */
int  rgpChemIntegrate
(
    int nCells,
    const double* dt,   /* [nCells] 셀별 적분 구간 (LTS 지원) */
    double relTol,
    double absTol,
    const double* p,
    double* T,
    double* c,          /* [nCells*nSpecies] 셀-우선 AoS */
    long long* stats    /* [2]: {총 스텝, 최대 스텝} — NULL 가능 */
);

void rgpChemFree(void);
const char* rgpChemLastError(void);

/*  디버그: 상태 y[n+1]에서 디바이스 쪽 RHS(dy0[n+1])와 FD Jacobian
    (J[(n+1)^2], 행우선)을 계산해 반환 — host 계산과 대조용.           */
int  rgpChemDebugJac
(
    double p,
    const double* y,
    double* dy0,
    double* J
);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
