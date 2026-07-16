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

/*  D2H 다이어트: 다음 rgpFgmEvaluate에서 fieldsOut의 [first, first+n)
    필드 슬라이스는 호스트로 내리지 않는다(copy 모드 전용 — zero-copy
    모드에선 무의미). 디바이스 상주 SoA(체인 소비)는 영향 없음.
    호스트 배열의 해당 슬라이스는 미갱신으로 남으므로, 호출측은 그
    필드들의 호스트 소비자가 없음을 보장할 것. (0,0)이 해제. */
void rgpFgmHostCopySkip(int first, int n);

/*  디바이스 체인용: 마지막 rgpFgmEvaluate의 SoA 출력(디바이스 포인터)
    과 레이아웃. field 1 = T, field 2+k = k번째 테이블 종 Y.           */
const double* rgpFgmDevOutPtr(void);
int rgpFgmDevNFields(void);
int rgpFgmDevLastN(void);

/*  Y-다이어트 부분 게더: 마지막 evaluate SoA에서 부분집합 셀의 필드 행
    [firstField, firstField+nF)를 압축 게더해 호스트로 내린다(hostOut
    레이아웃 [nF][nSub]). cells는 SetRowGather로 1회 업로드(재호출 시
    교체). FetchFields는 필드 슬라이스 전체 연속 D2H(폴백 복원용).
    HostCopyActive: copy 스테이징 모드(D2H 존재) 여부 — zero-copy
    (native) 모드면 0 (다이어트 무의미·비활성).                          */
int rgpFgmSetRowGather(const int* cells, int n);
int rgpFgmGatherRows(int firstField, int nF, double* hostOut);
int rgpFgmFetchFields(int firstField, int nF, double* hostOut);
int rgpFgmHostCopyActive(void);

/*  경계면 RG 계수 채움(Opt-2)의 GPU 배치: 경계 좌표(Z,gZ,C,c4src)에서
    테이블 필드 [firstField, firstField+nOut)를 16-코너 보간해 내린다
    (hostOut [nOut][nFaces]). 클램프·coord4 변환은 CPU fillRG 루프와
    1:1 (bmode 0=c4 직접(chi), 2=h→dh, 3=W 클램프). 산술은 fgmKernel과
    동일 디바이스 함수 — 호출측이 첫 호출에서 CPU 경로와 비트 비교로
    자가검증할 것.                                                     */
int rgpFgmBndCoeffs
(
    int nFaces, int bmode,
    double hOx, double hFuel, double Wlo, double Whi,
    const double* Zb, const double* gZb, const double* Cb,
    const double* c4b,
    int firstField, int nOut, double* hostOut
);

/*  M4: Lsqr(혼합길이²) 스탬프 — varModel les(순수 기하 V^(2/3))·laminar
    (0 상수)이고 정적 메시면 런-상수이므로, 같은 스탬프의 재업로드를
    생략하고 디바이스 잔존본을 재사용한다. -1=항상 업로드(RAS/동적
    메시). 버퍼 재할당 시 자동 무효화.                                 */
void rgpFgmLsqrStamp(long stamp);

/*  랭크 간 매니폴드 공유(CUDA IPC). 핸들 블롭은 5개 할당
    (Zax,Gax,Cax,Kax,tables) × 64B = 320B 고정.
    Supported: cudaIpcGetMemHandle 프로브(캐시) — 단 WSL은 핸들 생성은
    되지만 타 프로세스 open이 거부되므로, 최종 판정은 임포트 실패
    폴백이 담당한다.
    Export: 업로드 완료 상태에서만. Import: 기존 테이블 해제 후 리더의
    할당을 열어 gT에 매핑(dims는 리더의 rgpFgmUpload와 동일해야 함).
    임포트로 매핑된 포인터는 rgpFgmFree에서 cudaIpcCloseMemHandle로
    닫는다(cudaFree 아님).                                             */
#define RGP_FGM_IPC_BLOB 320
int rgpFgmIpcSupported(void);
int rgpFgmIpcExport(unsigned char* handles320);
int rgpFgmIpcImport
(
    int nZ, int nG, int nC, int nK, int nFields,
    const unsigned char* handles320
);

/*  현재 디바이스의 PCI busId 문자열 (랭크→GPU 그룹핑 키용) */
int rgpFgmDevBusId(char* out, int cap);

void rgpFgmFree(void);
const char* rgpFgmLastError(void);

#ifdef __cplusplus
}
#endif

#endif

/* ************************************************************************* */
