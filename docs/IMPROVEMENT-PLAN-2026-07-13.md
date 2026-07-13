# RGP-13-GPU 개선 실행 계획서 (Opus 4.8 실행용)

> 대상 모델에게: 이 문서는 그대로 실행 가능한 작업 지시서다. 각 작업(W1~)은
> "파일:라인 → 현재 코드 → 변경안 → 빌드 → 게이트" 순서로 되어 있다.
> **철칙**: ①mode-0(copy) 경로의 산술·순서를 바꾸지 말 것(비트-정합 계약)
> ②모든 작업 후 반드시 게이트 배터리 통과 ③리얼가스 경로는 rd0110 생존
> 게이트가 유일한 커버리지(소형 게이트가 못 잡음 — RG-diet 사고 전례).

## 0. 현황 (2026-07-13 실측)

### 성능 좌표 (rd0110 2Mv2 = 2.4M셀 fgm 풀스택)
| 머신 | 구성 | s/step |
|---|---|---|
| GTX 1660S | 직렬 | 59.0 |
| RTX 4060 | 직렬 / 4랭크 | 31.9 / 22.7 |
| RTX PRO 6000 | 직렬 | 27.7 |
| RTX PRO 6000 | **8랭크 (성능 최대)** | **8.8** |
| RTX PRO 6000 | 16 / 24랭크 | 8.7 / 10.7 (정체→악화; 32는 MPI 슬롯 한계, 물리 24코어) |
| RTX PRO 6000 | 14.2M셀 8랭크 | 42.1 (준선형, 셀당 12% 개선) |

### 스텝 분해 (4060 직렬 31.9s — thermoTimings on)
- pressureCorrector 14.2s: pEqn.solve 5.6 / manifold 5.0 / thermo correct 1.9 / 잔여 1.7
- thermophysicalPredictor 11.5s: pre(가중치) 2.1×3 / hEqn+Dh 1.5×3 / manifold 2.8 / ZC 2.5 / refresh 1.0
- momentumPredictor 7.5s
- Blackwell nsys: GPU 커널 합 0.94s + 전송 1.9s → **호스트 코드가 ~25s 지배**

### 게이트 배터리 (모든 작업 후 필수)
```bash
S=<scratchpad>; RGPLIB=$S/rgplib
# 로컬 5게이트 (fgmMD/gmcMD T-bit, chemEDC 전필드, DLB 4랭크, rd0110 생존)
SCRATCH=$S RGPLIB=$S/rgplib RGPROOT=~/openfoam/RGP-13-GPU \
  ~/openfoam/RGP-13-GPU/RGP-13-realFluid/kisti-gh200/local-gates.sh
```
빌드(컨테이너):
```bash
apptainer exec --no-home --nv --bind /usr/lib/wsl \
  --bind ~/openfoam/RGP-13-GPU/RGP-13-realFluid:/opt/OpenFOAM/RGP-13-realFluid \
  --bind $S/rgplib:/rgpout ~/openfoam/RGP-13-GPU/openfoam13-rgp-gpu.sif bash -c '
  . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null
  export FOAM_USER_LIBBIN=/rgpout LIB_RGP13_SRC=/opt/OpenFOAM/RGP-13-realFluid/src
  export WM_NCOMPPROCS=16 CUDA_HOME=/usr/local/cuda
  cd /opt/OpenFOAM/RGP-13-realFluid/src/thermoGPU && touch gpu/*.cu && wmake libso
  cd ../../applications/modules/fgmFluid && wmake libso'
# ⚠️ wmake는 .cu의 헤더 의존을 추적 안 함 — 헤더 수정 시 반드시 touch gpu/*.cu
# ⚠️ 빌드 성공 판정은 log의 "error:" 카운트로 (tail 파이프가 exit code 가림)
```

## W-작업 목록 (분석 결과로 채움 — 아래에 이어짐)

---
## W0. 운용 확정사항 (코드 변경 없음 — 즉시 적용)

- **랭크 수**: RTX PRO 6000(물리 24코어)에서 **8랭크가 성능 최대** —
  4:10.8 / 8:8.8 / 16:8.7 / 24:10.7 s/step. 정체 원인 = GPU 1장
  직렬화 + 전송 바닥(~3s) + MPI 오버헤드. 8GB 4060은 4랭크(VRAM 한계).
- **AMR**: refineInterval ≥ 5 (1이면 매 스텝 topo-체크가 GPU 재아밍을
  유발해 42→171 s/step, 4×). maxCells는 소프트캡(선택 축소 없음).
  refine 케이스는 fvSolution에 pcorr/"pcorr.*" 솔버 항목 필수.
- **케이스 규약**: writeControl 주입 시 adjustTimeStep 여부 확인
  (timeStep 기반 write는 가변 dt에서 미기록 위험).

## W-공통 구현 규약 (모든 작업에 적용)

1. **비트-정합 게이트 원칙**: 결정론 필드(T 등)는 mode-0에서 비트-불변.
   U/phi는 atomics 노이즈 클래스(같은 빌드 재실행도 상이) — 판정은
   same-mode 재실행(runA/runB)로 노이즈 집합을 만들고
   "runA==runB && ≠ref"만 실변경으로 본다.
2. **결정론 리덕션**: 셀-면 게더는 stGradGather 패턴(셀→면 CSR,
   내부면 오름차순→경계면 오름차순 — rgpPEqnKernels.cu:3184-3225)만 사용.
   atomics 산란(stGradFace류) 신설 금지.
3. **스킴 1:1**: CPU가 케이스 dict의 스킴(cellMDLimited 등)을 쓰면 GPU도
   같은 스킴이어야 함 — LAD 사건(무제한 Gauss vs cellMDLimited) 전례.
   check 모드(gpuPEqnCheck 패턴)로 nDiff/maxRel 검사 코드를 함께 심을 것.
4. **ULP-limiter 계약**: multivariate/limited 스킴의 **가중치·limiter
   이산 결정은 CPU 계산 고정** (gcc FMA vs nvcc -fmad=false ULP 차이가
   limiter 분기를 뒤집음 — rgpSTWeightsSet 도입 배경). GPU로 옮기지 말 것.
5. **스테이징**: 신규 H2D/D2H는 반드시 rgpInPtr/rgpOutPtr/rgpOutFinish
   (rgpStage.H) 경유 — GH200 native에서 자동 zero-copy.
6. **VRAM**: 신규 상주 버퍼는 첫 사용 시점 lazy + cudaMemGetInfo 여유
   판단 + 실패 시 기존 경로 폴백 (NO-src 캐시 OOM 사고 전례:
   아밍 시점 판단은 후속 할당을 밀어냄).

---
# W-작업 목록 (우선순위·의존성 순)

## W1. pEqn CG: 수렴판정 K-배치 ★★★ (난이도 하 / 이득 확실)
**근거**: CG 반복당 블로킹 8B D2H ×3 중 res 리덕션(#3)은 **수렴판정 전용** —
alpha/beta에 안 들어감. K−1 반복을 blind로 돌려도 실행된 반복의 산술 불변
(p는 이터레이트-클래스 계약).
**파일**: src/thermoGPU/gpu/rgpPEqnKernels.cu:2601-2606 (rgpPEqnSolve 루프 꼬리)
**현재**:
```c
iter++;
reduceHost(nc, 1, gB.rA, ..., res);   // 매 반복 D2H #3
greduce(res);
res /= normFactor;
if (converged(res)) break;
```
**변경**: `const int K = 4;` (상수 시작 — 추후 dict화 가능)
```c
iter++;
if (iter % K == 0 || iter >= maxIter)
{
    reduceHost(nc, 1, gB.rA, gB.red, gB.wA /*scratch*/, res);
    greduce(res);
    res /= normFactor;
    if (converged(res)) break;
}
```
(reduceHost 시그니처는 현재 코드 그대로 유지 — 위는 의사 표기. 병렬
greduce도 같은 가드 안으로.) **보존 필수**: 루프 앞 초기 converged 가드
(2565), pAwA==0 break(2595), maxIter 종료.
**게이트**: 5게이트 배터리 (T-bit 불변이어야 함 — p 반복수는 달라져도
됨). rd0110 로컬 성능 재실측 (기대: pEqn.solve 5.6s → ~4.5s).

## W2. pEqn CG: CSR 경로 dot-융합 + 디바이스 스칼라 ★★ (난이도 중)
**전제**: W1 후. CSR 경로(gM.dVals && gM.dRowPtr 시 pSpmv→csrSpmv:1922)
한정 — csrSpmv(:665)는 atomics 없는 row-per-thread라 융합 최적.
**내용**:
1. csrSpmv에 blockReduce+atomicAdd(gB.red+1)로 `pAwA=Σ pA·(A·pA)` 융합
   (faceSpmv LDU 경로는 손대지 말 것 — atomics로 wA 미완성 상태).
2. precondDir(:415)에 rArD 융합(gB.red+0), cgUpdate(:429)에 res 융합(gB.red+2).
3. alpha/beta를 디바이스 상주로: dirUpdate(:510)/cgUpdate가 host-value
   대신 device-pointer(gB.red 슬롯)를 받는 변형 커널 추가 → 반복당
   D2H 0 (W1의 주기적 res 체크만 남음).
**주의**: gB.red는 4 double 할당돼 있음(:2384) — 신규 할당 불필요.
반복 산술의 합산 순서가 기존 reduceK와 달라지면 res/알파값의 라운딩이
달라짐 → **p 이터레이트-클래스로 허용되지만**, LDU(비-CSR) 폴백 경로는
기존 그대로 남겨 mode-무관 안전판 유지. gpuPEqnCheck로 초기잔차
CPU-공식 대조(기존 체크 인프라) 통과 확인.
**게이트**: 배터리 + rd0110 pcg 반복수·잔차 로그 비교(수렴 열화 없어야).

## W3. pEqn 초기추정 시간외삽 ★★ (난이도 최하 / host-only)
**근거**: x0 = 현재 p (pressureCorrector.C:1955, 솔버는 :2508에서 1회만
읽음). `2·p^n − p^(n−1)` 주입은 반복수만 줄임 (실행 산술 불변).
**파일**: applications/modules/fgmFluid/pressureCorrector.C:1949-1963
**변경**: 호출 직전
```cpp
// 시간외삽 초기추정 (iterate-class: 반복수만 감소, 산술 불변)
scalarField pXtr(p.primitiveField());
pXtr += (p.primitiveField() - p.oldTime().primitiveField());
```
p0 인자(:1955)를 `pXtr.begin()`으로. **주의**: p.oldTime()(:1954, ddt용)은
절대 건드리지 말 것. 첫 스텝(oldTime==p)엔 자동으로 무해.
LTS/adjustTimeStep 케이스는 dt 비율 가중 없이 단순 외삽 유지(안전).
**계측**: 로그의 pcg iters (rgpBiCGStab/솔브 반복 출력) 전후 비교 —
개선 없으면 되돌림 (한 줄 스위치).
**게이트**: 배터리 (T-bit).

## W4. Deff 공통부 호이스트 ★★★ (난이도 하 / bit-safe 최대 host win)
**근거**: Deff(var) = mu/Le(var) + rho·nut/Sct (fgmFluid.C:1229-1246).
`turb=rho·nut/Sct`와 `thermo.mu()`가 outer당 **최대 5회 재계산**
(updateManifold의 Deff("Z"):741 + predictor의 DZ:706/DC:758/Dh:855
[/DW:943]) — 전부 동일 피연산자라 비트-동일.
**파일**: applications/modules/fgmFluid/fgmFluid.{H,C},
thermophysicalPredictor.C
**변경**:
1. fgmFluid.H에 mutable 캐시: `mutable tmp<volScalarField> DeffTurbCache_;
   mutable tmp<volScalarField> DeffMuCache_; mutable label DeffCacheStamp_ = -1;`
2. Deff() 진입부: `if (DeffCacheStamp_ != mesh.time().timeIndex()*100 +
   outerIndex)` 식의 스탬프 대신 — **가장 단순·안전**:
   thermophysicalPredictor() 시작에서 캐시 채우고 끝에서 clear();
   updateManifold이 predictor 안에서 불리므로 같은 캐시 사용.
   pressureCorrector 경로의 updateManifold(:526)은 **hostScatter=false라
   Deff 호출 없음** — 확인 후, 있으면 그 경로는 비캐시 폴백.
3. Deff 본문: 캐시 valid()면 `return DeffMuCache_()/Le + DeffTurbCache_()`,
   아니면 기존 계산.
**비트 주의**: `mu/Le + turb`의 연산 순서가 기존 `mu/Le + turb`와 동일해야
함 — 기존 식 그대로 유지(:1241/:1245 형태 보존), 캐시는 피연산자만 재사용.
**게이트**: 배터리 — **T-bit가 그대로여야 함** (다르면 순서가 바뀐 것).
**기대**: hEqn+Dh 1.5s 중 상당분 + ZC 블록 일부, outer당 ~1s급.

## W5. multivariate 스킴 객체 캐시 ★ (난이도 하 / minor)
**근거**: 스킴 객체 생성(ITstream 파싱+필드 할당,
thermophysicalPredictor.C:146-156)이 outer마다 반복 — 가중치 계산(4 grads,
ULP-계약 잠금)은 불변이지만 생성 오버헤드는 순수 낭비.
**변경**: 파싱 결과(스킴 이름/계수)만 멤버로 1회 캐시. 가중치 계산 자체는
매 outer 그대로. **multivariateScheme 생성자가 가중치 계산을 겸하므로**
객체 재사용은 불가 — 이 작업은 ITstream 파싱 캐시에 한정 (이득 소).
스킵 가능; W4 이후 계측으로 판단.

## W-금지 목록 (하지 말 것)
- multivariate limiter/가중치 GPU화 (ULP 계약 — thermophysicalPredictor.C:133-142 주석)
- LAD grad GPU화 (케이스 스킴 cellMDLimited 불일치 + CPU 0.2s라 이득 0 — #50 전례)
- RG_* 슬라이스 D2H 스킵 (리얼가스 산발 소비 — #47 sigFpe 전례)
- 아밍 시점 VRAM 잔량 판단 (첫 Prep lazy로 — #51 OOM 전례)

## W6. devChain을 비직교 케이스에서 활성화 ★★★★ (최대 단일 발견)
**근거**: pressureCorrector.C:585-586
```cpp
bool devChain = gpuUEqn_ && gpuPEqn_ && !gpuDevChainOff_ && !gpuNonOrtho_;
```
디바이스 prep 체인(pcPrepFace:1333/pcPsis:1381/pcPhiRecon:1393/pcUUpdate:1406,
rgpPEqnKernels.cu)이 **비직교 보정 armed 시 통째로 꺼짐** — rd0110(비직교)의
호스트 fvc 프렙 체인이 그래서 도는 것:
- :598-602 interpolate(rho), :726-732 interpolate(rho*rAU),
  :779-786 harmonic rAUf, :813-833 psis,
  :852-869 **flux(HbyA)+ddtCorr(최대 단일 항목)**,
  :2005-2112 질량플럭스 재구성 — 전부 기존 커널이 흡수 가능.
**작업**: devChain 조건에서 `!gpuNonOrtho_` 제거가 목표. 선행 확인:
1. nonOrthoSrc 람다(:1310-1649)가 호스트-상주로 가정하는 값들
   (harmonic rAUf, phiHbyAv 등)을 디바이스 상주본에서 소비하도록 연결
   — rgpPEqnNOCorrPrep의 gMsf 인자(:1370대)가 A(=rAUf) 호스트 배열을
   받는 부분을 devChain의 디바이스 rAUf로 대체(포인터 전달 or 접근자).
2. devChain의 경계 재구성(:1148-1261, :2076-2099)은 호스트 유지.
3. 단계적: 먼저 `gpuDevChainNO_` 실험 스위치(dict, 기본 off)로 새 경로를
   병행 구축 → rd0110에서 성능·생존 확인 → gpuPEqnCheck(비-devChain
   전제 :1652 — check 모드에선 자동 비활성 유지) → 기본 on 전환.
**게이트**: 배터리 + rd0110 (T-bit는 fgmMD, 리얼가스는 rd0110 생존).
**기대**: pCorr 잔여 1.7s의 대부분 + flux 재구성 — 스텝 -1.5s급.

## W7. SoA gather/scatter 융합 ★★ (제로 리스크 메모리 win)
**근거**: momentumPredictor.C:261-346(U/Uold/UB3 게더 4×3nc),
:758-772(U3out 산포), pressureCorrector.C:2269-2279(U 산포) — AoS↔SoA
호스트 루프들. 순수 메모리 작업.
**작업**: 기존 업로드 경로에 스트라이드 지원을 넣거나(rgpInPtr 3-성분
변형), 게더 루프를 OpenMP 4스레드로(호스트 유지, 단순) — **후자 권장**
(비트 영향 0, 30분 작업): `#pragma omp parallel for` — OF 필드 접근이
스레드-세이프한 순수 인덱싱 루프에 한정. 빌드에 -fopenmp 필요 여부 확인.
**게이트**: 배터리.

## W8. dev2 경계 플럭스·bCoeffs 디바이스화 ★ (난이도 상 / 보류 권장)
momentumPredictor.C:437-702 — fvPatchField 가상 API가 본질적으로 호스트.
nbf 크기라 이득 제한적. **W6·W4 효과 실측 후 재평가.**

## W9. div(phi,U) 가중치 — 금지 목록 재확인
uLimVWeights 커널이 존재하지만 은퇴됨(rgpSTDvecRelease) — ULP-limiter
계약. **재활성 금지** (호스트 가중치 업로드 규약 유지).

---
# 실행 순서 요약 (Opus 4.8용)

| 순서 | 작업 | 예상 이득(4060 직렬 기준) | 리스크 |
|---|---|---|---|
| 1 | W3 초기추정 외삽 | pEqn 반복수 −10~30% | 최하 (한 줄 되돌림) |
| 2 | W1 수렴판정 K-배치 | 반복당 sync 3→2 | 하 |
| 3 | W4 Deff 호이스트 | ~1s/스텝 | 하 (T-bit로 즉시 검증) |
| 4 | W6 devChain 비직교 활성화 | ~1.5s/스텝 | 중 (실험 스위치로 단계) |
| 5 | W7 SoA OpenMP | ~0.3-0.5s | 최하 |
| 6 | W2 CG dot-융합 | 반복당 sync →0 | 중 |

각 작업 = 독립 커밋 + 배터리 + rd0110 실측 한 줄을 커밋 메시지에.
목표: 4060 직렬 31.9 → ~27s, PRO6000 8랭크 8.8 → ~7s대.

# 참고: 최신 실측 좌표 (계획 수립 시점)
- PRO6000 14.2M: 8랭크 42.1s / **16랭크 32.0s** — 대격자에선 16랭크 유효
  (GPU 비중↑ → 랭크 한계 이동). 운용: 2.4M→8랭크, 10M+→16랭크.
