# KISTI 슈퍼컴퓨터 6호기 (GH200) 브링업 킷

KISTI 6호기 베타 계정 확보 시 **첫 할당 세션에서 그대로 실행**할 수 있도록 준비한
빌드·검증 절차. 목표: RGP-13-GPU 스택(x86 RTX에서 검증 완료)을 GH200
(Grace-Hopper, aarch64 + NVLink-C2C 코히런트 메모리)에서 기동하고,
**CPU 계산과의 비트-정합**과 **coherent-native 스테이징**을 확인한다.

코히런트 HW가 감지되지 않으면 코드는 자동으로 기존 discrete-GPU copy 경로로
동작한다(단일 GPU fallback — 수정 불필요, `RGP_GPU_UNIFIED` env로도 강제 가능).

## 0. 사전 확인 (로그인 노드)

```bash
uname -m                      # aarch64 여야 함
apptainer --version || singularity --version
nvidia-smi                    # GH200: "GH200 480GB" 류; 드라이버 ≥ 545
srun --partition=<beta-partition> --gres=gpu:1 --pty bash   # 정책 확인
```

- KISTI는 통상 Singularity/Apptainer 제공(뉴론 기준). 6호기 베타의 컨테이너
  정책·파티션 이름은 베타 안내 공지로 대체할 것.
- 컴퓨트 노드에서 외부 네트워크(도커 허브 pull)가 막혀 있으면:
  로그인 노드에서 `apptainer pull docker://nvidia/cuda:12.4.1-devel-ubuntu22.04`
  후 `Bootstrap: localimage`로 def 첫 줄만 바꿔 빌드.

## 1. 소스 반입

```bash
git clone https://github.com/sunkim252/RGP-13-GPU.git
# 또는 (외부망 차단 시) 로컬에서: tar 없이 git bundle 권장
#   git bundle create rgp13-gpu.bundle main && scp → git clone rgp13-gpu.bundle
```

⚠️ AMR core-mod가 필요한 케이스를 돌릴 거면 OpenFOAM-13은 upstream 클론이
아니라 수정 트리(x86 개발기의 `RGP-13-GPU/OpenFOAM-13`)를 rsync로 가져와
def의 클론 단계를 대체한다(소스는 아치 무관).

## 2. 컨테이너 빌드 (GH200 노드에서 — 2h± 예상)

```bash
cd RGP-13-GPU
apptainer build openfoam13-rgp-gpu-arm64.sif RGP-13-realFluid/../openfoam13-rgp-gpu-arm64.def
# def는 저장소 루트 기준: openfoam13-rgp-gpu-arm64.def
```

빌드가 검증하는 것: OF-13 aarch64 네이티브(WM_ARCH=linuxArm64) →
CUDA 룰(sm_90+PTX, -fmad=false) → RGP 스택 5개 libso → AmgX
(**cmake 로그 "This is a MPI build:TRUE" 필수** — 실패 시 즉시 중단됨).

- fakeroot 불가 시: `apptainer build --fakeroot` 대신 KISTI 제공 빌드 노드
  또는 `--sandbox` + `--fix-perms`.

## 3. 스모크 #1 — 기동 + coherent 감지 (GPU 1장, 5분)

```bash
sbatch slurm/smoke1-serial.sh     # 내부: testCases/h2CounterFlow_gpuChem 축소판
```

**확인 항목** (`log.smoke1`) — x86 4060에서 실측한 기대 배너(카피 모드):
```
gpuMulticomponentFluid: GPU transport mesh armed -- N cells; staging copy (discrete) [coherent HW: no]
gpuChemistryModel: ARMED — 9 species / 22 reactions exported from the case mechanism ...
```
1. GH200에서는 첫 줄이 **`staging unified-native (coherent, zero-copy)
   [coherent HW: yes]`** 로 나와야 함 — rgpGpuInit이
   `cudaDevAttrPageableMemoryAccessUsesHostPageTables`로 자동 감지.
   `no`면 즉시 이슈(드라이버/ATS 설정 확인; 코드는 copy로 fallback해
   결과는 정상이어야 함 — 그 상태로 스모크 계속 진행 가능).
2. `gpuChemistryModel: ARMED` 배너 + 크래시 없이 `End` 도달.

## 4. 스모크 #2 — native vs copy 비트-정합 (GPU 1장, 10분)

```bash
sbatch slurm/smoke2-parity.sh
```

같은 케이스를 ① 자동(native) ② `RGP_GPU_UNIFIED=0`(강제 copy)로 돌려
최종 시간 디렉터리 전 필드 `cmp` — **비트-동일이 통과 기준**.
(x86에서 mode-1 mapped 프록시로 사전 검증된 항목의 실기 확정.
 유일한 미검증 요소였던 'CUDA 런타임의 코히런스 시맨틱'이 여기서 닫힌다.)

## 5. 스모크 #3 — 4랭크 병렬 + DLB + AmgX (GPU 1~4장, 15분)

```bash
sbatch slurm/smoke3-parallel.sh
```

**확인 항목**:
1. 4랭크 pcg vs 직렬 pcg: T 비트-일치(p는 이터레이트 클래스 허용)
2. `gpuChemistry DLB:` 텔레메트리 출력 + on/off 최종장 비트-동일
3. (GPU 여러 장이면) 랭크→GPU 매핑 배너 확인: `rank r -> GPU (r % nDev)`
4. AmgX 분산: 케이스 dict의 `gpuPEqnSolver amgx;`(fgmProperties/모듈 dict)로
   재실행, 수렴 정상 여부 (성능은 pcg 우세가 기본값 — amgx는 동작 확인만)

## 6. 성능 기준점 (여유 시)

rd0110 2Mv2 (2.6M 셀) — x86 실측: 1660S 60.8 s/step, 4060 33.8 s/step.
GH200 (H100급 SM + HBM3 + C2C)에서 **< 15 s/step** 기대. native 모드에서
스테이징 H2D 소거 효과는 프로파일 라인(`pCorr`, `rgpChem`)으로 비교.

## 문제 발생 시 fallback 스위치

| 증상 | 스위치 |
|---|---|
| native에서 결과 이상 | `RGP_GPU_UNIFIED=0` (강제 copy — 기존 검증 경로) |
| mapped 실험 | `RGP_GPU_UNIFIED=1` |
| AmgX 이상 | pcg가 기본값 — dict `gpuPEqnSolver` 제거로 복귀 |
| AmgX 설정 실험 | `RGP_AMGX_JSON=<파일>` |
| 화학 DLB 의심 | `chemistryProperties.gpuCoeffs { dlb off; }` |
| GPU 화학 자체 의심 | `chemistryProperties`의 solver를 CPU ode로 |

모든 스위치는 **코드 수정 없이** 기존 x86 검증 경로로 되돌린다.
