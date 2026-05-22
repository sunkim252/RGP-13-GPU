# FGM (FPV + β-PDF) Table Generator

Cantera로 counterflow diffusion flamelet을 풀고 (Z̃, Z″², C̃) 3축으로 convolution하여 OpenFOAM dictionary 형식의 FGM 룩업 테이블을 생성한다.

## 타깃 case
- **Wang & Yang 2017** (J. Propulsion & Power), RD-0110-like biswirl injector
- P = 100 bar, T_LOX = 120 K, T_kerosene = 300 K, O/F = 2.31
- Surrogate: n-decane / n-propylbenzene / n-propylcyclohexane = 74 / 15 / 11 (vol)
- Mechanism: Wang et al. 2011 skeletal kerosene (108 species, 382 reactions)

## 실행 환경
- Cantera 3.1.0 + SRK 지원 (ct-env2 conda 환경, Windows)
- Activate: `conda activate ct-env2` (PowerShell)
- 스크립트는 `\\wsl.localhost\Ubuntu-22.04\home\sunkim\openfoam\RGP-13\RGP-13-realFluid\tools\fgm_table_gen\` 경로로 접근

## 파이프라인

| 단계 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 0 | `01_convert_chemkin.py` | `references/FGM/.../mmc{1,2,3}.txt` (CHEMKIN) | `data/wang2011_srk.yaml` (SRK Cantera YAML) |
| 1 | `03_flamelet_sweep_mpi.py` | `data/wang2011_srk.yaml` | `data/flamelets.h5` (Full-MPI: 각 rank가 자체 warmup + chunk sweep) |
| 2 | `04_build_fgm_table.py` *(미작성)* | `data/flamelets.h5` | `data/fgmProperties` (OpenFOAM dict) |

대안: `02_generate_flamelets.py`
- `--smoke` (default): 단일 flamelet, 동작 확인용
- `--sweep`: 순차 sweep (MPI 없을 때)
- `--warmup`: warmup baseline만 저장 (선택적 캐시)

## 진행상태
- [x] Step 0: CHEMKIN→SRK 변환 (Wang 2011: 106 sp, 382 rxn)
- [~] Step 1: flamelet 생성. **global 메커니즘(2/4-step)은 분수차수로 Cantera flame solver 비호환** → Wang 2011(elementary)만 사용. Wang production은 느림(overnight).
- [x] 합성 flamelet 생성기 (`make_synthetic_flamelets.py`) — Cantera 없이 다운스트림 검증용. 검증 완료.
- [x] Step 2: `04_build_fgm_table.py` — β-PDF convolution + OpenFOAM dict 출력. 합성으로 end-to-end 검증 완료.
- [x] C++ FGMTable 3D 확장 + FGM diffusion 로직 (컴파일 검증은 컨테이너에서 필요).
- [ ] 1D counterflow 검증 case
- [ ] Wang 2011 production 테이블

## 메커니즘 호환성 메모
- **global 2-step/4-step (`kero2s_srk.yaml`, `kero4s_srk.yaml`)**: 변환은 됐으나 sub-unity 반응차수(`KERO^0.55` 등)가 농도→0에서 Jacobian 발산 → Cantera 정상상태 flame solver와 비호환. flamelet 생성 불가. (P≈1.8 bar에서 막힘)
- **Wang 2011 (`wang2011_srk.yaml`)**: elementary 반응 → 작동. production용. solve당 6~50분.
- **합성 (`flamelets_synth.npz`)**: 다운스트림 코드 검증 전용.

## 사용법

### Step 0 (이미 통과)
```powershell
& "C:\Users\Sunchang Kim\miniconda3\envs\ct-env2\python.exe" 01_convert_chemkin.py
```

### Step 1 — 단일 flamelet smoke test
```powershell
& "C:\Users\Sunchang Kim\miniconda3\envs\ct-env2\python.exe" 02_generate_flamelets.py
```

### Step 1 — strain rate sweep

**옵션 A: Full-MPI (권장 — warmup도 병렬화)**

각 rank가 독립적으로 warmup + 자기 chunk sweep을 동시 실행. Wall time ≈ 단일 rank 워크플로우.

```powershell
mpiexec -n 8 "C:\Users\Sunchang Kim\miniconda3\envs\ct-env2\python.exe" `
    "\\wsl.localhost\Ubuntu-22.04\home\sunkim\openfoam\RGP-13\RGP-13-realFluid\tools\fgm_table_gen\03_flamelet_sweep_mpi.py"
```

**옵션 B: 순차 (single core, MPI 미설치 시)**
```powershell
& "C:\Users\Sunchang Kim\miniconda3\envs\ct-env2\python.exe" 02_generate_flamelets.py --sweep
```

**옵션 C: smoke test (단일 flamelet, 빠른 동작 확인)**
```powershell
& "C:\Users\Sunchang Kim\miniconda3\envs\ct-env2\python.exe" 02_generate_flamelets.py
```

**MPI 사전 작업 (mpi4py 미설치 시)**:
```powershell
# 1) MS-MPI 런타임 + SDK 설치 (Microsoft 공식): https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi
# 2) ct-env2에 mpi4py 설치
& "C:\Users\Sunchang Kim\miniconda3\envs\ct-env2\python.exe" -m pip install mpi4py
```

mpi4py 없을 시 03 스크립트는 자동 serial(size=1) fallback.

## 진행 노트
- C 정의: `C = Y_CO2 + Y_CO + Y_H2O + Y_H2` (Lacaze & Oefelein)
- Z 정의: Bilger 공식 (원소 mass fraction 기반)
- Z″² closure (LES): `C_v · Δ² · |∇Z̃|²`, `C_v ≈ 0.1`. OpenFOAM 측에서 계산하므로 테이블은 Z″²를 직접 입력으로 받음.
- β-PDF: Z̃, Z″² → P_β(Z; Z̃, Z″²). C는 δ-PDF (LES filter 내 분산 무시).
- T·ρ는 테이블에 저장하지 않음 (OpenFOAM 솔버의 SRK EOS가 runtime에 계산).
