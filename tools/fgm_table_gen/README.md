# FPV (β-PDF, normalized-c) Table Generator

Cantera로 counterflow diffusion flamelet 군을 풀고 **(Z̃, gZ, c̃) 3축 steady FPV
테이블**(OpenFOAM dictionary)을 생성한다. burnt 끝은 **단열 화학평형/manifold
envelope로 닫혀** ω̇_c(c=1)=0 — 솔버에서 수송되는 c̃가 [0,1]에 갇힌다.
(4축 (Z̃,gZ,c̃,χ_st) 레이아웃은 `--chi-axis`로 보존 — **unsteady FPV** 확장용.
steady flamelet 군에서는 c와 χ_st가 같은 1-파라미터 군의 중복 지표라 χ축이
정보를 더하지 않는다.)

## 타깃 case
- **Ahn et al. 2012** (Combust. Sci. Technol. 184:323, KARI 19-element
  bi-swirl): **P_c = 52.5 bar** 설계점 — production 테이블
- **Wang, Huo & Yang 2015** (CST 187:60) 검증: 1–200 atm 압력 sweep,
  **양쪽 인렛 800 K** (무거운 n-alkane 표준, Fig 18), T_max/δ/q̇ 스케일링 비교
- Surrogate: n-decane / n-propylbenzene / n-propylcyclohexane = 74/15/11
- Mechanism: Wang et al. 2011 skeletal kerosene (106 sp, 382 rxn)
- 진행변수: C = Y_CO2 + Y_CO + Y_H2O + Y_H2 (Ihme & Pitsch 2008과 동일)
- Z_st (Bilger, mass) = **0.2255** (구 0.0625는 3.6배 오류였음)

## 실행 환경
- **Cantera 3.2 + Python 3.12** — `conda activate ct-env3` (Linux, miniforge3)
- Cantera 3.2의 `Redlich-Kwong` thermo + species `acentric-factor` = SRK 동등
- transport: 구조는 `unity-Lewis-number`(솔버의 unity-Schmidt와 일관),
  물성 사후평가는 `high-pressure-Chung`(dense-gas, OpenFOAM Chung-Takahashi 대응)

## 파이프라인 (dual-gas)

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 0 | `01_convert_chemkin.py` | CHEMKIN → SRK Cantera YAML (`data/wang2011_srk_v32.yaml`) |
| 1 | `05_flamelet_sweep_dualgas.py` | **dual-gas flamelet 생성**: 구조는 ideal-gas+unity-Le로 풀고(robust), 저장 시 각 격자점에서 **SRK+high-pressure-Chung으로 ρ/μ/κ/c_p/ω̇_C/χ_st 사후평가** (Zips 2018과 같은 fidelity 분리). `--pressure-atm P`(압력별 디렉토리), `--warm-from ckpt`(고압 continuation), `--mdot-factor`(strain 스텝) |
| 2 | `04_build_fgm_table_4d.py` | flamelet 군 → **3D steady FPV 테이블** (기본). Zq별 HP-평형으로 C_eq/T_eq/Y_eq 계산 → **envelope 정규화** c=C/C_norm(Z), C_norm=max(C_eq, 군 envelope) → c=1열(ω̇=0)·c=0열(frozen) 경계 삽입 → β-PDF(Z) convolution. `--species`(LES용 축소), `--chi-axis`(UFPV 4D), `--yaml` |
| 3 | `yaml_to_openfoam_thermo.py` | Cantera YAML → OpenFOAM 106종 thermo (janaf+Sutherland+rfProperties; Vc는 Zc 상관식, sigmvi는 Fuller). **Thigh를 5000 K로 바닥** — OF mixture가 전 종 min(Thigh)로 T를 클램프하는 함정 회피 |
| – | `run_wang2015_sweep.sh` / `run_150_200.sh` | 압력 sweep 드라이버 (150/200 atm은 100 atm ckpt에서 `--warm-from` 연쇄) |
| – | `apriori_check_4d.py` | 테이블 자기일관/flamelet 재구성 a-priori 검증 |

## 수렴 레시피 (필수 노하우)
- **cold start는 mixture-averaged로** 수렴 후 unity-Le로 전환 (unity-Le cold
  start는 auto solve 실패)
- **모든 settle은 다회 라운드**: Cantera가 settle된 transient를 steady로
  선언 안 하는 일이 흔함 → "declared 또는 Tmax 정체(<2 K)"까지 반복,
  burning(Tmax>1500 K)이면 수용. 단일-라운드 accept는 고압에서 평형 천장을
  넘는 비물리 상태를 남긴다 (152 bar에서 +200 K 시프트 vs 114 K 여유)
- **transport 전환 후 재-refine 금지** (격자 과축소 + 미수렴 → Tmax 과열)
- **strain 스텝**: 기본 ×1.25, **저압(≤10 atm)은 ×1.10** — 거친 스텝은 가짜
  extinction(transient 붕괴)을 만든다 (10 atm: ×1.25→7개 vs ×1.10→64개)
- **SRK counterflow를 직접 풀지 말 것**: 케로신 임계점(n-decane Tc=618 K,
  Pc=21 bar) 근처에서 P/T가 움직이면 Newton 붕괴; 해상 격자(≥150점)에선
  고정점도 stall. → dual-gas가 해법

## 품질 기준 (flamelet 군 검수)
- Tmax ≤ 평형 천장 T_eq(P) ≈ 3791 K × (P/50.7 bar)^0.0474
- Z@Tmax ∈ [0.15, 0.35] (Z_st 부근), T 프로파일 2차차분 스파이크 0
- 조성 위생: |ΣY−1| > 5% 점은 빌더가 마스킹 (accepted transient의 고립 스파이크)

## 진행상태
- [x] dual-gas 파이프라인 + 평형/envelope closure + c 정규화
- [x] **OpenFOAM end-to-end 검증 PASS** (50 atm, `testCases/wang2015_P50atm/`):
      T_max=3636 K burning + c_max=1.0000 bounded
- [x] unity-Le flamelet 군: 1/25/50/100 atm + 52.5 bar 각 28개, 10 atm 64개
- [~] 150/200 atm: `--warm-from` 압력 continuation 실행 중
- [ ] OpenFOAM strain sweep → T_max/δ/q̇ 추출 → Wang 2015 스케일링 비교
      (δ~1/√(pa), q̇~p^0.53√a, T~p^0.047); q̇용 hrr 필드 추가 예정
- [ ] Ahn 52.5 bar production 테이블 → 인젝터 LES

## 메커니즘 호환성 메모
- **global 2/4-step**: sub-unity 반응차수(`KERO^0.55` 등)가 농도→0에서
  Jacobian 발산 → Cantera 정상상태 flame solver 비호환. flamelet 생성 불가
- **Wang 2011 (elementary)**: 작동. 일부 중간체의 NASA fit이 3000 K까지만 —
  OpenFOAM 변환 시 Thigh 바닥 필수 (위 3단계)

## 핵심 참고문헌
Pierce & Moin 2004 JFM 504:73 (FPV) · Ihme & Pitsch 2008 CnF 155:70/90
(C 정의·UFPV) · van Oijen & de Goey 2000 CST 161:113 (평형 경계) ·
Ma, Banuti, Hickey & Ihme arXiv:1704.02639 (transcritical FPV) ·
Zips, Müller & Pfitzner 2018 FTaC 101:821 (고압 tabulation fidelity 분리) ·
Wang, Huo & Yang 2015 CST 187:60 (O₂/n-alkane counterflow, 800 K 인렛) ·
Sharma et al. 2023 PoF 35:115125 (c=Yc/Yc_eq 초임계 FGM)
