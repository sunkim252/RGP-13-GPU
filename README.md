# RGP-13-GPU: Real-Fluid Thermophysics for OpenFOAM-13 — GPU fork

> **GPU 개발 포크** — [sunkim252/RGP-13](https://github.com/sunkim252/RGP-13)
> `main`(497152e, 2026-07-08)에서 분기. 목표: ① 열물리 핫루프
> (updateTRANS/FGMTable) CUDA 포팅(`src/thermoGPU/`, Phase-1 스캐폴딩 graft됨),
> ② 선형솔버 AmgX/PETSc 오프로드.
>
> - **리모트**: `origin` = RGP-13-GPU(이 저장소) / `upstream` = RGP-13(CPU 본가)
>   / `local` = 로컬 CPU 트리. CPU 쪽 개선은 `git fetch upstream && git merge upstream/main`.
> - **브랜치**: `main` = GPU 개발 기본선, `phase1-scaffold-archive` = 2026-04 구
>   스캐폴드 히스토리 보존용 (force-push 전 main).
> - **컨테이너**: 상위 디렉토리의 `openfoam13-rgp-gpu.def`
>   (nvidia/cuda:12.4.1 베이스, wmake CUDA 룰 sm_89 포함) —
>   `apptainer build openfoam13-rgp-gpu.sif openfoam13-rgp-gpu.def`.
>   WSL2 GPU 실행: `apptainer exec --nv --bind /usr/lib/wsl <sif>` +
>   `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`.
>   케이스 실행 시 `--no-home` 권장 (호스트 `$FOAM_USER_LIBBIN`의
>   이기종 gcc 빌드 lib가 dlopen을 가로채는 것 방지).
> - **thermoGPU** (`src/thermoGPU/`): SRK+Chung 핫루프 CUDA 커널.
>   디바이스 계층(`gpu/`)은 Foam-free(플레인 배열 C ABI — OF `Scalar.H`
>   Bessel 선언과 nvcc 충돌 회피), Foam 브리지는 `rgpGpuBridge.H`.
>   빌드 `build/Allwmake`, 검증 `Test-rgpThermoGPU <thermoDict>` —
>   CPU 경로 대비 rho/mu/kappa 최대 상대오차 ~1e-11 (stableRoot 양쪽,
>   RTX 4060에서 확인).

# RGP-13: Real-Fluid Thermophysics for OpenFOAM-13

Plug-in real-gas thermophysical-properties library for supercritical
combustion in OpenFOAM-13. Built around the SRK equation of state with
two interchangeable transport backends: **Chung-Lee-Starling (1988)**
and **Ely-Hanley extended-corresponding-states (1981)**, plus optional
Péneloux volume translation for cubic-EoS density correction. Compiles
to a single shared library (`libRGP13realFluid.so`) loaded at run time
via `controlDict` — no patching of the OpenFOAM-13 source tree.

## Models

| Component | Backend | Notes |
|---|---|---|
| **Equation of state** | SRK (Soave 1972) | Optional Péneloux volume translation — constant or **temperature-dependent `c(T)`** (Péneloux et al. 1982); stable root by minimum fugacity; optional binary-interaction matrix `k_ij` |
| **Transport — Chung path** | Chung-Lee-Starling 1988 | μ, κ from Lennard-Jones combining rules; non-polar / weakly-polar fluids |
| **Transport — Ely-Hanley path** | Ely-Hanley 1981a/b ECS, **Pedersen-Fredenslund mapping** | μ, κ via HMH methane reference; corresponding-states reference state from Pedersen critical-ratio mapping + rotational-coupling α with an SRK methane reference density (replaces the LCL shape factors — see [Model updates](#model-updates)); recommended for polar fluids (H₂O) |
| **Mass diffusivity** | Fuller 1969 + Takahashi 1974 | Pressure-corrected binary D_ij; identical between the two transport paths |
| **Mixture mixing rules** | Mole-weighted pair averages | Pre-computed at construction time; one matrix set per backend |
| **Thermo** | NASA JANAF polynomials | 7-coefficient `janafThermo` with `sensibleEnthalpy` (`absoluteEnthalpy` for the FPV solver) |
| **Combustion — FPV** | Flamelet/progress-variable (Pierce & Moin 2004) | 3-D table (Z̃, gZ ≡ normalized Z″², normalized c̃) with β-PDF(Z) × δ-PDF(c) closure; **burnt end closed at adiabatic chemical equilibrium / manifold envelope, so ω̇_c(c=1) = 0 and the transported c stays bounded in [0,1]**; dedicated `fgmFluid` solver module (Wang 2018 conservation↔table loop — species held inactive, composition imposed from the table, ρ/μ/κ recomputed by SRK + Chung/EH at run time); a 4-D (…, χ_st) layout is retained for a future unsteady-FPV extension — see [FPV combustion model](#fpv-combustion-model-fgmfluid) |

## Quick start

### 1. Build the plug-in library

```bash
cd RGP-13-realFluid
./Allwmake               # auto-detects OpenFOAM-13 environment, builds libRGP13realFluid.so
./Allwmake clean         # wclean before rebuild
./Allwmake -j 8          # forwarded to wmake
FOAM_BASHRC=/opt/openfoam13/etc/bashrc ./Allwmake   # explicit env override
```

`Allwmake` searches the usual locations (`/opt/openfoam13/etc/bashrc`,
`/usr/lib/openfoam/openfoam13/etc/bashrc`,
`$HOME/OpenFOAM/OpenFOAM-13/etc/bashrc`) and auto-sources the first
match. The compiled library lands in `$FOAM_USER_LIBBIN`.

### 2. Activate in a case

```c++
// system/controlDict
libs ("libRGP13realFluid.so");
```

```c++
// constant/physicalProperties — Chung backend
thermoType
{
    type            heRhoThermo;            // or hePsiThermo
    mixture         SRKchungTakaMixture;
    transport       chung;
    thermo          janaf;
    energy          sensibleEnthalpy;
    equationOfState SRKGas;
    specie          rfSpecie;
}
```

```c++
// constant/physicalProperties — Ely-Hanley backend
thermoType
{
    type            heRhoThermo;
    mixture         SRKelyHanleyMixture;    // ← native EH mixture
    transport       elyHanley;
    thermo          janaf;
    energy          sensibleEnthalpy;
    equationOfState SRKGas;
    specie          rfSpecie;
}
```

The two backends have **separate mixture classes** so that each one
drives its own native `updateTRANS` signature (no silent argument
discarding, no Pitzer-Z_c fallback for the Ely-Hanley side). The SRK
EoS routines are shared.

### 3. Per-species critical data

Each species needs an `rfProperties` sub-dictionary in the JANAF input
file:

```c++
O2
{
    specie       { molWeight 31.998; }
    rfProperties
    {
        Tc      154.58;            // critical T [K]
        Pc      5.043e6;           // critical p [Pa]
        Vc      73.37;             // critical V [cm^3/mol]   (rfSpecie unit)
        omega   0.0222;            // acentric factor [-]
        miui    0.0;               // dipole moment [Debye]    (only used by Chung)
        kappai  0.0;               // association factor [-]   (only used by Chung)
        sigmvi  16.6;              // Fuller diffusion volume [-]
        // optional Péneloux:
        penelouxShift true;        //   auto via Spencer-Danner Rackett
        // c   7.89e-3;            //   OR explicit shift in m^3/kmol
    }
    thermodynamics { /* JANAF Tlow Thigh Tcommon highCpCoeffs lowCpCoeffs */ }
    transport      { /* unused — kept empty for parser */ }
}
```

Optional binary-interaction matrix (top level of the same dict):

```c++
binaryInteraction
{
    O2_H2O  { kij -0.015; }
    H2_N2   { kij  0.093; }
}
```

Recommended set for the LOX/kerosene FPV species (N2, CO2, O2, CO, KERO,
H2O — KERO treated as an n-decane analog, M = 140.3, Tc = 635 K). Shipped in
the FGM test-case `thermo.compressibleGas4S` dicts:

| pair | kij | provenance |
|---|---|---|
| N2_O2   | -0.0119 | cryogenic-VLE compilations (Knapp et al. 1982 class) |
| N2_CO2  | -0.017  | same |
| O2_CO2  |  0.10   | estimate |
| O2_H2O  | -0.015  | repo reference value |
| CO2_H2O |  0.12   | T-dependent (0.07-0.3); combustion-range mean |
| KERO_N2 |  0.13   | N2/n-decane |
| KERO_CO2|  0.114  | CO2/n-decane |
| KERO_O2 |  0.10   | estimate (N2 analog) |
| KERO_CO |  0.10   | estimate (CO ~ N2) |
| KERO_H2O|  0.45   | water/alkane nonideality (VLE-class) |

Unlisted pairs (H2, remaining product pairs) default to 0 — those species only
appear in hot, low-density regions where the attraction correction is
negligible. Sensitivity at 52-55 bar: mixing-layer density shifts by up to
-5.2 % (O2/KERO 70/30 by mass, 200 K); pure streams and 1500 K products are
unaffected (0.0 %).

## Source layout

```
RGP-13-realFluid/
├── Allwmake / Allwclean              auto-detect OpenFOAM env + build
├── install.sh                        legacy patch-into-OpenFOAM-tree variant
├── Make/{files,options}              one-library wmake target
├── README.md
├── docs/                             FGM porting guide, Phase-2 update notes
├── src/
│   ├── rfSpecie/                     base specie class with critical-property accessors
│   ├── SRKGas/                       SRK equation of state + Péneloux shift
│   ├── chungTransport/               Chung-Lee-Starling 1988 μ, κ
│   ├── elyHanleyTransport/           Ely-Hanley ECS μ, κ — Pedersen-Fredenslund mapping
│   │       ├── elyHanleyTransport.{H,C,I.H}
│   │       └── (SRK methane reference density; C^2 quintic soft-clamp on T_o, ρ_o)
│   ├── SRKchungTakaMixture/          mixture class for the Chung backend
│   ├── SRKelyHanleyMixture/          mixture class for the Ely-Hanley backend
│   ├── FGM/                          FGMTable: 3-D/4-D FPV lookup (tri-/quadrilinear), multi-field (sourcePV, T, Y_k)
│   ├── include/                      forRealFluidGases* template-dispatch macros
│   └── realFluidThermos.C            explicit template instantiation
├── applications/modules/fgmFluid/    FPV solver module (transported Z̃ + normalized c̃,
│                                       algebraic gZ closure, Pitsch-Steiner χ_st,
│                                       inactive-species composition from the table)
├── tools/fgm_table_gen/              Cantera FPV table generator (dual-gas flamelets,
│                                       equilibrium/envelope closure — see its README)
├── testCases/
│   ├── bunsenFlame_FGM/              minimal premixed FGM combustion test
│   ├── counterflow_fgmFluid/         1-D opposed-jet FPV verification case
│   └── wang2015_P50atm/, _P1atm/     Wang-Huo-Yang (2015) pressure-sweep verification
└── codeTemplates/                    OpenFOAM dynamicCode templates (carried over)
```

## Choosing a transport backend

| Fluid class | Recommended | Why |
|---|---|---|
| Non-polar, near-spherical (O₂, N₂, CH₄, H₂, alkanes) | **Chung** | Direct empirical fit; sweet spot of the Chung correlation |
| Strongly polar (H₂O, NH₃, alcohols) | **Ely-Hanley** | Methane-reference X_λ correction reduces H₂O κ RMSE 96 % → 24 % vs CoolProp |
| Mixtures with both | Use Ely-Hanley if any major component is polar | Both classes are continuous in composition; no step jumps when species fractions change |

The two backends are bit-identical for **non-polar** species; switching
between them is safe for the LOX/LH₂ and LOX/CH₄ regimes.

## Model updates

Validated against NIST/REFPROP and CoolProp HEOS over 50–500 bar
(`test/*_CT`, `test/*` Ely-Hanley sweeps). MAPE = mean absolute percentage
error vs reference.

### Ely-Hanley — Pedersen-Fredenslund corresponding states

The reference-state mapping was replaced. The textbook **Leach-Chappelear-
Leland (1968) shape factors** (`phiShape`/`thetaShape`, retained as dead code
for provenance) pushed the methane reference into the wrong phase near a
species' pseudo-critical line — e.g. LCL drove `T_o` below methane's `T_c`
into the liquid branch while the target fluid was already gas-like, giving a
spurious μ/κ "bump" (CO₂ ≈ 300–360 K, H₂O ≈ 700–800 K) and over-predicting μ
by 3–4×.

The current path uses the **Pedersen-Fredenslund** critical-ratio mapping:

```
T_Ref = T · Tc_CH4 / Tc_mix      P_Ref = p · Pc_CH4 / Pc_mix
```

so the methane reference changes phase at the *same reduced state* as the
target, plus a Pedersen **rotational-coupling factor** `α = 1 + c·ρ_R^1.847·MW^k`.
The methane reference **density** comes from the SRK EoS (Cardano cubic with
minimum-fugacity root selection) rather than an ideal/empirical estimate, so
the dense-liquid reference is physical (`X_η = 1`; the Ely Enskog correction
is absorbed into α). The HMH (Hanley-McCarty-Haynes 1975 / 1977) methane
reference μ/κ correlations are unchanged.

μ MAPE, old LCL/Oefelein → Pedersen: **O₂ 33 → 2.9 %, CO₂ 67 → 5.9 %,
H₂ 16 → 8.6 %, H₂O 86 → 23 %, KERO 219 → 27 %, n-C₁₂ 173 → 57 %**; κ
improves comparably. The pseudo-critical bumps are removed.

### Chung — discontinuity fix + acentric coefficient

- **κ jump removed.** Chung's internal-mode α uses the *ideal-gas* Cv₀. The
  thermo chain returns `Cp_real = Cp_ig + Cp_dep(SRK)`, so evaluating it at
  1 atm picks up a compressed-liquid SRK departure below a species' 1-atm
  boiling point (H₂O at 373 K, O₂ at 90 K) and produces a κ discontinuity
  (e.g. H₂O at ≈ 380 K). `Cp_ig` is now taken at a low reference pressure
  (`p_ref = 100 Pa`) where every species stays single-phase vapour. Mirrored
  in the Ely-Hanley path.
- **Acentric coefficient corrected** in the κ β-term to **1.3168** (Poling
  5th ed. Eq. 10-3.14; the previous 1.1368 was a transposition — negligible
  for low-ω O₂ but matters for H₂O ω = 0.344 and kerosene ω = 0.467).

Current Chung MAPE (pressure-averaged): O₂ ρ 0.7 / μ 2.1 / κ 4.2 / Cp 0.6 %;
CO₂ 1.3 / 2.7 / 7.6 / 1.0 %; H₂O 4.0 / 12 / 41 / 6.8 % (κ degrades for the
strongly H-bonded liquid — use Ely-Hanley there).

### EoS — temperature-dependent Péneloux `c(T)`

A constant Péneloux shift matches one reference temperature, but the SRK
liquid-density error grows with T (H₂O drifts to ≈ −10 % by 500 K even when
calibrated at 300 K). An optional quadratic `c(T)` ramp closes this:

```c++
rfProperties
{
    c              5.9448e-3;                    // baseline shift [m^3/kmol]
    penelouxCoeffs (9.6031e-3 -2.5316e-5 4.5560e-8 580 760); // cq0 cq1 cq2 Tlo Thi
    // c(T) = cq0 + cq1·T + cq2·T^2 for T ≤ Tlo, smoothstep-ramped to the
    // constant baseline c over [Tlo, Thi]; constant c above Thi.
}
```

Applied to H₂O this brings the 300–1000 K liquid-density MAPE from 7.7 % to
0.8 %. Species without `penelouxCoeffs` fall back to the constant `c`.

### Hot-path optimisation (both backends, bit-identical)

Composition-only quantities are precomputed once per cell in `updateTRANS`
(Pedersen Tc_mix/Pc_mix combining loop, mixture constants), and the
per-(p, T) reference evaluation is cached so the consecutive `mu(p,T)` and
`kappa(p,T)` calls on one cell share a single solve — the Pedersen
`srkMethaneRho` cubics for Ely-Hanley, and the `rho(p,T)` cubic plus
collision integral Ω* for Chung. Verified bit-for-bit identical to the
pre-optimisation results (ρ/Cp exactly 0, μ/κ at floating-point reorder
level). The cache is `mutable` and assumes OpenFOAM's serial per-process
cell loop.

### FPV combustion model (`fgmFluid`)

Flamelet/progress-variable combustion for supercritical LOX/kerosene,
verified end-to-end in a 1-D opposed-jet case at 50 atm
(T_max = 3636 K burning, transported c̃ exactly bounded in [0, 1]).

* **Manifold**: steady counterflow diffusion flamelets (Cantera, Wang 2011
  skeletal kerosene, 106 sp) parameterized as a 3-D table
  (Z̃, gZ, c̃) — the full strain family fills the (Z, c) plane. For *steady*
  manifolds the progress variable and χ_st index the same one-parameter
  family, so a χ axis is redundant; the 4-D (Z̃, gZ, c̃, χ_st) layout and the
  solver-side Pitsch-Steiner χ_st closure are retained for a future
  **unsteady-FPV** extension (Ihme & Pitsch 2008), where transient
  extinction/re-ignition states populate the (c, χ) plane.
* **Burnt-end closure / boundedness**: the table's c = 1 boundary is the
  **adiabatic chemical equilibrium** (HP-equilibrate per Z̃; *not*
  complete-combustion/Burke–Schumann, which is a non-physical ≈6000 K
  state), generalized to the **manifold envelope**
  `C_norm(Z) = max(C_eq, family max)` because cross-Z diffusion holds the
  steady reaction zone a few % above the *local* equilibrium. ω̇_c ≡ 0 on
  the c = 1 row, plus a solver-side clamp c ∈ [0, 1] (the explicit source
  can step through the zero in one Δt) — the standard layering
  (Pierce & Moin 2004 library truncation; van Oijen & de Goey 2000
  equilibrium boundary; cf. ANSYS Fluent FGM, PelePhysics Manifold).
* **Dual-gas flamelet generation** (`tools/fgm_table_gen/`): the flame
  *structure* is solved ideal-gas + unity-Lewis (consistent with the
  solver's unity-Schmidt c̃/Z̃ transport; the SRK steady Newton is
  intractable on resolved grids at trans-critical conditions), then ρ, μ, κ,
  c_p, ω̇_c and χ_st are **re-evaluated pointwise with SRK +
  high-pressure-Chung** before tabulation — consistent with the run-time
  thermophysics, and the same fidelity split as Zips, Müller & Pfitzner
  (2018).
* **JANAF pitfall**: the multicomponent mixture clamps the temperature
  solve at the *minimum* `Thigh` over **all** species — kerosene-mechanism
  intermediates fitted only to 3000 K pinned the whole flame at exactly
  3000.0 K. The YAML→OpenFOAM converter floors `Thigh` at 5000 K
  (benign: those intermediates vanish above ~1500 K).

## Numerical robustness (Ely-Hanley specifics)

The Hanley-McCarty-Haynes methane reference μ/κ correlations diverge outside
their fitted range. With the Pedersen mapping the reference state `(T_o, ρ_o)`
comes from the SRK methane density and is physical, but to keep μ(p, T),
κ(p, T) C¹/C² continuous on the entire CFD envelope two **C² quintic Hermite
soft-clamps** (identity on the interior; value-, slope- and curvature-
continuous at every junction) bound the reference state fed to the HMH
correlation (`elyHanleyTransportI.H` `softClamp`):

- methane reference temperature `T_o ∈ [40, 2000] K` (half-width w = 30)
- methane reference density `ρ_o ∈ [0, 569] kg/m³` (= 3.5·ρ_c,CH₄; half-width w = 80)

The earlier Leach-Chappelear-Leland shape-factor soft-clamps on `V_r, T_r`
are no longer in the active path — the Pedersen critical-ratio mapping
replaces them; `phiShape`/`thetaShape` remain only as dead code for
provenance. See `RGP-13_ECS_deviations.ipynb` (in the parent repo) for the
full enumeration of corrections and their derivations.

## References

1. Soave, G. *Equilibrium constants from a modified Redlich-Kwong
   equation of state*. Chem. Eng. Sci. **27** (1972) 1197.
2. Péneloux, A., Rauzy, E. & Frèze, R. *A consistent correction for
   Redlich-Kwong-Soave volumes*. Fluid Phase Equilib. **8** (1982) 7.
3. Spencer, C.F. & Danner, R.P. *Improved equation for prediction of
   saturated liquid density*. J. Chem. Eng. Data **17** (1972) 236.
4. Leach, J.W., Chappelear, P.S. & Leland, T.W. *Use of molecular
   shape factors in vapor-liquid equilibrium calculations with the
   corresponding states principle*. AIChE J. **14** (1968) 568.
5. Hanley, H.J.M., McCarty, R.D. & Haynes, W.M. *Equations for the
   viscosity and thermal conductivity coefficients of methane*.
   Cryogenics **15** (1975) 413; and Hanley, H.J.M., Haynes, W.M. &
   McCarty, R.D. *The viscosity and thermal conductivity coefficients
   for dense gaseous and liquid methane*. J. Phys. Chem. Ref. Data
   **6** (1977) 597.
6. Ely, J.F. & Hanley, H.J.M. *Prediction of transport properties.
   1. Viscosity of fluids and mixtures*. Ind. Eng. Chem. Fundam.
   **20** (1981) 323.
7. Ely, J.F. & Hanley, H.J.M. *Prediction of transport properties.
   2. Thermal conductivity of pure fluids and mixtures*. Ind. Eng.
   Chem. Fundam. **22** (1983) 90.
8. Ely, J.F. *An Enskog correction for size and mass difference effects
   in mixture viscosity prediction*. J. Res. Natl. Bur. Stand. **86**
   (1981) 597.
9. Chung, T.-H., Ajlan, M., Lee, L.L. & Starling, K.E. *Generalized
   multiparameter correlation for nonpolar and polar fluid transport
   properties*. Ind. Eng. Chem. Res. **27** (1988) 671.
10. Fuller, E.N., Schettler, P.D. & Giddings, J.C. *A new method for
    prediction of binary gas-phase diffusion coefficients*. Ind. Eng.
    Chem. **58** (1966) 18.
11. Takahashi, S. *Preparation of a generalized chart for the diffusion
    coefficients of gases at high pressures*. J. Chem. Eng. Jpn.
    **7** (1974) 417.
12. Bell, I.H., Wronski, J., Quoilin, S. & Lemort, V. *Pure and
    pseudo-pure fluid thermophysical property evaluation and the
    open-source thermophysical property library CoolProp*. Ind. Eng.
    Chem. Res. **53** (2014) 2498.
13. Oefelein, J.C. *Advances in modeling supercritical fluid behavior
    and combustion in high-pressure propulsion systems*. AIAA Paper
    2019-0634 (SciTech 2019).
14. Pedersen, K.S. & Fredenslund, Aa. *An improved corresponding states
    model for the prediction of oil and gas viscosities and thermal
    conductivities*. Chem. Eng. Sci. **42** (1987) 182.

## Acknowledgments

Based on [realFluidFoam-8](https://github.com/danhnam11/realFluidFoam-8)
by Nam Danh Nguyen, Combustion & Propulsion Lab, UNIST
(Prof. Chun Sang Yoo). Ported to OpenFOAM-13 with API adaptation,
performance optimisations, Phase-1 (Péneloux + binary-interaction
parsing) and Phase-2 (Ely-Hanley standalone, separate mixture class,
C² boundary smoothing) extensions.
