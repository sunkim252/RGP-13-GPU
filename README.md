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
| **Equation of state** | SRK (Soave 1972) | Optional Péneloux volume translation (Péneloux et al. 1982); optional binary-interaction matrix `k_ij` |
| **Transport — Chung path** | Chung-Lee-Starling 1988 | μ, κ from Lennard-Jones combining rules; non-polar / weakly-polar fluids |
| **Transport — Ely-Hanley path** | Ely-Hanley 1981a/b ECS | μ, κ via methane reference + Leach-Chappelear-Leland 1968 shape factors + Ely 1981 Enskog X correction; recommended for polar fluids (H₂O) |
| **Mass diffusivity** | Fuller 1969 + Takahashi 1974 | Pressure-corrected binary D_ij; identical between the two transport paths |
| **Mixture mixing rules** | Mole-weighted pair averages | Pre-computed at construction time; one matrix set per backend |
| **Thermo** | NASA JANAF polynomials | 7-coefficient `janafThermo` with `sensibleEnthalpy` |

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
│   ├── elyHanleyTransport/           Ely-Hanley ECS μ, κ (standalone, no Chung fallback)
│   │       ├── elyHanleyTransport.{H,C,I.H}
│   │       └── (C^2 quintic soft-clamp on V_r, T_r, ρ_r boundaries)
│   ├── SRKchungTakaMixture/          mixture class for the Chung backend
│   ├── SRKelyHanleyMixture/          mixture class for the Ely-Hanley backend
│   ├── FGM/                          flamelet-generated-manifold combustion model
│   ├── include/                      forRealFluidGases* template-dispatch macros
│   └── realFluidThermos.C            explicit template instantiation
├── testCases/
│   └── bunsenFlame_FGM/              minimal FGM combustion test
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

## Numerical robustness (Ely-Hanley specifics)

The textbook Leach 1968 / Hanley-McCarty-Haynes 1975 correlations
diverge outside their fitted range. To keep μ(p, T), κ(p, T) C¹
continuous on the entire CFD operating envelope the Ely-Hanley
implementation applies **C² quintic Hermite soft-clamps** on:

- shape-factor input `V_r, T_r ∈ [0.5, 2.0]`  (`elyHanleyTransportI.H` `phiShape`/`thetaShape`, half-width w = 0.25)
- methane reference temperature `T_o ∈ [1, 10⁵] K` (loose hard guard only — Sutherland is bounded)
- methane reduced density `ρ_r ∈ [0, 3]` for the cubic dense-correction (`muRef`/`lambdaRef`, half-width w = 0.30)

Each soft-clamp is value-, slope-, and curvature-continuous at every
junction. See `RGP-13_ECS_deviations.ipynb` (in the parent repo) for
the full enumeration of corrections and their derivations.

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

## Acknowledgments

Based on [realFluidFoam-8](https://github.com/danhnam11/realFluidFoam-8)
by Nam Danh Nguyen, Combustion & Propulsion Lab, UNIST
(Prof. Chun Sang Yoo). Ported to OpenFOAM-13 with API adaptation,
performance optimisations, Phase-1 (Péneloux + binary-interaction
parsing) and Phase-2 (Ely-Hanley standalone, separate mixture class,
C² boundary smoothing) extensions.
