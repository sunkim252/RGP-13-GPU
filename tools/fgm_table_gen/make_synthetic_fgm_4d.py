"""Generate a small, fully analytic 4-D fgmProperties dictionary so the
counterflow_fgmFluid case (and the rest of the fgmFluid solver path) can be
smoke-tested before a real Cantera-based 4-D table is available.

Axes are deliberately coarse (nZ=11, nGz=5, nC=11, nChi=5) so the dict file
stays small enough to scan by eye. Field shapes are bell-curve products of
a normalised progress lambda = C/C_eq(Z) and a chi-modulation that mimics
extinction at high scalar dissipation:

    omega_C   ~  A * lambda*(1-lambda) * shape(Z) * chi_mod(chi)
    T         ~  T_ox + (T_max - T_ox) * lambda * shape(Z) * chi_mod(chi)
    Y_CO2     ~  yield * shape(Z) * chi_mod(chi) * lambda
    Y_H2O     ~  yield * shape(Z) * chi_mod(chi) * lambda
    Y_O2      ~  (1 - lambda) * (1 - Z)
    Y_KERO    ~  (1 - lambda) * Z
    Y_CO      ~  small fraction of yield
    Y_N2      ~  remainder so sum(Y) ≈ 1

The peak is centred at Z_st (default 0.0625 for LOX/kerosene). The chi
modulation is

    chi_mod(chi) = 1 / (1 + (chi/chi_q)^2)

with chi_q the synthetic quenching strain rate.

Run:
    python make_synthetic_fgm_4d.py
        --out ../testCases/counterflow_fgmFluid/constant/fgmProperties_synth4d
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Reuse the writer from the real builder so the dictionary format stays
# bit-for-bit identical to what FGMTable expects.
import importlib.util
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "build4d", HERE / "04_build_fgm_table_4d.py"
)
build4d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build4d)
write_fgm_dict = build4d.write_fgm_dict
write_fgm_npz  = build4d.write_fgm_npz

# ---- Axes (small) ----
N_Z   = 11
N_G   =  5
N_C   = 11
N_CHI =  5

# ---- Synthetic constants ----
Z_ST     = 0.0625
SIGMA_Z  = 0.06       # Gaussian width of shape(Z) around Z_st
C_MAX    = 0.30
T_OX     = 120.0
T_FUEL   = 300.0
T_PEAK   = 3700.0
CHI_Q    = 50.0       # 1/s, "quenching" chi at which the source drops sharply
CHI_LO   = 0.1
CHI_HI   = 500.0
A_SRC    = 200.0      # peak omega_C amplitude [1/s]

SPECIES  = ["CO", "CO2", "H2O", "KERO", "N2", "O2"]


def shape_Z(Z):
    return np.exp(-0.5 * ((Z - Z_ST)/SIGMA_Z)**2)


def chi_mod(chi):
    # smooth Lorentzian-ish fall-off; chi <= chi_q -> ~1, chi >> chi_q -> 0
    r = chi / CHI_Q
    return 1.0 / (1.0 + r*r)


def Ceq(Z):
    # mirror the 3-D synthetic table: bell-shaped progress ceiling
    return C_MAX * shape_Z(Z) / shape_Z(np.array(Z_ST))


def build_tables():
    Z_axis  = np.linspace(0.0, 1.0, N_Z)
    g_axis  = np.linspace(0.0, 0.9, N_G)
    C_axis  = np.linspace(0.0, C_MAX, N_C)
    chi_axis = np.geomspace(CHI_LO, CHI_HI, N_CHI)

    # Pre-compute broadcasted axis grids
    Zg, _, Cg, Kg = np.meshgrid(Z_axis, g_axis, C_axis, chi_axis,
                                indexing="ij")
    # lambda = C / C_eq(Z); clamp into [0, 1]
    Ce = Ceq(Zg)
    lam = np.clip(Cg / np.maximum(Ce, 1e-9), 0.0, 1.0)
    sZ  = shape_Z(Zg)
    cm  = chi_mod(Kg)

    tables = {}
    tables["omega_C"] = A_SRC * lam * (1.0 - lam) * sZ * cm
    tables["T"]       = T_OX + (T_PEAK - T_OX) * lam * sZ * cm
    # Reactant side: linear mixing at lambda=0, full consumption at lambda=1
    Y_O2   = (1.0 - lam) * (1.0 - Zg)
    Y_KERO = (1.0 - lam) * Zg
    # Product yields scaled by chi_mod so high chi keeps reactants
    yield_ = lam * sZ * cm
    Y_CO2  = 0.40 * yield_
    Y_H2O  = 0.40 * yield_
    Y_CO   = 0.05 * yield_
    rem    = np.clip(1.0 - (Y_O2 + Y_KERO + Y_CO2 + Y_H2O + Y_CO), 0.0, 1.0)
    Y_N2   = rem
    for sp, arr in zip(SPECIES,
                       [Y_CO, Y_CO2, Y_H2O, Y_KERO, Y_N2, Y_O2]):
        tables[f"Y_{sp}"] = arr

    print(f"[synth4d] axes nZ={N_Z} nGz={N_G} nC={N_C} nChi={N_CHI}  "
          f"chi=[{chi_axis[0]:.2g}, {chi_axis[-1]:.2g}] 1/s")
    for f, arr in tables.items():
        print(f"[synth4d]   {f}: range "
              f"[{arr.min():.3g}, {arr.max():.3g}]")
    return Z_axis, g_axis, C_axis, chi_axis, tables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output dict path")
    args = p.parse_args()

    Z_axis, g_axis, C_axis, chi_axis, tables = build_tables()
    out_dict = Path(args.out)
    meta = {"src": "make_synthetic_fgm_4d.py", "P": 1.0e7}
    write_fgm_dict(out_dict, Z_axis, g_axis, C_axis, chi_axis,
                   tables, SPECIES, meta)
    out_npz = out_dict.with_suffix(out_dict.suffix + ".npz") \
              if out_dict.suffix else out_dict.with_name(out_dict.name + ".npz")
    write_fgm_npz(out_npz, Z_axis, g_axis, C_axis, chi_axis,
                  tables, SPECIES, meta)


if __name__ == "__main__":
    main()
