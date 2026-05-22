"""Generate an ANALYTIC (synthetic) flamelet library for pipeline testing.

Produces data/flamelets_synth.npz with the SAME layout as the merged output of
03_flamelet_sweep_mpi.py, so that 04_build_fgm_table.py and the OpenFOAM-side
FGM code can be developed and verified WITHOUT running Cantera.

Model (FPV-like, not physically calibrated):
  - mixture fraction Z in [0, 1], stoichiometric Z_st.
  - equilibrium progress variable profile C_eq(Z): beta-shaped bump peaking
    at Z_st, zero at Z=0 (pure oxidizer) and Z=1 (pure fuel).
  - each flamelet k has a burning level lambda_k in [0,1]:
        C_k(Z)       = lambda_k * C_eq(Z)
        omega_C_k(Z) = A * lambda_k*(1-lambda_k) * shape(Z)
    so omega_C is zero for mixing (lambda=0) and equilibrium (lambda=1),
    peaks at intermediate burning, and the family sweeps the (Z, C) plane.

Run (any python with numpy; no Cantera needed):
    python make_synthetic_flamelets.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)
OUT = DATA_DIR / "flamelets_synth.npz"

# ---- Model parameters ----
Z_ST = 0.2          # stoichiometric mixture fraction
C_MAX = 0.30        # peak equilibrium progress variable (Y_CO2+CO+H2O+H2)
OMEGA_SCALE = 1.0e3 # peak source magnitude [1/s]
N_FLAMELETS = 20    # number of strain/burning levels
NZ = 120            # points per flamelet in Z
T_OX = 120.0
T_FUEL = 300.0
T_AD = 3600.0       # synthetic adiabatic flame temperature
P_OPER = 100.0e5
PV_SPECIES = ("CO2", "CO", "H2O", "H2")

# Species set matching the real-fluid OpenFOAM thermo (O2_phase2 mech).
# Composition is reconstructed analytically so that sum(Y)=1 and
# C = Y_CO2 + Y_CO + Y_H2O holds exactly:
#   reactants (mixing line, scaled by 1-C):  KERO=Z(1-C), O2=(1-Z)(1-C)
#   products (scaled by C):                  CO2=0.6C, H2O=0.3C, CO=0.1C
#   N2 = 0  (pure-O2 oxidizer)
THERMO_SPECIES = ("N2", "CO2", "O2", "CO", "KERO", "H2O")
PROD_SPLIT = {"CO2": 0.6, "H2O": 0.3, "CO": 0.1}  # sums to 1.0

# beta-shape exponents giving a bump peaking at Z_ST:  Z^a (1-Z)^b,
# peak at a/(a+b) = Z_ST -> b = a*(1-Z_ST)/Z_ST.
_A = 2.0
_B = _A * (1.0 - Z_ST) / Z_ST


def _bump(Z):
    """Beta-shaped bump, peaks at Z_ST with value 1, zero at Z=0,1."""
    peak = (Z_ST ** _A) * ((1.0 - Z_ST) ** _B)
    return (Z ** _A) * ((1.0 - Z) ** _B) / peak


def main():
    Z = np.linspace(0.0, 1.0, NZ)
    C_eq = C_MAX * _bump(Z)
    shape = _bump(Z)

    lambdas = np.linspace(0.0, 1.0, N_FLAMELETS)

    out = {
        "P": np.asarray(P_OPER),
        "T_fuel": np.asarray(T_FUEL),
        "X_fuel": np.asarray("SYNTHETIC"),
        "X_ox": np.asarray("O2:1.0"),
        "pv_species": np.asarray(",".join(PV_SPECIES)),
        "width": np.asarray(0.02),
        "n_flamelets": np.asarray(N_FLAMELETS),
    }
    idx_list, mdots = [], []

    for k, lam in enumerate(lambdas):
        C = lam * C_eq
        omega_C = OMEGA_SCALE * lam * (1.0 - lam) * shape
        T = T_OX + (T_AD - T_OX) * lam * shape
        rho = P_OPER / (287.0 * np.maximum(T, 1.0))  # crude ideal-gas-ish

        # Analytic composition (sum = 1, consistent with C = CO2+CO+H2O)
        Yk = {
            "KERO": Z * (1.0 - C),
            "O2":   (1.0 - Z) * (1.0 - C),
            "N2":   np.zeros_like(Z),
        }
        for sp, frac in PROD_SPLIT.items():
            Yk[sp] = frac * C

        pre = f"f{k:03d}_"
        out[pre + "Z"] = Z
        out[pre + "C"] = C
        out[pre + "omega_C"] = omega_C
        out[pre + "T"] = T
        out[pre + "rho"] = rho
        out[pre + "z"] = Z * 0.02            # dummy physical coordinate
        for sp in THERMO_SPECIES:
            out[pre + f"Y_{sp}"] = Yk[sp]
        out[pre + "mdot"] = np.asarray(float(k + 1))
        out[pre + "idx"] = np.asarray(int(k))
        out[pre + "npts"] = np.asarray(int(NZ))
        out[pre + "Tmax"] = np.asarray(float(T.max()))
        idx_list.append(k)
        mdots.append(float(k + 1))

    out["idx_list"] = np.asarray(idx_list, dtype=np.int32)
    out["mdots"] = np.asarray(mdots, dtype=np.float64)

    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}")
    print(f"  {N_FLAMELETS} flamelets, NZ={NZ}, Z_st={Z_ST}, C_max={C_MAX}")
    print(f"  C range [0, {C_eq.max():.3f}], omega_C peak "
          f"{(OMEGA_SCALE*0.25*shape.max()):.3g} 1/s")


if __name__ == "__main__":
    main()
