"""Generate counterflow diffusion flamelets for Wang & Yang 2017 conditions.

Operating point (Wang & Yang 2017, J. Propulsion & Power, RD-0110-like):
    P     = 100 bar
    T_ox  = 120 K  (LOX, transcritical -- clamped if NASA-7 polynomial fails)
    T_fuel= 300 K  (kerosene)
    fuel  = 74% NC10H22 + 15% PHC3H7 + 11% CYC9H18 (mol/vol)
    ox    = 100% O2

Pipeline:
    1) Load wang2011_srk.yaml (from step 0)
    2) Build CounterflowDiffusionFlame, solve at low strain (burning)
    3) Sweep strain rate up to (near) extinction
    4) Save each flamelet's (grid, T, Y_k, sourcePV) to HDF5

Run from Windows side (ct-env2):
    (ct-env2)$ python 02_generate_flamelets.py            # SMOKE: 1 flamelet
    (ct-env2)$ python 02_generate_flamelets.py --sweep    # full sweep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cantera as ct
except ImportError as e:
    sys.exit(
        "Cantera not importable. Activate ct-env2 first.\n"
        f"({e})"
    )

try:
    import h5py
except ImportError:
    h5py = None  # optional: only needed for --sweep


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SRK_YAML = DATA_DIR / "wang2011_srk.yaml"
FLAMELET_DIR = DATA_DIR / "flamelets"
FLAMELET_DIR.mkdir(exist_ok=True, parents=True)

# Cached warmup state. Single Cantera YAML containing the baseline flamelet
# at (P_OPER, T_ox_target, mdot_init) — produced once by `--warmup`,
# consumed by MPI sweep workers via `--load`.
WARMUP_FLAME = DATA_DIR / "warmup_flame.yaml"
WARMUP_NAME = "baseline"


# ---- Operating conditions ----
P_OPER = 100.0e5         # Pa
T_OX_TARGET = 120.0      # K  (will clamp upward if NASA-7 polynomial unhappy)
T_OX_CLAMP_MIN = 200.0   # K  (typical lower bound of NASA-7 polynomial)
T_FUEL = 300.0           # K

X_FUEL = "NC10H22:0.74, PHC3H7:0.15, CYC9H18:0.11"
X_OX = "O2:1.0"

# Counterflow geometry: separation between inlets [m]
DOMAIN_WIDTH = 0.02

# Progress variable definition: C = Y_CO2 + Y_CO + Y_H2O + Y_H2
PV_SPECIES = ("CO2", "CO", "H2O", "H2")


def make_gas() -> ct.Solution:
    if not SRK_YAML.exists():
        sys.exit(f"missing {SRK_YAML}. Run 01_convert_chemkin.py first.")
    return ct.Solution(str(SRK_YAML))


def resolve_T_ox(gas: ct.Solution) -> float:
    """Pick a T_ox that NASA-7 polynomial can evaluate. Try the physical
    value first; if it fails, clamp upward.
    """
    for T_try in (T_OX_TARGET, T_OX_CLAMP_MIN):
        try:
            gas.TPX = T_try, P_OPER, X_OX
            _ = gas.cp_mass  # forces polynomial eval
            print(f"[oxid] T_ox = {T_try} K  (rho={gas.density:.2f} kg/m^3)")
            return T_try
        except Exception as e:
            print(f"[oxid] T_ox={T_try} K failed: {e}")
    sys.exit("could not pick a valid T_ox.")


def _density(flame) -> np.ndarray:
    """Mass density along flame grid (handle naming across Cantera versions)."""
    for attr in ("density_mass", "density"):
        v = getattr(flame, attr, None)
        if v is not None:
            return np.asarray(v)
    raise AttributeError("flame has neither density_mass nor density")


def progress_variable(flame, gas) -> np.ndarray:
    """C = sum_k Y_k for k in PV_SPECIES, evaluated along the flame grid."""
    Y = np.zeros(flame.grid.size)
    for sp in PV_SPECIES:
        Y += flame.Y[gas.species_index(sp)]
    return Y


def source_pv(flame, gas) -> np.ndarray:
    """omega_dot_C = sum_k W_k * omega_dot_mol_k / rho   for k in PV_SPECIES.

    Returns mass-based source [1/s] (Y_k production rate).
    """
    omega_mol = flame.net_production_rates   # shape (n_species, n_pts), kmol/m^3/s
    Mw = gas.molecular_weights               # kg/kmol
    src = np.zeros(flame.grid.size)
    for sp in PV_SPECIES:
        k = gas.species_index(sp)
        src += omega_mol[k] * Mw[k]          # kg/m^3/s
    rho = _density(flame)
    return src / np.maximum(rho, 1e-30)      # 1/s


def _configure(flame, P, T_ox, mdot):
    flame.P = P
    flame.fuel_inlet.T = T_FUEL
    flame.fuel_inlet.X = X_FUEL
    flame.fuel_inlet.mdot = mdot
    flame.oxidizer_inlet.T = T_ox
    flame.oxidizer_inlet.X = X_OX
    flame.oxidizer_inlet.mdot = mdot


def warmup_solve(gas, T_ox_target, mdot, log=0):
    """Solve at the target operating point via low-P continuation.

    Sequence:
      1. P=1 atm, T_ox=300 K, mdot=0.05 (easy ideal-gas-like)  -> auto solve
      2. ramp P up to P_OPER via intermediate stops
      3. ramp T_ox down to T_ox_target if it is below 300 K
      4. ramp mdot to mdot
    """
    flame = ct.CounterflowDiffusionFlame(gas, width=DOMAIN_WIDTH)
    flame.transport_model = "mixture-averaged"
    flame.set_refine_criteria(ratio=4.0, slope=0.2, curve=0.3, prune=0.04)

    # ---- Stage 1: easy initial solve ----
    P_seq = [1.0e5, 5.0e5, 20.0e5, 50.0e5, P_OPER]
    T_seq_after = [280.0, 250.0, 220.0, T_ox_target] if T_ox_target < 280 else [T_ox_target]
    mdot_init = 0.05

    print(f"[warmup] stage 1: P=1 atm, T_ox=300 K, mdot={mdot_init} ...")
    _configure(flame, P=P_seq[0], T_ox=300.0, mdot=mdot_init)
    t0 = time.time()
    flame.solve(loglevel=log, auto=True)
    print(f"[warmup] stage 1 OK  npts={flame.grid.size}  Tmax={flame.T.max():.1f} K  "
          f"dt={time.time()-t0:.1f}s")

    # ---- Stage 2: ramp pressure ----
    for P in P_seq[1:]:
        print(f"[warmup] ramp P -> {P/1e5:.0f} bar ...")
        flame.P = P
        t0 = time.time()
        flame.solve(loglevel=log, auto=True)
        print(f"[warmup]   OK  npts={flame.grid.size}  Tmax={flame.T.max():.1f} K  "
              f"dt={time.time()-t0:.1f}s")

    # ---- Stage 3: ramp T_ox down ----
    for T_ox in T_seq_after:
        if abs(T_ox - flame.oxidizer_inlet.T) < 1.0:
            continue
        print(f"[warmup] ramp T_ox -> {T_ox:.0f} K ...")
        flame.oxidizer_inlet.T = T_ox
        t0 = time.time()
        flame.solve(loglevel=log, auto=True)
        print(f"[warmup]   OK  npts={flame.grid.size}  Tmax={flame.T.max():.1f} K  "
              f"dt={time.time()-t0:.1f}s")

    # ---- Stage 4: ramp mdot to requested value ----
    if mdot > mdot_init * 1.1:
        # geometric ramp from current to target
        ratios = np.geomspace(mdot_init, mdot, num=5)[1:]
        for m in ratios:
            print(f"[warmup] ramp mdot -> {m:.3f} ...")
            flame.fuel_inlet.mdot = m
            flame.oxidizer_inlet.mdot = m
            t0 = time.time()
            flame.solve(loglevel=log, auto=True)
            print(f"[warmup]   OK  npts={flame.grid.size}  Tmax={flame.T.max():.1f} K  "
                  f"dt={time.time()-t0:.1f}s")

    # ---- Final refine ----
    flame.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.02)
    flame.solve(loglevel=log, auto=False)
    print(f"[warmup] final  npts={flame.grid.size}  Tmax={flame.T.max():.1f} K")
    return flame


def solve_one_flamelet(
    gas: ct.Solution,
    T_ox: float,
    mdot: float,
    refine: bool = True,
    log: int = 0,
    init_from: ct.CounterflowDiffusionFlame | None = None,
) -> ct.CounterflowDiffusionFlame:
    """Solve a single flamelet at (P_OPER, T_ox, mdot). If init_from is None,
    runs warmup continuation; otherwise reuses the previous flame's state
    (caller is responsible for mutating mdot and re-solving).
    """
    if init_from is None:
        return warmup_solve(gas, T_ox, mdot, log=log)
    flame = init_from
    flame.fuel_inlet.mdot = mdot
    flame.oxidizer_inlet.mdot = mdot
    t0 = time.time()
    flame.solve(loglevel=log, auto=True)
    print(f"[flame] mdot={mdot:6.3f}  npts={flame.grid.size:4d}  "
          f"Tmax={flame.T.max():6.1f} K  dt={time.time()-t0:5.1f}s")
    return flame


def smoke_test():
    gas = make_gas()
    T_ox = resolve_T_ox(gas)
    print(f"[smoke] solving single flamelet at P=100 bar, T_ox={T_ox} K "
          f"(via warm-up continuation)")
    flame = solve_one_flamelet(gas, T_ox, mdot=0.2, log=1)

    C = progress_variable(flame, gas)
    omega_C = source_pv(flame, gas)
    rho = _density(flame)
    print(f"[smoke] grid range:  {flame.grid[0]*1000:6.3f} - {flame.grid[-1]*1000:6.3f} mm")
    print(f"[smoke] T range:     {flame.T.min():6.1f} - {flame.T.max():6.1f} K")
    print(f"[smoke] C range:     {C.min():.4f} - {C.max():.4f}")
    print(f"[smoke] omega_C max: {omega_C.max():.3e}  (1/s, mass-fraction)")
    print(f"[smoke] rho range:   {rho.min():6.2f} - {rho.max():6.2f} kg/m^3")

    # Save single flamelet for inspection
    out = FLAMELET_DIR / "smoke_flamelet.csv"
    arr = np.column_stack([flame.grid, flame.T, rho, C, omega_C])
    np.savetxt(out, arr,
               header="z[m]  T[K]  rho[kg/m^3]  C[-]  omega_C[1/s]",
               fmt="%.6e")
    print(f"[smoke] OK -> {out}")


def sweep():
    if h5py is None:
        sys.exit("h5py not installed in ct-env2. Run:\n  pip install h5py")
    gas = make_gas()
    T_ox = resolve_T_ox(gas)

    # Strain rate sweep: ramp mdot up to near extinction.
    # Start low (near equilibrium), step geometric.
    mdots = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    h5_path = DATA_DIR / "flamelets.h5"
    print(f"[sweep] -> {h5_path}")
    with h5py.File(h5_path, "w") as f:
        f.attrs.update(P=P_OPER, T_ox=T_ox, T_fuel=T_FUEL,
                       X_fuel=X_FUEL, X_ox=X_OX,
                       pv_species=",".join(PV_SPECIES),
                       width=DOMAIN_WIDTH)
        prev = None
        for k, mdot in enumerate(mdots):
            try:
                flame = solve_one_flamelet(
                    gas, T_ox, mdot=mdot, log=0, init_from=prev
                )
            except ct.CanteraError as e:
                print(f"[sweep] mdot={mdot}: extinction or solver failure ({e})")
                break
            prev = flame
            g = f.create_group(f"flamelet_{k:02d}")
            g.attrs.update(mdot=mdot, npts=flame.grid.size,
                           Tmax=float(flame.T.max()))
            g["z"] = flame.grid
            g["T"] = flame.T
            g["rho"] = _density(flame)
            g["C"] = progress_variable(flame, gas)
            g["omega_C"] = source_pv(flame, gas)
            for sp in PV_SPECIES + ("O2", "NC10H22", "PHC3H7", "CYC9H18", "N2"):
                if sp in gas.species_names:
                    g[f"Y_{sp}"] = flame.Y[gas.species_index(sp)]
            prev = flame
    print(f"[sweep] done -> {h5_path}")


def warmup_only(mdot_init=0.05):
    """Run continuation warm-up and save the baseline flame to YAML so MPI
    sweep workers (03_flamelet_sweep_mpi.py) can pick it up and avoid
    repeating the serial continuation.
    """
    gas = make_gas()
    T_ox = resolve_T_ox(gas)
    print(f"[warmup-only] target P={P_OPER/1e5:.0f} bar, T_ox={T_ox} K, "
          f"mdot={mdot_init}")
    flame = warmup_solve(gas, T_ox, mdot=mdot_init, log=1)

    print(f"[warmup-only] saving baseline -> {WARMUP_FLAME}")
    # Cantera 3.x: save(filename, name=..., description=...)
    flame.save(
        str(WARMUP_FLAME),
        name=WARMUP_NAME,
        description=(
            f"warm-up baseline at P={P_OPER:.1f} Pa, T_ox={T_ox} K, "
            f"T_fuel={T_FUEL} K, mdot={mdot_init} kg/m^2/s"
        ),
        overwrite=True,
    )
    print(f"[warmup-only] done")


def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--smoke", action="store_true",
                     help="single flamelet (default) with full warmup")
    grp.add_argument("--warmup", action="store_true",
                     help="run warmup continuation and save baseline state "
                          "to data/warmup_flame.yaml (for MPI sweep)")
    grp.add_argument("--sweep", action="store_true",
                     help="sequential strain-rate sweep (saves flamelets.h5)")
    args = p.parse_args()
    if args.sweep:
        sweep()
    elif args.warmup:
        warmup_only()
    else:
        smoke_test()


if __name__ == "__main__":
    main()
