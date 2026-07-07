"""Fill the low-mdot RF-HPChung points by mdot DOWN-continuation.

Cold-start and big-jump seeding both failed for the wide-domain / thin
Takahashi flames (extinguish on coarse grids; SRK auto=False Newton stall).
The proven-robust path is SMALL mdot steps: start from a converged neighbour,
transplant onto the next (slightly wider) matched-width flame, and re-solve
HP-Chung with auto=True. Each step is a mild perturbation, so the transplanted
profile is a good guess and the grid sequencing stays lit.

Runs sequentially DOWN the ladder, each point seeded from the previous NEW
result. Saves into data/flamelets_rf525 (same family). Logs to stdout with
flush (no pipe buffering).
"""
import time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
MECH = HERE / "data/wang2011_srk_v32.yaml"
OUT = HERE / "data/flamelets_rf525"
P = 52.5e5
TIN = 800.0
FUEL = {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
OXID = {"O2": 1.0}
PV = ("CO2", "CO", "H2O", "H2")
ZST = 0.2255
MAX_GRID = 200          # low-mdot points are near-equilibrium / thick; a
# fine grid on the wide domain made the HP-Chung auto=True refinement grind
# for hours (one point stuck 2.5 h). Coarse is fine here (source is weak).

# ladder: first entry is the converged SEED (must already exist), rest are targets
LADDER = [0.153, 0.136, 0.122, 0.109, 0.0977, 0.0873,
          0.0781, 0.0699, 0.0625, 0.0559, 0.05]
IDX = {0.136: 609, 0.122: 608, 0.109: 607, 0.0977: 606, 0.0873: 605,
       0.0781: 604, 0.0699: 603, 0.0625: 602, 0.0559: 601, 0.05: 600}


def log(m):
    print(m, flush=True)


def matched_width(mdot):
    best, bw = None, 0.02
    for fp in (HERE/"data/flamelets_dualgas_MA").glob("flamelet_*.npz"):
        a = np.load(fp); dm = abs(float(a["mdot"]) - mdot)
        if best is None or dm < best:
            z = np.asarray(a["z"]); best, bw = dm, float(z.max()-z.min())
    return bw


def load_seed(mdot):
    for fp in OUT.glob("flamelet_*.npz"):
        a = np.load(fp)
        if abs(float(a["mdot"]) - mdot) < 2e-3:
            return a
    raise SystemExit(f"seed mdot={mdot} not found in {OUT}")


def save_family(f, gas, idx):
    z = f.grid.copy(); T = f.T.copy(); Yall = f.Y.copy()
    Yall = Yall/np.maximum(Yall.sum(0), 1e-12)[None, :]
    n = z.size; Mw = gas.molecular_weights
    pv = [gas.species_index(s) for s in PV if s in gas.species_names]
    rho = np.zeros(n); lam = np.zeros(n); mu = np.zeros(n)
    cp = np.zeros(n); om = np.zeros(n)
    for i in range(n):
        gas.TPY = float(T[i]), P, Yall[:, i]
        rho[i] = gas.density_mass; lam[i] = gas.thermal_conductivity
        mu[i] = gas.viscosity; cp[i] = gas.cp_mass
        w = gas.net_production_rates
        om[i] = float(sum(w[k]*Mw[k] for k in pv))/max(rho[i], 1e-30)
    Z = f.mixture_fraction("Bilger")
    alpha = lam/np.maximum(rho*cp, 1e-30)
    chi = 2.0*alpha*np.gradient(Z, z)**2
    o = np.argsort(Z)
    a = {"z": z, "Z": Z, "T": T, "rho": rho, "lam": lam, "mu": mu, "cp": cp,
         "alpha": alpha, "chi": chi,
         "chi_st": np.asarray(float(np.interp(ZST, Z[o], chi[o]))),
         "C": Yall[pv, :].sum(0), "omega_C": om,
         "mdot": np.asarray(float(f.fuel_inlet.mdot)), "P": np.asarray(P),
         "T_fuel": np.asarray(TIN), "T_ox": np.asarray(TIN),
         "Z_st_ref": np.asarray(ZST), "npts": np.asarray(int(n)),
         "Tmax": np.asarray(float(T.max())), "idx": np.asarray(int(idx)),
         "struct_transport": np.asarray("real-fluid-SRK-HPChung"),
         "meta_fixed": np.asarray("hp-chung")}
    for k, sp in enumerate(gas.species_names):
        a[f"Y_{sp}"] = Yall[k]
    np.savez_compressed(OUT/f"flamelet_{idx:03d}.npz", **a)
    log(f"  saved flamelet_{idx:03d} mdot={float(f.fuel_inlet.mdot):g} "
        f"chi_st={float(a['chi_st']):.3g} Tmax={T.max():.0f}K npts={n}")


def continue_step(prev, mdot, idx):
    gas = ct.Solution(str(MECH))
    gas.transport_model = "high-pressure-Chung"
    w = matched_width(mdot)
    f = ct.CounterflowDiffusionFlame(gas, width=w)
    f.P = P
    f.fuel_inlet.mdot = mdot; f.fuel_inlet.X = FUEL; f.fuel_inlet.T = TIN
    f.oxidizer_inlet.mdot = mdot; f.oxidizer_inlet.X = OXID
    f.oxidizer_inlet.T = TIN
    f.set_max_grid_points(f.flame, MAX_GRID)
    f.set_refine_criteria(ratio=4.0, slope=0.2, curve=0.4, prune=0.05)
    f.transport_model = "mixture-averaged"
    f.set_initial_guess()
    # transplant previous converged profile (normalized grid)
    z = np.asarray(prev["z"]); zrel = (z - z[0])/(z[-1] - z[0])
    f.set_profile("T", zrel, np.asarray(prev["T"]))
    for k, sp in enumerate(gas.species_names):
        key = f"Y_{sp}"
        if key in prev.files:
            f.set_profile(sp, zrel, np.asarray(prev[key]))
    t0 = time.time()
    # settle MA (thicker, robust) then HP-Chung -- both auto=True (grid seq)
    f.transport_model = "mixture-averaged"
    f.solve(loglevel=0, auto=True)
    log(f"  [mdot={mdot:g}] MA warm settled Tmax={f.T.max():.0f}K "
        f"({time.time()-t0:.0f}s)")
    f.transport_model = "high-pressure-Chung"
    try:
        f.solve(loglevel=0, auto=False)      # Newton on the MA-settled grid
    except Exception:
        f.solve(loglevel=0, auto=True)       # fallback: grid sequencing
    log(f"  [mdot={mdot:g}] HP-Chung Tmax={f.T.max():.0f}K "
        f"({time.time()-t0:.0f}s)")
    if f.T.max() < 1500:
        raise RuntimeError("extinguished")
    save_family(f, gas, idx)
    # return this solution as the next seed
    out = {"z": f.grid.copy(), "T": f.T.copy()}
    for k, sp in enumerate(gas.species_names):
        out[f"Y_{sp}"] = f.Y[k]
    out["files"] = list(out.keys())

    class _Seed:
        def __init__(self, d): self._d = d
        def __getitem__(self, k): return self._d[k]
        @property
        def files(self): return list(self._d.keys())
    return _Seed(out)


def main():
    seed = load_seed(LADDER[0])
    log(f"seed mdot={LADDER[0]} loaded (Tmax={float(seed['Tmax']):.0f}K)")
    for mdot in LADDER[1:]:
        idx = IDX[mdot]
        try:
            seed = continue_step(seed, mdot, idx)
        except Exception as e:
            log(f"  [mdot={mdot:g}] FAILED: {type(e).__name__}: {e} "
                "-- stopping chain (need this seed for the next step)")
            break
    log("continuation done")


if __name__ == "__main__":
    main()
