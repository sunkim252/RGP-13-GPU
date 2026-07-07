"""PURE real-fluid FPV family sweep: SRK EOS + high-pressure-Chung transport
for the flamelet STRUCTURE itself (no dual-gas split), saved in the standard
family npz schema so 04_build_fgm_table_4d.py consumes it unchanged.

Differences from the production dual-gas path (05_flamelet_sweep_dualgas):
  * structure solved on the SRK gas with 'high-pressure-Chung' (Chung mu/kappa
    + Takahashi high-pressure diffusion) -- the fluid the solver actually sees;
  * property/omega evaluation uses the SAME gas object => single-gas
    consistency (dual-gas re-evaluated an ideal-gas structure with SRK);
  * two-stage solve per point (proven in rf_flamelet_sg.py): ignite+resolve
    with plain mixture-averaged first (thicker flame survives coarse grids),
    then switch to high-pressure-Chung and re-solve from the burning state.
    EOS is SRK in BOTH stages.
Both inlets 800 K (Wang, Huo & Yang 2015 rationale, same as production).

Usage:
  python3 rf_sweep_family.py [--transport high-pressure-Chung|multicomponent]
                             <idx0> <mdot1> [mdot2 ...]
Output: data/flamelets_rf525[_MC]/flamelet_<idx>.npz  (family schema)

--transport multicomponent: stage-2 uses the full Stefan-Maxwell diffusion
matrix (no Takahashi high-pressure correction -- Cantera has no HP-MC model),
still on the SRK EOS. Comparing the two RF families isolates the MA-vs-MC
closure difference at identical EOS/conditions.
"""
import sys, time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
MECH = HERE / "data/wang2011_srk_v32.yaml"
TRANSPORT2 = "high-pressure-Chung"
STAGE2_NOAUTO = "--stage2-noauto" in sys.argv
if STAGE2_NOAUTO:
    sys.argv.remove("--stage2-noauto")
# --stage1-only: save the SRK + plain mixture-averaged solution itself
# (the "RF-MA" family). Completes the factorial decomposition:
#   dual-gas MA -> RF-MA   isolates EOS-in-structure,
#   RF-MA -> RF-MC         isolates the MA->MC closure,
#   RF-MA -> RF-HPChung    isolates the Chung/Takahashi HP correction.
STAGE1_ONLY = "--stage1-only" in sys.argv
if STAGE1_ONLY:
    sys.argv.remove("--stage1-only")
SEED_RFMA = "--seed-rfma" in sys.argv
if SEED_RFMA:
    sys.argv.remove("--seed-rfma")
if "--transport" in sys.argv:
    _i = sys.argv.index("--transport")
    TRANSPORT2 = sys.argv[_i+1]
    del sys.argv[_i:_i+2]
OUT = HERE / ("data/flamelets_rf525" if TRANSPORT2 == "high-pressure-Chung"
              else "data/flamelets_rf525_MC")
TAG = ("real-fluid-SRK-HPChung" if TRANSPORT2 == "high-pressure-Chung"
       else "real-fluid-SRK-MC")
if "--stage1-only" in sys.argv or globals().get("STAGE1_ONLY"):
    pass  # resolved after flag parse below
P = 52.5e5
TIN = 800.0
WIDTH = 0.02          # fallback; per-mdot width matched to the dual-gas
# family below. The production dual-gas ladder was built with mdot
# CONTINUATION + solve(auto=True), whose internal domain re-fitting shrank
# the width per point (0.02 -> 0.001 m across the ladder). A counterflow
# problem is defined by the (mdot, width) PAIR, so a fixed-width RF sweep at
# the same mdot solves a much WEAKER strain (measured: chi_st 0.59 vs 3.6 at
# mdot=0.298). Match each point's geometry to the dual-gas file so the RF
# families overlay on the same chi ladder / manifold coordinates.


def matched_width(mdot):
    import glob as _g
    best, bw = None, WIDTH
    for fpath in _g.glob(str(HERE/"data/flamelets_dualgas_MA/flamelet_*.npz")):
        a = np.load(fpath)
        dm = abs(float(a["mdot"]) - mdot)
        if best is None or dm < best:
            z = np.asarray(a["z"])
            best, bw = dm, float(z.max() - z.min())
    return bw
FUEL = {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
OXID = {"O2": 1.0}
PV = ("CO2", "CO", "H2O", "H2")
ZST = 0.2255
MAX_GRID = 600


def log(m):
    print(m, flush=True)


def save_family(f, gas, idx):
    """Save in the steady-family npz schema; props/omega from the SRK gas."""
    z = f.grid.copy()
    T = f.T.copy()
    Yall = f.Y.copy()                      # (ns, n)
    sumY = Yall.sum(axis=0)
    Yall = Yall / np.maximum(sumY, 1e-12)[None, :]
    n = z.size
    Mw = gas.molecular_weights
    pv_idx = [gas.species_index(s) for s in PV if s in gas.species_names]
    rho = np.zeros(n); lam = np.zeros(n); mu = np.zeros(n)
    cp = np.zeros(n); omega_C = np.zeros(n)
    for i in range(n):
        gas.TPY = float(T[i]), P, Yall[:, i]
        rho[i] = gas.density_mass
        lam[i] = gas.thermal_conductivity
        mu[i] = gas.viscosity
        cp[i] = gas.cp_mass
        wdot = gas.net_production_rates
        omega_C[i] = float(sum(wdot[k]*Mw[k] for k in pv_idx)) \
            / max(rho[i], 1e-30)
    Z = f.mixture_fraction("Bilger")
    alpha = lam/np.maximum(rho*cp, 1e-30)
    dZdx = np.gradient(Z, z)
    chi = 2.0*alpha*dZdx**2
    # chi at the stoichiometric crossing (interp on Z, robust to direction)
    order = np.argsort(Z)
    chi_st = float(np.interp(ZST, Z[order], chi[order]))
    arrs = {
        "z": z, "Z": Z, "T": T, "rho": rho, "lam": lam, "mu": mu, "cp": cp,
        "alpha": alpha, "chi": chi, "chi_st": np.asarray(chi_st),
        "C": Yall[pv_idx, :].sum(axis=0),
        "omega_C": omega_C,
        "mdot": np.asarray(float(f.fuel_inlet.mdot)),
        "P": np.asarray(P),
        "T_fuel": np.asarray(TIN), "T_ox": np.asarray(TIN),
        "Z_st_ref": np.asarray(ZST),
        "npts": np.asarray(int(n)),
        "Tmax": np.asarray(float(T.max())),
        "idx": np.asarray(int(idx)),
        "struct_transport": np.asarray(TAG),
    }
    for k, sp in enumerate(gas.species_names):
        arrs[f"Y_{sp}"] = Yall[k]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/f"flamelet_{idx:03d}.npz"
    np.savez_compressed(path, **arrs)
    log(f"  saved {path.name} npts={n} chi_st={chi_st:.3g}/s "
        f"Tmax={T.max():.0f}K (pure-RF {TAG})")


IDEAL_MECH = HERE / "data/wang2011_ideal_v32.yaml"


def _new_srk_flame(mdot):
    gas = ct.Solution(str(MECH))
    # save_family evaluates lam/mu/alpha/chi on THIS gas object: pin its
    # transport to the family's stage-2 model so the saved metadata (incl.
    # chi_st) is consistent with the structure transport. (Discovered: the
    # yaml default left HPChung-family metadata evaluated with plain MA.)
    gas.transport_model = ("mixture-averaged" if STAGE1_ONLY else TRANSPORT2)
    f = ct.CounterflowDiffusionFlame(gas, width=matched_width(mdot))
    f.P = P
    f.fuel_inlet.mdot = mdot; f.fuel_inlet.X = FUEL; f.fuel_inlet.T = TIN
    f.oxidizer_inlet.mdot = mdot; f.oxidizer_inlet.X = OXID
    f.oxidizer_inlet.T = TIN
    f.set_max_grid_points(f.flame, MAX_GRID)
    f.set_refine_criteria(ratio=3.0, slope=0.12, curve=0.24, prune=0.03)
    return f, gas


def _ideal_bootstrap(mdot, t0):
    """Stage-0 fallback (user-suggested continuation): converge the flame on
    the IDEAL-GAS mechanism first (the path the whole dual-gas family proved
    robust), then transplant the profile onto a fresh SRK flame and re-solve
    with SRK + high-pressure-Chung."""
    gi = ct.Solution(str(IDEAL_MECH))
    fi = ct.CounterflowDiffusionFlame(gi, width=matched_width(mdot))
    fi.P = P
    fi.fuel_inlet.mdot = mdot; fi.fuel_inlet.X = FUEL; fi.fuel_inlet.T = TIN
    fi.oxidizer_inlet.mdot = mdot; fi.oxidizer_inlet.X = OXID
    fi.oxidizer_inlet.T = TIN
    fi.set_max_grid_points(fi.flame, MAX_GRID)
    fi.set_refine_criteria(ratio=3.0, slope=0.12, curve=0.24, prune=0.03)
    fi.transport_model = "mixture-averaged"
    fi.set_initial_guess()
    fi.solve(loglevel=0, auto=True)
    log(f"[mdot={mdot:g}] stage0 IDEAL: Tmax={fi.T.max():.0f}K "
        f"npts={fi.grid.size} ({time.time()-t0:.0f}s)")
    # transplant profile onto the SRK flame (normalized grid)
    f, gas = _new_srk_flame(mdot)
    g0, g1 = fi.grid[0], fi.grid[-1]
    zrel = (fi.grid - g0)/(g1 - g0)
    f.transport_model = "mixture-averaged"
    f.set_initial_guess()
    f.set_profile("velocity", zrel, fi.velocity)
    f.set_profile("T", zrel, fi.T)
    for k, sp in enumerate(gi.species_names):
        f.set_profile(sp, zrel, fi.Y[k])
    f.solve(loglevel=0, auto=True)      # SRK + MA from warm ideal state
    log(f"[mdot={mdot:g}] stage0b SRK-MA warm: Tmax={f.T.max():.0f}K "
        f"({time.time()-t0:.0f}s)")
    return f, gas


def _seed_from_rfma(mdot):
    """Plan-B for stubborn low-mdot (wide-domain) points where the cold SRK+MA
    stage-1 Newton fails nondeterministically: the RF-MA family already holds
    that CONVERGED structure at the matched width, so transplant it onto a
    fresh SRK flame and go straight to stage-2 target transport."""
    import glob as _g
    best, bp = None, None
    for fp in _g.glob(str(HERE/"data/flamelets_rf525_MAonly/flamelet_*.npz")):
        a = np.load(fp); dm = abs(float(a["mdot"]) - mdot)
        if best is None or dm < best:
            best, bp = dm, fp
    if best is None or best > 1e-3:
        raise RuntimeError(f"no RF-MA seed within tol (nearest {best})")
    a = np.load(bp)
    f, gas = _new_srk_flame(mdot)
    z = np.asarray(a["z"]); zrel = (z - z[0])/(z[-1] - z[0])
    f.transport_model = "mixture-averaged"
    f.set_initial_guess()
    # velocity from stored (u = mdot/rho); fall back to guess if absent
    f.set_profile("T", zrel, np.asarray(a["T"]))
    for k, sp in enumerate(gas.species_names):
        key = f"Y_{sp}"
        if key in a.files:
            f.set_profile(sp, zrel, np.asarray(a[key]))
    # NO auto=False settle here -- that reproduces the historical SRK Newton
    # stall (hung 10 min). The transplanted MA profile is a good guess; the
    # caller goes straight to the target transport with auto=True (robust grid
    # sequencing).
    log(f"[mdot={mdot:g}] seed-RFMA transplanted (Tmax_seed={f.T.max():.0f}K)")
    return f, gas


def solve_one(mdot, idx):
    t0 = time.time()
    f = gas = None
    if SEED_RFMA:
        try:
            f, gas = _seed_from_rfma(mdot)
        except Exception as e:
            log(f"[mdot={mdot:g}] seed-RFMA failed ({e})"); return
        f.transport_model = TRANSPORT2
        f.solve(loglevel=0, auto=True)
        log(f"[mdot={mdot:g}] stage2 {TRANSPORT2}: Tmax={f.T.max():.0f}K "
            f"({time.time()-t0:.0f}s)")
        if f.T.max() < 1500:
            log(f"[mdot={mdot:g}] cold after seed -- not saved"); return
        save_family(f, gas, idx); return
    try:
        f, gas = _new_srk_flame(mdot)
        # stage 1: plain mixture-averaged (robust ignition on coarse grids)
        f.transport_model = "mixture-averaged"
        f.set_initial_guess()
        f.solve(loglevel=0, auto=True)
        if f.T.max() < 1500:
            raise RuntimeError("stage1 extinguished")
        log(f"[mdot={mdot:g}] stage1 MA: Tmax={f.T.max():.0f}K "
            f"npts={f.grid.size} ({time.time()-t0:.0f}s)")
    except Exception as e:
        log(f"[mdot={mdot:g}] stage1 direct failed ({e}) -> ideal bootstrap")
        f, gas = _ideal_bootstrap(mdot, t0)
        if f.T.max() < 1500:
            log(f"[mdot={mdot:g}] bootstrap also cold -- giving up")
            return
    if STAGE1_ONLY:
        save_family(f, gas, idx)
        return
    # stage 2: target transport, re-solve from burning state.
    # auto=True redoes the full coarse->fine sequencing with the EXPENSIVE
    # transport (observed: 75+ min without completion); from a converged
    # same-grid MA state a plain Newton (auto=False) is the right move --
    # the historical "SRK auto=False stall" was for FROM-SCRATCH settles.
    f.transport_model = TRANSPORT2
    if STAGE2_NOAUTO:
        try:
            f.solve(loglevel=0, refine_grid=True, auto=False)
        except Exception as e:
            log(f"[mdot={mdot:g}] stage2 noauto failed ({e}) -> auto=True")
            f.solve(loglevel=0, auto=True)
    else:
        f.solve(loglevel=0, auto=True)
    log(f"[mdot={mdot:g}] stage2 {TRANSPORT2}: Tmax={f.T.max():.0f}K "
        f"npts={f.grid.size} ({time.time()-t0:.0f}s)")
    if f.T.max() < 1500:
        log(f"[mdot={mdot:g}] EXTINGUISHED after {TRANSPORT2} -- not saved")
        return
    save_family(f, gas, idx)


if STAGE1_ONLY:
    OUT = HERE / "data/flamelets_rf525_MAonly"
    TAG = "real-fluid-SRK-MA"


def main():
    idx0 = int(sys.argv[1])
    mdots = [float(x) for x in sys.argv[2:]]
    for k, mdot in enumerate(mdots):
        try:
            solve_one(mdot, idx0 + k)
        except Exception as e:
            log(f"[mdot={mdot:g}] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
