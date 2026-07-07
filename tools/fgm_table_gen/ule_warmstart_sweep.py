"""unity-Le upper branch via PER-POINT MA warm-start (52.5 bar).

Two-point control cannot even be ARMED on a unity-Le flame at these
conditions (settle never 'declared'; arming solve collapses), so the ule
branch is mapped point-by-point instead: at each mdot the robust
mixture-averaged flame is converged first, then transport is switched to
unity-Lewis on that structure and settled (production-pipeline recipe:
short transient cap + stationarity acceptance). Each ule point is an
INDEPENDENT warm start -- no ule-to-ule continuation fragility.

Deliverable: matched-mdot (T_max, a, chi_st) for MA and ule up to the
MA-reachable strain range (chi ~ 1e6/s), quantifying the structural
transport difference under identical numerics; plus the mdot (if any)
beyond which the ule structure genuinely cannot hold.

Run: python3 ule_warmstart_sweep.py
"""
import json, time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
YAML = HERE / "data/wang2011_ideal_v32.yaml"
P = 52.5e5
T_IN = 800.0
WIDTH = 0.01
ZST = 0.2255
BURN_T = 1500.0


def log(m):
    print(m, flush=True)


def chi_st_of(f):
    Z = f.mixture_fraction("Bilger")
    D = f.thermal_conductivity/np.maximum(f.density*f.cp_mass, 1e-30)
    chi = 2.0*D*np.gradient(Z, f.grid)**2
    return float(chi[int(np.argmin(np.abs(Z - ZST)))])


def mean_strain(f):
    gas = f.gas
    gas.TPX = f.fuel_inlet.T, f.P, f.fuel_inlet.X
    rf = gas.density
    gas.TPX = f.oxidizer_inlet.T, f.P, f.oxidizer_inlet.X
    ro = gas.density
    return (f.fuel_inlet.mdot/rf + f.oxidizer_inlet.mdot/ro)/WIDTH


def settle_ule(f):
    """Pipeline recipe: cap transient, accept stationary burning state."""
    f.transport_model = "unity-Lewis-number"
    try:
        f.max_time_step_count = 200
    except Exception:
        pass
    Tprev = None
    for rnd in range(1, 6):
        declared = True
        try:
            f.solve(loglevel=0, auto=False)
        except Exception:
            declared = False
            if f.T.max() < BURN_T:
                return False, float(f.T.max())
        Tm = float(f.T.max())
        if declared or (Tprev is not None and abs(Tm - Tprev) < 2.0):
            return Tm > BURN_T, Tm
        Tprev = Tm
    return f.T.max() > BURN_T, float(f.T.max())


def main():
    out = HERE/"data/rf_scurve"
    gas = ct.Solution(str(YAML))
    gas.transport_model = "mixture-averaged"
    f = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
    f.P = P
    f.fuel_inlet.X = {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
    f.fuel_inlet.T = T_IN
    f.oxidizer_inlet.X = "O2:1.0"
    f.oxidizer_inlet.T = T_IN
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.03)

    mdot = 0.5
    f.fuel_inlet.mdot = mdot
    f.oxidizer_inlet.mdot = mdot
    t0 = time.time()
    log(f"[uwarm] MA base at mdot={mdot} ...")
    f.solve(loglevel=0, auto=True)
    log(f"[uwarm] base T_max={f.T.max():.0f}K npts={f.grid.size} "
        f"({time.time()-t0:.0f}s)")

    rows = []   # mdot, T_ma, a_ma, chi_ma, ule_ok, T_ule, a_ule, chi_ule
    ma_state = f.to_array()
    k = 0
    while mdot < 2.0e4:
        # --- ule warm-start at this mdot (independent per point) ---
        ma_state = f.to_array()
        T_ma = float(f.T.max()); a_ma = mean_strain(f); c_ma = chi_st_of(f)
        ok, T_ule = settle_ule(f)
        a_ule = mean_strain(f) if ok else float("nan")
        c_ule = chi_st_of(f) if ok else float("nan")
        rows.append((mdot, T_ma, a_ma, c_ma, bool(ok), T_ule, a_ule, c_ule))
        log(f"[uwarm] mdot={mdot:9.3g}  MA: T={T_ma:6.0f} chi={c_ma:11.1f} | "
            f"ule: {'BURN' if ok else 'LOST'} T={T_ule:6.0f} "
            f"chi={c_ule if ok else float('nan'):11.1f} "
            f"({time.time()-t0:.0f}s)")
        # restore MA and march. settle_ule leaves max_time_step_count=200 on
        # the Sim1D -- restore a full transient budget or the MA continuation
        # dies immediately ("maximum number of timesteps (200)").
        f.from_array(ma_state)
        f.transport_model = "mixture-averaged"
        try:
            f.max_time_step_count = 1000
        except Exception:
            pass
        mdot *= 1.8
        f.fuel_inlet.mdot = mdot
        f.oxidizer_inlet.mdot = mdot
        k += 1
        try:
            f.solve(loglevel=0, auto=(k % 3 == 0))
        except Exception:
            # one retry with full auto (regrid) before declaring reach limit
            try:
                f.from_array(ma_state)
                f.fuel_inlet.mdot = mdot
                f.oxidizer_inlet.mdot = mdot
                f.solve(loglevel=0, auto=True)
            except Exception as e:
                log(f"[uwarm] MA solver fail at mdot={mdot:.3g} "
                    f"({type(e).__name__}) -- stop (MA reach limit)")
                break
        if f.T.max() < BURN_T:
            log(f"[uwarm] MA lost at mdot={mdot:.3g} -- stop")
            break

    with open(out/"ule_warmstart_matched.json", "w") as fh:
        json.dump({"P_bar": P/1e5, "T_in": T_IN, "width": WIDTH,
                   "rows_mdot_Tma_ama_chima_uleok_Tule_aule_chiule": rows},
                  fh, indent=1)
    log(f"[uwarm] saved ule_warmstart_matched.json ({len(rows)} points, "
        f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
