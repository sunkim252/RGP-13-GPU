"""Validate the DEPLOYMENT (dual-gas MA) structure against the Wang et al.
2015 counterflow diffusion-burner data (Fig19): T_max / delta(FWHM) vs
pressure at a=1000. Dual-gas structure = ideal-gas EOS + mixture-averaged
transport (exactly what the v4 deployment family uses for the flame
structure). Contrasts against the already-validated RF-HPChung result
(rf_fig19_a1000.json) and the Wang digitized data.

Same session geometry as rf_flamelet_sg (0.02 m domain, a=100*U_ox, equal
mdot both inlets) so the strain axis matches. Ideal-gas solve is cheap, so we
sweep pressures directly.

Usage: python3 dualgas_fig19.py
Output: data/dualgas_fig19.json  (+ printed table for the overlay)
"""
import json, time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
MECH = HERE / "data/wang2011_ideal_v32.yaml"
PS_ATM = [10, 25, 50, 100]
A = 1000.0
TIN = 800.0
WIDTH = 0.02
FUEL = {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
OXID = {"O2": 1.0}


def log(m):
    print(m, flush=True)


def fwhm(x, y, base):
    yy = y - base; pk = yy.max()
    if pk <= 0:
        return float("nan")
    half = base + 0.5*pk; idx = np.where(y >= half)[0]
    if len(idx) < 2:
        return float("nan")
    iL, iR = idx[0], idx[-1]
    xL = x[iL] if iL == 0 else np.interp(half, [y[iL-1], y[iL]], [x[iL-1], x[iL]])
    xR = x[iR] if iR == len(x)-1 else np.interp(half, [y[iR], y[iR+1]], [x[iR], x[iR+1]])
    return xR - xL


def solve(P_atm):
    P = P_atm*101325.0
    gas = ct.Solution(str(MECH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = TIN, P, OXID; rho_ox = gas.density
    mdot = A*rho_ox/100.0
    f = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
    f.P = P
    f.fuel_inlet.mdot = mdot; f.fuel_inlet.X = FUEL; f.fuel_inlet.T = TIN
    f.oxidizer_inlet.mdot = mdot; f.oxidizer_inlet.X = OXID
    f.oxidizer_inlet.T = TIN
    f.set_max_grid_points(f.flame, 600)
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.02)
    f.set_initial_guess()
    t0 = time.time()
    f.solve(loglevel=0, auto=True)
    Tmax = float(f.T.max())
    dT = fwhm(f.grid, f.T, TIN)
    q = float(np.trapz(f.heat_release_rate, f.grid))
    try:
        pf_ox = float(f.strain_rate("potential_flow_oxidizer"))
    except Exception:
        pf_ox = float("nan")
    log(f"P={P_atm}atm: Tmax={Tmax:.0f}K delta={dT*1e3:.3f}mm q={q:.3e} "
        f"pf_ox={pf_ox:.0f} ({time.time()-t0:.0f}s)")
    return {"P": P_atm, "Tmax": Tmax, "delta_mm": dT*1e3, "q": q,
            "pf_ox": pf_ox, "mdot": float(mdot)}


def solve_cont(P_atm, prev):
    """Pressure continuation: seed from the previous (lower-P) converged flame
    so the high-P a=1000 point does not cold-extinguish."""
    P = P_atm*101325.0
    gas = ct.Solution(str(MECH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = TIN, P, OXID; rho_ox = gas.density
    mdot = A*rho_ox/100.0
    f = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
    f.P = P
    f.fuel_inlet.mdot = mdot; f.fuel_inlet.X = FUEL; f.fuel_inlet.T = TIN
    f.oxidizer_inlet.mdot = mdot; f.oxidizer_inlet.X = OXID
    f.oxidizer_inlet.T = TIN
    f.set_max_grid_points(f.flame, 600)
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.02)
    f.set_initial_guess()
    z = np.asarray(prev["grid"]); zr = (z-z[0])/(z[-1]-z[0])
    f.set_profile("T", zr, np.asarray(prev["T"]))
    for k, sp in enumerate(gas.species_names):
        if f"Y_{sp}" in prev:
            f.set_profile(sp, zr, np.asarray(prev[f"Y_{sp}"]))
    t0 = time.time()
    f.solve(loglevel=0, auto=True)
    Tmax = float(f.T.max()); dT = fwhm(f.grid, f.T, TIN)
    q = float(np.trapz(f.heat_release_rate, f.grid))
    try: pf = float(f.strain_rate("potential_flow_oxidizer"))
    except Exception: pf = float("nan")
    log(f"P={P_atm}atm(cont): Tmax={Tmax:.0f}K delta={dT*1e3:.3f}mm "
        f"q={q:.3e} pf_ox={pf:.0f} ({time.time()-t0:.0f}s)")
    prof = {"grid": f.grid.copy(), "T": f.T.copy()}
    for k, sp in enumerate(gas.species_names):
        prof[f"Y_{sp}"] = f.Y[k]
    return ({"P": P_atm, "Tmax": Tmax, "delta_mm": dT*1e3, "q": q,
             "pf_ox": pf, "mdot": float(mdot)}, prof, f)


def main():
    out = {}
    # base: 10 atm cold (works), then pressure-continue upward
    P = 10*101325.0
    gas = ct.Solution(str(MECH)); gas.transport_model = "mixture-averaged"
    gas.TPX = TIN, P, OXID; rho_ox = gas.density; mdot = A*rho_ox/100.0
    f = ct.CounterflowDiffusionFlame(gas, width=WIDTH); f.P = P
    f.fuel_inlet.mdot = mdot; f.fuel_inlet.X = FUEL; f.fuel_inlet.T = TIN
    f.oxidizer_inlet.mdot = mdot; f.oxidizer_inlet.X = OXID; f.oxidizer_inlet.T = TIN
    f.set_max_grid_points(f.flame, 600)
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.02)
    f.set_initial_guess(); f.solve(loglevel=0, auto=True)
    dT = fwhm(f.grid, f.T, TIN)
    out["10"] = {"P": 10, "Tmax": float(f.T.max()), "delta_mm": dT*1e3,
                 "q": float(np.trapz(f.heat_release_rate, f.grid)),
                 "pf_ox": float(f.strain_rate("potential_flow_oxidizer")),
                 "mdot": float(mdot)}
    log(f"P=10atm base Tmax={f.T.max():.0f}K")
    prof = {"grid": f.grid.copy(), "T": f.T.copy()}
    for k, sp in enumerate(gas.species_names): prof[f"Y_{sp}"] = f.Y[k]
    for P_atm in [25, 50, 100]:
        try:
            rec, prof, _ = solve_cont(P_atm, prof)
            out[str(P_atm)] = rec
        except Exception as e:
            log(f"P={P_atm}atm cont FAILED: {e}"); break
    with open(HERE/"data/dualgas_fig19.json", "w") as fp:
        json.dump(out, fp, indent=1)
    log("saved data/dualgas_fig19.json")


if __name__ == "__main__":
    main()
