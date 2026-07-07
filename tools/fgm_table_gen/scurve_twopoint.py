"""S-curve through the extinction fold via Cantera two-point flame control.

Rigor upgrades over extinction_le_sweep.py (which detects extinction as
solver-failure of an mdot-continuation):
  * two-point control makes the inlet mass fluxes eigenvalues while imposing
    the temperature at two control points -> marching the control temperature
    DOWN traces the S-curve THROUGH the fold onto the middle branch, so the
    extinction point is the observed strain maximum (fold), not a solver-fail.
  * --width / refine options enable the domain-width & grid convergence study
    at the nose (a scales with width; chi_st should be ~invariant).
  * --transport ule bootstraps unity-Lewis from a converged mixture-averaged
    solution (the production pipeline's own warm-start trick), giving the
    unity-Le TRUE extinction under identical definitions.

Outputs: data/rf_scurve/tp_<tag>.json (+ curve npz) with per-step
(T_max, a_mean, chi_st, control T).

Run:
  python3 scurve_twopoint.py --transport ma  --width 0.01
  python3 scurve_twopoint.py --transport ule --width 0.01
  python3 scurve_twopoint.py --transport ma  --width 0.005
  python3 scurve_twopoint.py --transport ma  --width 0.02
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
YAML = HERE / "data/wang2011_ideal_v32.yaml"
P = 52.5e5
T_IN = 800.0
ZST = 0.2255


def log(m):
    print(m, flush=True)


def mean_strain(f, width):
    gas = f.gas
    gas.TPX = f.fuel_inlet.T, f.P, f.fuel_inlet.X
    rho_f = gas.density
    gas.TPX = f.oxidizer_inlet.T, f.P, f.oxidizer_inlet.X
    rho_o = gas.density
    return (f.fuel_inlet.mdot/rho_f + f.oxidizer_inlet.mdot/rho_o)/width


def chi_st_of(f):
    Z = f.mixture_fraction("Bilger")
    y = f.grid
    D = f.thermal_conductivity/np.maximum(f.density*f.cp_mass, 1e-30)
    dZdy = np.gradient(Z, y)
    chi = 2.0*D*dZdy**2
    i = int(np.argmin(np.abs(Z - ZST)))
    return float(chi[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=("ma", "ule"), required=True)
    ap.add_argument("--width", type=float, default=0.01)
    ap.add_argument("--mdot0", type=float, default=0.5)
    ap.add_argument("--dT0", type=float, default=20.0)
    ap.add_argument("--dTmin", type=float, default=0.5)
    ap.add_argument("--nmax", type=int, default=500)
    ap.add_argument("--slope", type=float, default=0.1)
    ap.add_argument("--curve", type=float, default=0.2)
    ap.add_argument("--stopT", type=float, default=2000.0,
                    help="stop once T_max falls below this (middle branch)")
    ap.add_argument("--out", default=str(HERE/"data/rf_scurve"))
    args = ap.parse_args()

    tname = {"ma": "mixture-averaged", "ule": "unity-Lewis-number"}
    tag = f"{args.transport}_w{args.width:g}_s{args.slope:g}"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gas = ct.Solution(str(YAML))
    gas.transport_model = "mixture-averaged"       # base always MA (robust)
    f = ct.CounterflowDiffusionFlame(gas, width=args.width)
    f.P = P
    f.fuel_inlet.X = {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
    f.fuel_inlet.T = T_IN
    f.oxidizer_inlet.X = "O2:1.0"
    f.oxidizer_inlet.T = T_IN
    f.fuel_inlet.mdot = args.mdot0
    f.oxidizer_inlet.mdot = args.mdot0
    f.set_refine_criteria(ratio=3.0, slope=args.slope, curve=args.curve,
                          prune=0.02*args.slope/0.1)

    log(f"[{tag}] base MA solve (mdot={args.mdot0}, width={args.width}, "
        f"slope={args.slope})")
    f.solve(loglevel=0, auto=True)
    log(f"[{tag}] base: T_max={f.T.max():.0f}K npts={f.grid.size} "
        f"({time.time()-t0:.0f}s)")

    if args.transport == "ule":
        # bootstrap: switch transport on the converged MA structure and settle.
        # Pipeline recipe (05_flamelet_sweep_dualgas stage A): Cantera often
        # refuses to DECLARE steady after a transport swap even though the
        # transient has settled -> cap the transient short and multi-round
        # settle until declared or Tmax stationary (<2 K/round), accepting the
        # time-stepped solution as long as the flame burns.
        f.transport_model = tname["ule"]
        try:
            f.max_time_step_count = 200
        except Exception:
            pass
        Tprev = None
        for rnd in range(1, 6):
            declared = True
            try:
                f.solve(loglevel=0, auto=False)
            except ct.CanteraError:
                if f.T.max() < 1500.0:
                    raise
                declared = False
            Tm = float(f.T.max())
            log(f"[{tag}] ule settle round {rnd}: declared={declared} "
                f"Tmax={Tm:.0f}K")
            if declared or (Tprev is not None and abs(Tm - Tprev) < 2.0):
                break
            Tprev = Tm
        log(f"[{tag}] ule settle: T_max={f.T.max():.0f}K npts={f.grid.size} "
            f"({time.time()-t0:.0f}s)")

    # ---- enable two-point control ----
    # Control points near the peak (official spacing recipe).
    Tmin0, Tmax0 = float(f.T.min()), float(f.T.max())
    Tctrl = Tmin0 + 0.95*(Tmax0 - Tmin0)
    f.two_point_control_enabled = True
    f.set_left_control_point(Tctrl)
    f.set_right_control_point(Tctrl)
    try:
        f.max_time_step_count = 500
    except Exception:
        pass
    # Arming can fail on a not-fully-declared settle state (ule): retry with
    # extra settle rounds in between.
    for attempt in range(3):
        try:
            f.solve(loglevel=0, auto=False)   # re-converge w/ control eqs
            break
        except Exception as e:
            log(f"[{tag}] two-point arming attempt {attempt} failed "
                f"({type(e).__name__}); extra settle then retry")
            f.two_point_control_enabled = False
            try:
                f.solve(loglevel=0, auto=False)
            except Exception:
                if f.T.max() < 1500.0:
                    raise
            f.two_point_control_enabled = True
            f.set_left_control_point(Tctrl)
            f.set_right_control_point(Tctrl)
    else:
        raise SystemExit(f"[{tag}] two-point arming failed after retries")
    log(f"[{tag}] two-point control armed at Tctrl={Tctrl:.0f}K")

    hist = []          # (Tctrl, T_max, a, chi_st, mdot_f, mdot_o, npts)
    state = f.to_array()
    dT = args.dT0
    a_max = -1.0; chi_at_amax = -1.0; T_at_amax = -1.0
    past_fold = False
    nfail = 0

    SPACING = 0.95      # control points at Tmin + 0.95*(Tmax-Tmin): near-peak
    for it in range(args.nmax):
        try:
            # Re-anchor the control-point LOCATIONS to the CURRENT profile
            # every step (official continuation recipe): without this the
            # points slide down the cold flanks and Tctrl bottoms out at the
            # inlet temperature long before the fold (observed: Tctrl 3554 ->
            # 1641 K with T_max unchanged).
            Tcp = float(f.T.min()) + SPACING*(float(f.T.max()) - float(f.T.min()))
            f.set_left_control_point(Tcp)
            f.set_right_control_point(Tcp)
            f.left_control_point_temperature -= dT
            f.right_control_point_temperature -= dT
            f.solve(loglevel=0, auto=False)
        except Exception as e:
            nfail += 1
            f.from_array(state)
            dT *= 0.5
            log(f"[{tag}] step {it}: FAIL ({type(e).__name__}) -> dT={dT:.2f}")
            if dT < args.dTmin:
                log(f"[{tag}] dT floor reached -- stopping")
                break
            continue

        state = f.to_array()
        Tm = float(f.T.max())
        a = mean_strain(f, args.width)
        cst = chi_st_of(f)
        hist.append((float(f.left_control_point_temperature), Tm, a, cst,
                     float(f.fuel_inlet.mdot), float(f.oxidizer_inlet.mdot),
                     int(f.grid.size)))
        if a > a_max:
            a_max, chi_at_amax, T_at_amax = a, cst, Tm
        elif not past_fold and a < 0.985*a_max and Tm < T_at_amax:
            past_fold = True
            log(f"[{tag}] *** FOLD PASSED: a_ext={a_max:.1f}/s "
                f"chi_ext={chi_at_amax:.1f}/s T_at_fold={T_at_amax:.0f}K ***")
        if it % 5 == 0 or past_fold:
            log(f"[{tag}] step {it:3d}: Tctrl={hist[-1][0]:6.0f} "
                f"T_max={Tm:6.0f} a={a:9.2f}/s chi={cst:9.2f}/s "
                f"npts={f.grid.size} ({time.time()-t0:.0f}s)")
        # occasional refine so the middle-branch structure stays resolved
        if it % 15 == 14:
            try:
                f.solve(loglevel=0, refine_grid=True, auto=False)
                state = f.to_array()
            except Exception:
                f.from_array(state)
        # success -> try growing dT back
        dT = min(dT*1.3, args.dT0)
        if Tm < args.stopT:
            log(f"[{tag}] middle/lower branch reached (T_max={Tm:.0f}) -- done")
            break

    res = {"tag": tag, "transport": tname[args.transport],
           "P_bar": P/1e5, "T_in": T_IN, "width": args.width,
           "slope": args.slope,
           "a_ext_fold": a_max, "chi_ext_fold": chi_at_amax,
           "T_at_fold": T_at_amax, "fold_passed": bool(past_fold),
           "nfail": nfail,
           "hist_Tctrl_Tmax_a_chi_mf_mo_npts": hist}
    with open(out/f"tp_{tag}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"[{tag}] saved tp_{tag}.json  a_ext(fold)={a_max:.1f}/s "
        f"chi_ext(fold)={chi_at_amax:.1f}/s fold_passed={past_fold} "
        f"(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
