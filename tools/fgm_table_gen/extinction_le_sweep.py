"""Extinction-strain quantification: unity-Lewis vs mixture-averaged transport
at 52.5 bar (wang2011 kerosene surrogate / O2, T_in = 800 K both streams).

Uses Cantera's ideal-gas CounterflowDiffusionFlame -- the SAME structural
approximation as the production dual-gas FGM table -- so the measured
extinction gap isolates the TRANSPORT MODEL (Le) effect that the unity-Le
table bakes into the manifold. (The real-fluid structure correction is the
separate flamelet-equation track; Wang-2015-style analysis says real-fluid
effects concentrate in the cold inlet, so the Le effect dominates the
extinction shift at these T_in.)

Strategy per transport mode:
  1. solve base flame at moderate strain (mdot0), full auto refinement
  2. strain continuation: mdot *= fac, re-solve auto=False on the same grid
     (loose refine every few steps); extinction detected when T_max drops
     below burn threshold or the solver fails after retries
  3. bisection refine between last-burning and first-extinguished mdot
  4. record global strain rate a = (2*|u_f| + |u_o|... ) -- we report the
     standard mean axial strain a = (u_f + u_o)/L and chi_st from the
     converged profiles for direct comparison with the FGM chi axis.

Outputs: data/rf_scurve/ext_<mode>.json + per-branch profiles npz.
Run:  python3 extinction_le_sweep.py --transport unity-Lewis-number
      python3 extinction_le_sweep.py --transport mixture-averaged
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
YAML = HERE / "data/wang2011_ideal_v32.yaml"   # ideal-gas variant (structure)
P = 52.5e5
T_IN = 800.0
WIDTH = 0.01        # m, domain width (counterflow gap)
BURN_T = 1500.0
ZST = 0.2255


def log(m):
    print(m, flush=True)


def mean_strain(f):
    """Global strain rate a = (u_fuel + u_ox)/width [1/s]."""
    uf = f.fuel_inlet.mdot/f.fuel_inlet.T   # placeholder; use velocities
    # velocities from mdot/rho at the inlets:
    gas = f.gas
    gas.TPX = f.fuel_inlet.T, f.P, f.fuel_inlet.X
    rho_f = gas.density
    gas.TPX = f.oxidizer_inlet.T, f.P, f.oxidizer_inlet.X
    rho_o = gas.density
    return (f.fuel_inlet.mdot/rho_f + f.oxidizer_inlet.mdot/rho_o)/WIDTH


def chi_st_of(f, Zst=ZST):
    """chi_st = 2 D_th |dZ/dy|^2 at Z=Zst from the converged profiles,
    with Bilger-like Z from the local mixture fraction of Cantera."""
    Z = f.mixture_fraction("Bilger")
    y = f.grid
    lam = f.thermal_conductivity
    rho = f.density
    cp = f.cp_mass
    D = lam/np.maximum(rho*cp, 1e-30)
    dZdy = np.gradient(Z, y)
    chi = 2.0*D*dZdy**2
    # value at Z closest to Zst
    i = int(np.argmin(np.abs(Z - Zst)))
    return float(chi[i]), float(np.max(chi))


def make_flame(transport):
    gas = ct.Solution(str(YAML))
    gas.transport_model = transport
    f = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
    f.P = P
    f.fuel_inlet.X = {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
    f.fuel_inlet.T = T_IN
    f.oxidizer_inlet.X = "O2:1.0"
    f.oxidizer_inlet.T = T_IN
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", required=True,
                    choices=("unity-Lewis-number", "mixture-averaged",
                             "multicomponent"))
    ap.add_argument("--mdot0", type=float, default=0.5)   # kg/m2/s
    ap.add_argument("--fac", type=float, default=1.6)
    ap.add_argument("--nbisect", type=int, default=6)
    ap.add_argument("--out", default=str(HERE/"data/rf_scurve"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = {"unity-Lewis-number": "uleCF", "mixture-averaged": "maCF",
           "multicomponent": "mcCF"}[args.transport]

    f = make_flame(args.transport)
    # momentum-balanced inlets: mdot_o = mdot_f (equal), Cantera balances
    f.fuel_inlet.mdot = args.mdot0
    f.oxidizer_inlet.mdot = args.mdot0
    f.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.2, prune=0.03)
    t0 = time.time()
    log(f"[{tag}] base solve at mdot={args.mdot0} kg/m2/s "
        f"(P={P/1e5} bar, T_in={T_IN} K, {args.transport})")
    # base-solve ladder: full auto at mdot0, then progressively gentler
    # (lower strain converges easier from the auto ignition guess).
    base_ok = False
    for m0 in (args.mdot0, args.mdot0/2.5, args.mdot0/6.25):
        f.fuel_inlet.mdot = m0
        f.oxidizer_inlet.mdot = m0
        try:
            f.solve(loglevel=0, auto=True)
            if float(f.T.max()) > BURN_T:
                base_ok = True
                args.mdot0 = m0
                break
        except Exception as e:
            log(f"[{tag}] base fail at mdot={m0:.3g} "
                f"({type(e).__name__}) -- retry gentler")
    if not base_ok:
        raise SystemExit(f"[{tag}] base flame failed at all mdot0 ladder")
    Tmax = float(f.T.max())
    a = mean_strain(f); cst, cmax = chi_st_of(f)
    log(f"[{tag}] base: T_max={Tmax:.0f}K a={a:.1f}/s chi_st={cst:.1f}/s "
        f"({time.time()-t0:.0f}s, {f.grid.size} pts)")
    if Tmax < BURN_T:
        raise SystemExit(f"[{tag}] base flame not burning -- lower mdot0")

    hist = [(args.mdot0, Tmax, a, cst)]
    burn_state = f.to_array()      # snapshot of last burning solution
    mdot_burn = args.mdot0
    mdot_lo, mdot_hi = None, None

    mdot = args.mdot0
    nref = 0
    while True:
        mdot *= args.fac
        f.fuel_inlet.mdot = mdot
        f.oxidizer_inlet.mdot = mdot
        try:
            nref += 1
            f.solve(loglevel=0, auto=(nref % 4 == 0))
            Tmax = float(f.T.max())
        except Exception as e:
            log(f"[{tag}] solver fail at mdot={mdot:.3g}: {type(e).__name__}")
            Tmax = 0.0
        if Tmax > BURN_T:
            a = mean_strain(f); cst, cmax = chi_st_of(f)
            hist.append((mdot, Tmax, a, cst))
            log(f"[{tag}] mdot={mdot:8.3g} BURN T={Tmax:6.0f} a={a:9.1f}/s "
                f"chi_st={cst:9.1f}/s")
            burn_state = f.to_array(); mdot_burn = mdot
        else:
            log(f"[{tag}] mdot={mdot:8.3g} EXTINGUISHED (T={Tmax:.0f})")
            mdot_lo, mdot_hi = mdot_burn, mdot
            break
        if mdot > 1e5:
            log(f"[{tag}] no extinction below mdot=1e5 -- stopping")
            break

    if mdot_lo is not None:
        for k in range(args.nbisect):
            mid = np.sqrt(mdot_lo*mdot_hi)
            f.from_array(burn_state)
            f.fuel_inlet.mdot = mid
            f.oxidizer_inlet.mdot = mid
            try:
                f.solve(loglevel=0, auto=False)
                Tmax = float(f.T.max())
            except Exception:
                Tmax = 0.0
            if Tmax > BURN_T:
                a = mean_strain(f); cst, cmax = chi_st_of(f)
                hist.append((mid, Tmax, a, cst))
                burn_state = f.to_array(); mdot_lo = mid
                log(f"[{tag}] bisect{k}: mdot={mid:.4g} BURN "
                    f"a={a:.1f} chi_st={cst:.1f}")
            else:
                mdot_hi = mid
                log(f"[{tag}] bisect{k}: mdot={mid:.4g} ext")

        # extinction point = last burning state
        f.from_array(burn_state)
        a_ext = mean_strain(f); cst_ext, cmax_ext = chi_st_of(f)
        log(f"[{tag}] EXTINCTION: mdot in [{mdot_lo:.4g}, {mdot_hi:.4g}] "
            f"-> a_ext={a_ext:.1f}/s chi_st_ext={cst_ext:.1f}/s "
            f"T_max(last burn)={float(f.T.max()):.0f}K")
        np.savez(out/f"extflame_{tag}.npz",
                 grid=f.grid, T=f.T, Z=f.mixture_fraction("Bilger"),
                 mdot=mdot_lo, P=P)
    else:
        a_ext = cst_ext = None

    with open(out/f"ext_{tag}.json", "w") as fh:
        json.dump({"transport": args.transport, "P_bar": P/1e5,
                   "T_in": T_IN, "width": WIDTH,
                   "hist_mdot_T_a_chi": hist,
                   "mdot_ext_lo": mdot_lo, "mdot_ext_hi": mdot_hi,
                   "a_ext": a_ext, "chi_st_ext": cst_ext}, fh, indent=1)
    log(f"[{tag}] saved ext_{tag}.json  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
