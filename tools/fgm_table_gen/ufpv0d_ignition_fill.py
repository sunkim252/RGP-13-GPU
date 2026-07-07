"""UFPV-lite mid-c fill via 0-D constant-pressure autoignition (robust path).

The custom unsteady flamelet-equation integrator rams its T-clip during the
kernel runaway (rhs -> 1e29, sum(Y) drift) and NaNs BDF. This generator
avoids custom numerics entirely: at each mixture fraction Z_j, integrate a
homogeneous constant-pressure reactor (Cantera ReactorNet / CVODES) from the
FROZEN MIXING state at 800 K, 52.5 bar. That is the chi->0 limit of the
unsteady flamelet equation, and its trajectory sweeps c from 0 to the local
equilibrium -- exactly the mid-c ignition states the steady MA family lacks
(the 1-D A/B quench root cause).

A "pseudo-flamelet" = the SNAPSHOT ACROSS ALL Z at one shared time t: profile
{T(Z_j;t), Y(Z_j;t), omega_C(Z_j;t)} saved in the steady-family npz schema so
04_build_fgm_table_4d.py consumes it unchanged. Rich/lean bins ignite at
different delays, so the snapshot family covers the (Z, c) plane densely.

Ignition aid: 800 K autoignition of the kerosene surrogate can be slow at
some Z; each reactor gets a small temperature offset ramp option -- default
uses plain 800 K (52.5 bar => delays are short) with a long-enough horizon.

Usage: python3 ufpv0d_ignition_fill.py [idx0=300] [outdir]
"""
import sys, time
from pathlib import Path

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
import os as _os
YAML = HERE / _os.environ.get("FGM_IGNITION_MECH", "data/wang2011_ideal_v32.yaml")
P = 52.5e5
T_IN = 800.0
ZST = 0.2255
PV = ("CO2", "CO", "H2O", "H2")
NZ = 41                    # Z bins (matches flamelet N; builder regrids anyway)
NSNAP = 30                 # snapshot times (log-spaced over ignition window)


def log(m):
    print(m, flush=True)


def main():
    idx0 = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = Path(sys.argv[2] if len(sys.argv) > 2
               else HERE/"data/flamelets_dualgas_MA_uf")
    out.mkdir(parents=True, exist_ok=True)

    gas = ct.Solution(str(YAML))
    ns = gas.n_species
    MW = gas.molecular_weights
    gas.TPX = T_IN, P, {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
    Yf = gas.Y.copy()
    gas.TPX = T_IN, P, "O2:1.0"
    Yo = gas.Y.copy()
    pv_idx = [gas.species_index(s) for s in PV if s in gas.species_names]

    Zg = np.linspace(0.0, 1.0, NZ)
    # target NORMALISED progress levels (vs each bin's own final C).
    # Capped at 0.75: the fill's purpose is the mid-c source support; above
    # that the steady flamelet family owns the region, and letting the 0-D
    # ADIABATIC end-states coexist there with the STRAINED steady states
    # (same (Z,c), up to ~900 K apart at rich Z) made griddata blend two
    # physical branches -> non-monotone T(c) (-368 K dips) and jagged
    # contours at Z~0.4-0.5, c>0.85.
    clevels = np.linspace(0.05, 0.75, NSNAP)

    t0w = time.time()
    # ---- per-bin trajectory, sampled in ITS OWN progress space ----
    # (thermal runaway is ~us while delays are ~0.05-0.3 s, so shared-time
    # snapshots can never catch mid-c states; polling CVODES internal steps
    # resolves the runaway in C by construction.)
    prof_T = np.full((NSNAP, NZ), T_IN)
    prof_Y = np.zeros((NSNAP, NZ, ns))
    prof_om = np.zeros((NSNAP, NZ))
    for j, Z in enumerate(Zg):
        g = ct.Solution(str(YAML))
        g.TPY = T_IN, P, Z*Yf + (1 - Z)*Yo
        Y0 = g.Y.copy()
        r = ct.IdealGasConstPressureReactor(g)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = 1e-9, 1e-15
        Cs=[]; Ts=[]; Ys=[]; oms=[]
        tend = 1.0
        while net.time < tend:
            try:
                net.step()
            except Exception:
                break
            ph = r.thermo
            C = float(ph.Y[pv_idx].sum())
            w = ph.net_production_rates
            Cs.append(C); Ts.append(r.T); Ys.append(ph.Y.copy())
            oms.append(float((w[pv_idx]*MW[pv_idx]).sum()
                             / max(ph.density, 1e-30)))
            # stop once chemistry is done (near its own equilibrium)
            if len(Cs) > 10 and abs(oms[-1]) < 1e-3 and r.T > T_IN + 200:
                break
            if len(Cs) > 20000:
                break
        Cs = np.array(Cs); Ts = np.array(Ts)
        oms = np.array(oms); Ys = np.array(Ys)
        Cfin = Cs[-1] if len(Cs) else 0.0
        if Cfin < 1e-6 or Ts.max() < T_IN + 100:
            # bin never ignited within horizon: leave frozen mixing state
            for k in range(NSNAP):
                prof_Y[k, j, :] = Y0
            continue
        # sample this bin at its own normalised progress levels
        cn = Cs/Cfin
        # enforce monotonicity for interp (runaway is monotone in C)
        mono = np.maximum.accumulate(cn)
        for k, cl in enumerate(clevels):
            i = int(np.searchsorted(mono, cl))
            i = min(max(i, 0), len(Cs) - 1)
            prof_T[k, j] = Ts[i]
            prof_Y[k, j, :] = Ys[i]
            prof_om[k, j] = oms[i]
        if j % 10 == 0:
            log(f"[0Dign] bin {j}/{NZ} Z={Z:.3f}: steps={len(Cs)} "
                f"Cfin={Cfin:.3f} Tfin={Ts[-1]:.0f} "
                f"om_peak={oms.max():.3g} ({time.time()-t0w:.0f}s)")

    # ---- assemble iso-progress pseudo-flamelets ----
    idx = idx0
    for k, cl in enumerate(clevels):
        T = prof_T[k]; Y = prof_Y[k]; om = prof_om[k]
        arr = ct.SolutionArray(gas, NZ)
        arr.TPY = (np.maximum(T, 300.0), P, np.clip(Y, 0, None))
        rho = arr.density; lam = arr.thermal_conductivity
        mu = arr.viscosity; cp = arr.cp_mass
        alpha = lam/np.maximum(rho*cp, 1e-30)
        a = {"z": Zg.copy(), "Z": Zg.copy(), "T": T,
             "rho": rho, "lam": lam, "mu": mu, "cp": cp, "alpha": alpha,
             "chi": np.zeros(NZ), "chi_st": np.asarray(0.0),
             "C": Y[:, pv_idx].sum(1), "omega_C": om,
             "mdot": np.asarray(0.0), "P": np.asarray(P),
             "T_fuel": np.asarray(T_IN), "T_ox": np.asarray(T_IN),
             "Z_st_ref": np.asarray(ZST), "npts": np.asarray(NZ),
             "Tmax": np.asarray(float(T.max())),
             "struct_transport": np.asarray("mixture-averaged"),
             "kind": np.asarray("0D-ignition-isoC")}
        for m2, sp in enumerate(gas.species_names):
            a[f"Y_{sp}"] = Y[:, m2]
        np.savez_compressed(out/f"flamelet_{idx:03d}.npz", **a)
        log(f"[0Dign] iso-c={cl:.2f}: T_max={T.max():6.0f}K "
            f"om_max={om.max():.3g} -> flamelet_{idx:03d}.npz")
        idx += 1
    log(f"[0Dign] done: {NSNAP} iso-c pseudo-flamelets in "
        f"{time.time()-t0w:.0f}s")


if __name__ == "__main__":
    main()
