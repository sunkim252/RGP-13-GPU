"""chi-damped 0-D ignition manifold: dissipation-vs-chemistry competition.

Stage-2 of UFPV-lite. The plain 0-D fill (ufpv0d_ignition_fill.py) is the
chi->0 limit, so its source is an UPPER BOUND: in a real mixing layer the
scalar-dissipation rate chi relaxes the state back toward the frozen mixing
line while chemistry drives it up the ignition trajectory. This script adds
that competition with the classical IEM/LMSE surrogate of the flamelet
dissipation term:

    dphi/dt = S_chem(phi) - r(Z) * (phi - phi_mix(Z)),
    r(Z)    = chi_st * shape(Z) / (2 sigma^2)

sigma (the reacting-layer width in Z-space) is the ONLY free constant and is
CALIBRATED so the model's steady extinction at Z_st matches our own measured
two-point fold: chi_ext = 1.47e6 1/s (52.5 bar S-curve campaign). With that
anchor the model gives, per (Z, chi):
  * ignition boundary chi_ign(Z): does the FROZEN state auto-ignite?
  * sustaining boundary chi_ext(Z): does the BURNING state survive?
  * chi-damped ignition trajectories -> iso-c sampled omega(Z, c; chi)

Outputs: data/ufpv_chi_damped/{summary.json, fam_chi<val>/flamelet_*.npz}
(iso-c pseudo-flamelet families per chi, steady-family npz schema, each
stamped with its chi_st so the 4-D chi-axis builder or a solver-side gating
field can consume them).

Usage: python3 ufpv0d_chi_damped.py [--calibrate-only]
"""
import json, sys, time
from pathlib import Path

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from scipy.special import erfcinv

HERE = Path(__file__).resolve().parent
YAML = HERE / "data/wang2011_ideal_v32.yaml"
P = 52.5e5
T_IN = 800.0
ZST = 0.2255
PV = ("CO2", "CO", "H2O", "H2")
NZ = 41
NSNAP = 24
CHI_EXT_MEAS = 1.47e6          # our two-point fold measurement [1/s]
CHI_GRID = (1e3, 1e4, 1e5, 3e5, 1e6, 2e6)


def log(m):
    print(m, flush=True)


class Bin0D:
    """One mixture-fraction bin: chemistry + IEM relaxation to mixing state."""

    def __init__(self, gas, Z, Ymix, sigma, chi_st, shape):
        self.gas = gas
        self.ns = gas.n_species
        self.MW = gas.molecular_weights
        self.Ymix = Ymix
        self.r = chi_st*shape/(2.0*sigma*sigma)
        self.pv_idx = [gas.species_index(s) for s in PV
                       if s in gas.species_names]

    def rhs(self, t, u):
        if not np.isfinite(u).all():
            raise FloatingPointError("non-finite state")
        T = min(max(u[0], 300.0), 4500.0)
        Y = np.clip(u[1:], 0.0, 1.0)
        g = self.gas
        g.TPY = T, P, Y
        wdot = g.net_production_rates
        rho = g.density
        cp = g.cp_mass
        hmol = g.partial_molar_enthalpies
        dT = -float(hmol @ wdot)/(rho*cp) - self.r*(T - T_IN)
        dY = wdot*self.MW/rho - self.r*(Y - self.Ymix)
        return np.concatenate([[dT], dY])

    def run(self, T0, Y0, tend, nrec=400):
        """Integrate; record trajectory (T, C, omega) at solver steps."""
        u0 = np.concatenate([[T0], Y0])
        try:
            sol = solve_ivp(self.rhs, (0.0, tend), u0, method="BDF",
                            rtol=1e-7, atol=1e-12, dense_output=False,
                            max_step=tend/10)
        except FloatingPointError:
            return None
        if not sol.success and sol.t[-1] < 0.5*tend:
            return None
        ts = sol.t
        # subsample if huge
        if len(ts) > 4000:
            keep = np.unique(np.linspace(0, len(ts)-1, 4000).astype(int))
            ys = sol.y[:, keep]; ts = ts[keep]
        else:
            ys = sol.y
        return ts, ys


def build_bins(sigma, chi_st):
    gas = ct.Solution(str(YAML))
    gas.TPX = T_IN, P, {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
    Yf = gas.Y.copy()
    gas.TPX = T_IN, P, "O2:1.0"
    Yo = gas.Y.copy()
    Zg = np.linspace(0.0, 1.0, NZ)
    Zc = np.clip(Zg, 1e-4, 1-1e-4)
    shape = np.exp(-2*erfcinv(2*Zc)**2)/np.exp(-2*erfcinv(2*ZST)**2)
    bins = []
    for j, Z in enumerate(Zg):
        g = ct.Solution(str(YAML))
        Ymix = Z*Yf + (1-Z)*Yo
        bins.append((Z, Ymix, Bin0D(g, Z, Ymix, sigma, chi_st, shape[j])))
    return Zg, bins


def burning_survives(sigma, chi_st):
    """Zst bin from HP-equilibrium: does it stay lit? (for calibration)"""
    Zg, bins = build_bins(sigma, chi_st)
    j = int(np.argmin(abs(Zg - ZST)))
    Z, Ymix, b = bins[j]
    g = ct.Solution(str(YAML))
    g.TPY = T_IN, P, Ymix
    g.equilibrate("HP")
    r = b.r
    tend = 40.0/max(r, 1e-3)
    out = b.run(g.T, g.Y, tend)
    if out is None:
        return False
    ts, ys = out
    return ys[0, -1] > 0.5*(g.T + T_IN)


def calibrate_sigma():
    """Bisect sigma so the burning Zst state extinguishes exactly at
    CHI_EXT_MEAS (log bisection on sigma)."""
    lo, hi = 0.02, 1.0        # sigma in Z units
    # smaller sigma => stronger relaxation => quenches easier
    assert not burning_survives(lo, CHI_EXT_MEAS), "lo too large?"
    assert burning_survives(hi, CHI_EXT_MEAS), "hi too small?"
    for it in range(14):
        mid = np.sqrt(lo*hi)
        ok = burning_survives(mid, CHI_EXT_MEAS)
        log(f"[cal] sigma={mid:.4f}: burning@chi_ext "
            f"{'SURVIVES' if ok else 'quenches'}")
        if ok:
            hi = mid
        else:
            lo = mid
    sigma = np.sqrt(lo*hi)
    log(f"[cal] sigma* = {sigma:.4f}  (r(Zst)@chi_ext = "
        f"{CHI_EXT_MEAS/(2*sigma**2):.3g} 1/s)")
    return sigma


def main():
    out = HERE/"data/ufpv_chi_damped"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sigma = calibrate_sigma()
    if "--calibrate-only" in sys.argv:
        return

    gas0 = ct.Solution(str(YAML))
    summary = {"sigma": float(sigma), "chi_ext_meas": CHI_EXT_MEAS,
               "chi": {}}
    clevels = np.linspace(0.05, 0.95, NSNAP)

    for chi in CHI_GRID:
        Zg, bins = build_bins(sigma, chi)
        prof_T = np.full((NSNAP, NZ), T_IN)
        prof_Y = np.zeros((NSNAP, NZ, gas0.n_species))
        prof_om = np.zeros((NSNAP, NZ))
        ign = np.zeros(NZ, dtype=bool)
        pv_idx = bins[0][2].pv_idx
        MW = gas0.molecular_weights
        for j, (Z, Ymix, b) in enumerate(bins):
            prof_Y[:, j, :] = Ymix          # default: frozen (not ignited)
            r = max(b.r, 1e-3)
            tend = min(max(60.0/r, 0.05), 2.0)
            res = b.run(T_IN, Ymix, tend)
            if res is None:
                continue
            ts, ys = res
            T = ys[0]; Y = ys[1:].T
            C = Y[:, pv_idx].sum(1)
            if T.max() < T_IN + 100 or C[-1] < 1e-6:
                continue                    # never ignites at this chi
            ign[j] = True
            cn = np.maximum.accumulate(C/C[-1])
            for k, cl in enumerate(clevels):
                i = min(int(np.searchsorted(cn, cl)), len(ts)-1)
                g = b.gas
                g.TPY = max(T[i], 300.0), P, np.clip(Y[i], 0, None)
                w = g.net_production_rates
                prof_T[k, j] = T[i]
                prof_Y[k, j, :] = Y[i]
                prof_om[k, j] = float((w[pv_idx]*MW[pv_idx]).sum()
                                      / max(g.density, 1e-30))
        # save iso-c family for this chi
        fam = out/f"fam_chi{chi:g}"
        fam.mkdir(exist_ok=True)
        for k, cl in enumerate(clevels):
            arr = ct.SolutionArray(gas0, NZ)
            arr.TPY = (np.maximum(prof_T[k], 300.0), P,
                       np.clip(prof_Y[k], 0, None))
            a = {"z": Zg.copy(), "Z": Zg.copy(), "T": prof_T[k],
                 "rho": arr.density, "lam": arr.thermal_conductivity,
                 "mu": arr.viscosity, "cp": arr.cp_mass,
                 "alpha": arr.thermal_conductivity /
                          np.maximum(arr.density*arr.cp_mass, 1e-30),
                 "chi": chi*np.ones(NZ), "chi_st": np.asarray(float(chi)),
                 "C": prof_Y[k][:, pv_idx].sum(1), "omega_C": prof_om[k],
                 "mdot": np.asarray(0.0), "P": np.asarray(P),
                 "T_fuel": np.asarray(T_IN), "T_ox": np.asarray(T_IN),
                 "Z_st_ref": np.asarray(ZST), "npts": np.asarray(NZ),
                 "Tmax": np.asarray(float(prof_T[k].max())),
                 "struct_transport": np.asarray("mixture-averaged"),
                 "kind": np.asarray("0D-ignition-isoC-chiDamped")}
            for m2, sp in enumerate(gas0.species_names):
                a[f"Y_{sp}"] = prof_Y[k][:, m2]
            np.savez_compressed(fam/f"flamelet_{k:03d}.npz", **a)
        om_mid = prof_om[np.argmin(abs(clevels-0.6)),
                         int(np.argmin(abs(Zg-ZST)))]
        summary["chi"][str(chi)] = {
            "n_ignited_bins": int(ign.sum()),
            "Z_ignited": [float(Zg[j]) for j in range(NZ) if ign[j]],
            "omega_midc_Zst": float(om_mid)}
        log(f"[chi={chi:g}] ignited bins {ign.sum()}/{NZ}, "
            f"omega(c=0.6,Zst)={om_mid:.3g} ({time.time()-t0:.0f}s)")

    with open(out/"summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
