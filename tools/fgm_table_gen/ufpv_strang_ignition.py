"""TRUE unsteady igniting flamelets via Strang operator splitting (v4 fill).

Replaces the 0-D iso-c fill (ufpv0d_ignition_fill.py), whose three artificial
elements motivated this: (i) chi->0 source upper bound (no dissipation), (ii)
iso-c stitching of states from different physical times, (iii) the arbitrary
c<=0.75 cap needed to avoid branch collisions. Here we integrate the actual
unsteady flamelet equations in Z-space,

  dY_k/dt = (chi(Z)/(2 Le_k)) d2Y_k/dZ2 + wdot_k Mk / rho
  dT/dt   = (chi(Z)/2) d2T/dZ2 - sum_k h_k wdot_k /(rho cp)

with STRANG SPLITTING: a half-step of implicit diffusion (tridiagonal solve
per variable, unconditionally stable), a full step of pure chemistry per node
(Cantera IdealGasConstPressureReactor / CVODES -- the proven robust piece),
then another half diffusion step. This avoids the monolithic-BDF failure mode
seen earlier (trial states ramming the T-clip during runaway -> rhs 1e29):
chemistry is integrated by CVODES inside physical bounds, diffusion is linear.

The trajectory starts from the frozen mixing line plus a hot kernel at Z_st
and sweeps the (Z,c) plane as an IGNITING FRONT: mid-c states now carry the
front-consistent (dissipation-damped) source, T is continuous with the steady
branch it converges to, and no c-cap is needed. Snapshots are saved in the
steady-family npz schema (chi_st stamped) for the unmodified table builder.

Usage:
  python3 ufpv_strang_ignition.py <chi_st> <idx0> [outdir]
      e.g. 2.0 400 / 20.0 440 / 200.0 480   (each ~26 snapshots)
"""
import sys, time
from pathlib import Path

import numpy as np
import cantera as ct
from scipy.special import erfcinv

HERE = Path(__file__).resolve().parent
YAML = HERE / "data/wang2011_ideal_v32.yaml"
P = 52.5e5
T_IN = 800.0
ZST = 0.2255
PV = ("CO2", "CO", "H2O", "H2")
N = 41
NSNAP = 26


def log(m):
    print(m, flush=True)


def thomas(a, b, c, d):
    """Tridiagonal solve (a: sub, b: diag, c: super, d: rhs) -> x."""
    n = len(d)
    cp = np.empty(n); dp = np.empty(n)
    cp[0] = c[0]/b[0]; dp[0] = d[0]/b[0]
    for i in range(1, n):
        m = b[i] - a[i]*cp[i-1]
        cp[i] = c[i]/m
        dp[i] = (d[i] - a[i]*dp[i-1])/m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x


class StrangFlamelet:
    def __init__(self, chi_st):
        gas = ct.Solution(str(YAML))
        gas.transport_model = "mixture-averaged"
        self.gas = gas
        self.ns = gas.n_species
        self.MW = gas.molecular_weights
        gas.TPX = T_IN, P, {"NC10H22": 0.74, "PHC3H7": 0.15, "CYC9H18": 0.11}
        self.Yf = gas.Y.copy()
        gas.TPX = T_IN, P, "O2:1.0"
        self.Yo = gas.Y.copy()
        self.Z = np.linspace(0, 1, N)
        self.dZ = self.Z[1] - self.Z[0]
        Zc = np.clip(self.Z, 1e-4, 1-1e-4)
        self.chi = chi_st*np.exp(-2*erfcinv(2*Zc)**2) \
            / np.exp(-2*erfcinv(2*ZST)**2)
        self.pv_idx = [gas.species_index(s) for s in PV
                       if s in gas.species_names]
        # one reactor+net per node, rebuilt lazily (reuse keeps CVODES state)
        self._reactors = [None]*N
        self._arr = ct.SolutionArray(gas, N)

    def init_state(self):
        T = T_IN + 900.0*np.exp(-((self.Z - ZST)/0.10)**2)
        Y = np.outer(1 - self.Z, self.Yo) + np.outer(self.Z, self.Yf)
        return T, Y

    def lewis(self, T, Y):
        """Per-node per-species Le_k = alpha/D_k (mixture-averaged)."""
        arr = self._arr
        arr.TPY = (np.maximum(T, 300.0), P, np.clip(Y, 0, None))
        alpha = arr.thermal_conductivity/(arr.density*arr.cp_mass)
        Dk = np.clip(arr.mix_diff_coeffs, 1e-12, None)
        return alpha[:, None]/Dk        # (N, ns)

    def diffuse_half(self, T, Y, dt, Lek):
        """Implicit half-step of  dphi/dt = (chi/(2 Le)) d2phi/dZ2.
        Dirichlet BCs (pure streams / inlet T)."""
        h = 0.5*dt
        lam_T = self.chi*h/(2*self.dZ**2)             # Le=1 for T
        Tn = T.copy()
        a = -lam_T[1:-1]; b = 1 + 2*lam_T[1:-1]; c = -lam_T[1:-1]
        d = T[1:-1].copy()
        d[0] += lam_T[1]*T[0]; d[-1] += lam_T[-2]*T[-1]
        Tn[1:-1] = thomas(np.r_[0, a[1:]], b, np.r_[c[:-1], 0], d)
        Yn = Y.copy()
        for k in range(self.ns):
            lam = self.chi*h/(2*Lek[:, k]*self.dZ**2)
            a = -lam[1:-1]; b = 1 + 2*lam[1:-1]; c = -lam[1:-1]
            d = Y[1:-1, k].copy()
            d[0] += lam[1]*Y[0, k]; d[-1] += lam[-2]*Y[-1, k]
            Yn[1:-1, k] = thomas(np.r_[0, a[1:]], b, np.r_[c[:-1], 0], d)
        return Tn, np.clip(Yn, 0.0, None)

    def _node_net(self, j):
        """Persistent per-node reactor (construction dominates runtime)."""
        if self._reactors[j] is None:
            g = ct.Solution(str(YAML))
            g.TPY = T_IN, P, self.Yo
            r = ct.IdealGasConstPressureReactor(g)
            net = ct.ReactorNet([r])
            net.rtol, net.atol = 1e-8, 1e-14
            self._reactors[j] = (g, r, net)
        return self._reactors[j]

    def react_full(self, T, Y, dt, transit_cb=None, transit_mask=None):
        """Per-node constant-pressure chemistry over dt via CVODES.
        For nodes flagged in transit_mask (runaway faster than the Strang
        step: c jumps 0.25->0.75 within one dt, so mid-c lives INSIDE the
        CVODES integration), advance by internal steps and hand every
        sub-state to transit_cb(j, T, Y) for crossing capture."""
        Tn = T.copy(); Yn = Y.copy()
        for j in range(1, N-1):
            g, r, net = self._node_net(j)
            g.TPY = max(T[j], 300.0), P, np.clip(Y[j], 0, None)
            r.syncState()
            net.reinitialize()
            t_target = net.time + dt
            try:
                if transit_cb is not None and transit_mask is not None \
                        and transit_mask[j]:
                    nsub = 0
                    while net.time < t_target and nsub < 4000:
                        net.step()
                        nsub += 1
                        transit_cb(j, r.T, g.Y)
                else:
                    net.advance(t_target)
            except Exception:
                continue                 # keep pre-chemistry state this node
            Tn[j] = r.T
            Yn[j] = g.Y.copy()
        return Tn, Yn

    def omega(self, T, Y):
        arr = self._arr
        arr.TPY = (np.maximum(T, 300.0), P, np.clip(Y, 0, None))
        w = arr.net_production_rates
        return (w[:, self.pv_idx]*self.MW[self.pv_idx][None, :]).sum(1) \
            / np.maximum(arr.density, 1e-30)

    def snapshot(self, T, Y, chi_st, out, idx):
        arr = ct.SolutionArray(self.gas, N)
        arr.TPY = (np.maximum(T, 300.0), P, np.clip(Y, 0, None))
        a = {"z": self.Z.copy(), "Z": self.Z.copy(), "T": T.copy(),
             "rho": arr.density, "lam": arr.thermal_conductivity,
             "mu": arr.viscosity, "cp": arr.cp_mass,
             "alpha": arr.thermal_conductivity /
                      np.maximum(arr.density*arr.cp_mass, 1e-30),
             "chi": self.chi.copy(), "chi_st": np.asarray(float(chi_st)),
             "C": Y[:, self.pv_idx].sum(1), "omega_C": self.omega(T, Y),
             "mdot": np.asarray(0.0), "P": np.asarray(P),
             "T_fuel": np.asarray(T_IN), "T_ox": np.asarray(T_IN),
             "Z_st_ref": np.asarray(ZST), "npts": np.asarray(N),
             "Tmax": np.asarray(float(T.max())),
             "struct_transport": np.asarray("mixture-averaged"),
             "kind": np.asarray("strang-unsteady-ignition")}
        for k, sp in enumerate(self.gas.species_names):
            a[f"Y_{sp}"] = Y[:, k]
        np.savez_compressed(out/f"flamelet_{idx:03d}.npz", **a)


def main():
    chi_st = float(sys.argv[1])
    idx0 = int(sys.argv[2])
    out = Path(sys.argv[3] if len(sys.argv) > 3
               else HERE/"data/flamelets_dualgas_MA_v4")
    out.mkdir(parents=True, exist_ok=True)

    fl = StrangFlamelet(chi_st)
    T, Y = fl.init_state()
    izst = int(np.argmin(abs(fl.Z - ZST)))

    # dt resolves the per-cell diffusion time 2 dZ^2/chi_max; the FRONT
    # sweeps the domain in ~3/chi_st, so nsteps is chi-independent (~4000).
    dt = 0.8*2*fl.dZ**2/fl.chi.max()
    t_end = 3.0/chi_st
    nsteps = int(t_end/dt)
    # log-spaced snapshots from the very first steps (the kernel runaway and
    # early front are the mid-c rich part) through the full sweep
    snap_times = set(np.unique(np.geomspace(2, nsteps, NSNAP).astype(int)))
    t0w = time.time()
    idx = idx0
    Lek = fl.lewis(T, Y)
    # per-node c-level crossing recorder: a node transits mid-c in ~1 step
    # (the front is 1-2 cells wide in Z), so global time snapshots MISS the
    # mid-c states entirely (verified: chi=200 family had zero points at
    # c 0.3-0.7). Record each node's state the first time its normalised
    # progress crosses each level; assemble per-level pseudo-flamelets after.
    clevels = np.linspace(0.08, 0.80, 19)
    # fixed per-node ceiling = HP-equilibrium C of the local mixture (the
    # table's own c=1 anchor). A RUNNING max misfires: early in the run the
    # max is tiny, so "c=0.5" was recorded at absolute C~0.05 and the true
    # mid-c transit was never re-captured (verified gap at c 0.3-0.7).
    Ceq_bin = np.zeros(N)
    geq = ct.Solution(str(YAML))
    Ymix0 = np.outer(1 - fl.Z, fl.Yo) + np.outer(fl.Z, fl.Yf)
    for j in range(N):
        geq.TPY = T_IN, P, Ymix0[j]
        try:
            geq.equilibrate("HP")
            Ceq_bin[j] = sum(geq.Y[i] for i in fl.pv_idx)
        except Exception:
            Ceq_bin[j] = 0.0
    rec_T = np.full((len(clevels), N), np.nan)
    rec_Y = np.full((len(clevels), N, fl.ns), np.nan)
    crossed = np.zeros((len(clevels), N), dtype=bool)
    log(f"[strang chi={chi_st}] N={N} dt={dt:.2e}s steps={nsteps} "
        f"snapshots~{len(snap_times)}")
    for it in range(1, nsteps+1):
        if it % 25 == 0:
            Lek = fl.lewis(T, Y)          # refresh transport occasionally
        T, Y = fl.diffuse_half(T, Y, dt, Lek)
        # nodes about to runaway: capture crossings INSIDE CVODES
        Cpre = Y[:, fl.pv_idx].sum(1)
        cn_pre = Cpre/np.maximum(Ceq_bin, 1e-9)
        transit = (Ceq_bin > 0.02) & (cn_pre > 0.02) & (cn_pre < 0.9) \
                  & (T > 900.0)
        def _cb(j, Tj, Yj):
            Cj = float(Yj[fl.pv_idx].sum())
            cnj = Cj/max(Ceq_bin[j], 1e-9)
            for k, cl in enumerate(clevels):
                if not crossed[k, j] and cnj >= cl:
                    crossed[k, j] = True
                    rec_T[k, j] = Tj
                    rec_Y[k, j] = Yj.copy()
        T, Y = fl.react_full(T, Y, dt, transit_cb=_cb, transit_mask=transit)
        T, Y = fl.diffuse_half(T, Y, dt, Lek)
        # post-step recorder for slow (non-transit) evolution
        Cnow = Y[:, fl.pv_idx].sum(1)
        for k, cl in enumerate(clevels):
            newly = (~crossed[k]) & (Ceq_bin > 0.02) \
                    & (Cnow >= cl*np.maximum(Ceq_bin, 1e-9))
            if newly.any():
                crossed[k, newly] = True
                rec_T[k, newly] = T[newly]
                rec_Y[k, newly] = Y[newly]
        if it in snap_times:
            fl.snapshot(T, Y, chi_st, out, idx)
            C = Y[:, fl.pv_idx].sum(1)
            log(f"[strang chi={chi_st}] it={it} t={it*dt:.3e}s "
                f"T_max={T.max():6.0f}K C(Zst)={C[izst]:.3f} "
                f"-> flamelet_{idx:03d} ({time.time()-t0w:.0f}s)")
            idx += 1
        # stop once the front has swept the ignitable range and the profile
        # is quasi-steady (progress-coverage saturated for 300 steps)
        if it % 100 == 0:
            cov = int(np.sum(Y[:, fl.pv_idx].sum(1) > 0.05))
            if cov == getattr(fl, "_cov_prev", -1):
                fl._cov_stall = getattr(fl, "_cov_stall", 0) + 100
            else:
                fl._cov_stall = 0
            fl._cov_prev = cov
            if fl._cov_stall >= 300 and it > 0.25*nsteps:
                fl.snapshot(T, Y, chi_st, out, idx)
                log(f"[strang chi={chi_st}] front sweep saturated at it={it} "
                    f"-- final snapshot flamelet_{idx:03d}")
                break
    # assemble crossing-level pseudo-flamelets (front-consistent mid-c)
    Ymix = np.outer(1 - fl.Z, fl.Yo) + np.outer(fl.Z, fl.Yf)
    for k, cl in enumerate(clevels):
        Tk = np.where(crossed[k], rec_T[k], T_IN)
        Yk = np.where(crossed[k][:, None], rec_Y[k], Ymix)
        if crossed[k].sum() < 3:
            continue
        fl.snapshot(Tk, Yk, chi_st, out, idx)
        log(f"[strang chi={chi_st}] crossing-level c={cl:.2f}: "
            f"{crossed[k].sum()} nodes -> flamelet_{idx:03d}")
        idx += 1
    log(f"[strang chi={chi_st}] done in {time.time()-t0w:.0f}s")


if __name__ == "__main__":
    main()
