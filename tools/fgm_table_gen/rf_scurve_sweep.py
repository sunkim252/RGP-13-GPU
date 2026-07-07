"""Real-fluid flamelet S-curve sweep: chi_st continuation to extinction.

Solves the Z-space flamelet equations (chi imposed -> no momentum, no SRK
counterflow Newton stall) with SRK EOS + high-pressure-Chung transport at
P = 52.5 bar (wang2011 kerosene surrogate), in two transport modes:

    dd   differential diffusion: per-species Le_k = alpha/D_k
         (mixture-averaged, Takahashi-corrected high-pressure diffusivities)
    ule  unity Lewis: Le_k = 1  (the current dual-gas table's structure)

Ascending chi_st continuation with WARM START (previous converged solution)
+ periodic-reuse Jacobian pseudo-transient Newton, then bisection refine of
the extinction chi_ext. The dd/ule chi_ext ratio quantifies how much
extinction strain the unity-Le manifold under-predicts -- the documented
"a=1000" high-pressure gap, now with real-fluid transport.

  dY_k/dt = (chi/(2 Le_k)) d2Y_k/dZ2 + wdot_k*MW_k/rho
  dT/dt   = (chi/2) d2T/dZ2 - sum(hmol_k*wdot_k)/(rho*cp)
  chi(Z)  = chi_st * f(Z)/f(Zst),  f = exp(-2 erfc^-1(2Z)^2)

Usage:
  python3 rf_scurve_sweep.py --mode dd  --out data/rf_scurve [--n 41]
  python3 rf_scurve_sweep.py --mode ule --out data/rf_scurve
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import cantera as ct
from scipy.integrate import solve_ivp
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import lil_matrix, identity as sp_identity
from scipy.sparse.linalg import spsolve
from scipy.special import erfcinv

HERE = Path(__file__).resolve().parent
SRK = HERE / "data/wang2011_srk_v32.yaml"

P_BAR = 52.5
T_IN = 800.0
ZST = 0.2255          # chi-profile shape parameter (proto continuity)
BURN_T = 1500.0       # burning-branch classifier (frozen mixing ~800 K)


def log(m):
    print(m, flush=True)


class Flamelet:
    def __init__(self, mode, N):
        self.mode = mode
        self.N = N
        self.P = P_BAR * 1e5
        gas = ct.Solution(str(SRK))
        gas.transport_model = "high-pressure-Chung"
        self.gas = gas
        self.ns = gas.n_species
        self.MW = gas.molecular_weights
        gas.TPX = T_IN, self.P, {"NC10H22": 0.74, "PHC3H7": 0.15,
                                 "CYC9H18": 0.11}
        self.Yf = gas.Y.copy()
        gas.TPX = T_IN, self.P, "O2:1.0"
        self.Yo = gas.Y.copy()

        self.Z = np.linspace(0, 1, N)
        self.dZ = self.Z[1] - self.Z[0]
        Zc = np.clip(self.Z, 1e-4, 1 - 1e-4)
        self.shape = np.exp(-2*erfcinv(2*Zc)**2) \
            / np.exp(-2*erfcinv(2*ZST)**2)

        self.NI = N - 2
        self.nv = self.ns + 1
        # block-tridiagonal sparsity
        S = lil_matrix((self.NI*self.nv, self.NI*self.nv), dtype=np.int8)
        for j in range(self.NI):
            b = j*self.nv
            S[b:b+self.nv, b:b+self.nv] = 1
            for v in range(self.nv):
                if j > 0:
                    S[b+v, (j-1)*self.nv+v] = 1
                if j < self.NI-1:
                    S[b+v, (j+1)*self.nv+v] = 1
        self.S = S.tocsr()
        self.Imat = sp_identity(self.NI*self.nv, format="csc")
        self.neval = 0

    def pack(self, T, Y):
        return np.concatenate([T[1:-1], Y[1:-1].ravel()])

    def unpack(self, u):
        T = np.empty(self.N)
        Y = np.empty((self.N, self.ns))
        T[0] = T_IN; T[-1] = T_IN
        Y[0] = self.Yo; Y[-1] = self.Yf
        T[1:-1] = u[:self.NI]
        Y[1:-1] = u[self.NI:].reshape(self.NI, self.ns)
        return T, Y

    def init_equilibrium(self):
        T0 = np.zeros(self.N); Y0 = np.zeros((self.N, self.ns))
        g = self.gas
        for i in range(self.N):
            z = self.Z[i]
            g.TPY = T_IN, self.P, z*self.Yf + (1 - z)*self.Yo
            try:
                g.equilibrate("HP")
                T0[i] = g.T; Y0[i] = g.Y
            except Exception:
                T0[i] = T_IN; Y0[i] = z*self.Yf + (1 - z)*self.Yo
        T0[0] = T_IN; Y0[0] = self.Yo
        T0[-1] = T_IN; Y0[-1] = self.Yf
        return self.pack(T0, Y0)

    def rhs(self, chi_st, u):
        """Vectorised via ct.SolutionArray: one batched Cantera evaluation of
        all interior points instead of a Python per-point loop (~10x)."""
        self.neval += 1
        chi = chi_st*self.shape
        T, Y = self.unpack(u)

        if not hasattr(self, "_arr"):
            self._arr = ct.SolutionArray(self.gas, self.NI)
        arr = self._arr
        arr.TPY = (np.maximum(T[1:-1], 300.0), self.P,
                   np.clip(Y[1:-1], 0, None))

        wdot = arr.net_production_rates          # (NI, ns) kmol/m3/s
        rho = arr.density                        # (NI,)
        cp = arr.cp_mass
        hmol = arr.partial_molar_enthalpies      # (NI, ns) J/kmol

        lapT = (T[2:] - 2*T[1:-1] + T[:-2])/self.dZ**2
        lapY = (Y[2:] - 2*Y[1:-1] + Y[:-2])/self.dZ**2

        if self.mode == "dd":
            alpha = arr.thermal_conductivity/(rho*cp)
            Dk = np.clip(arr.mix_diff_coeffs, 1e-12, None)
            Lek = alpha[:, None]/Dk              # (NI, ns)
        else:
            Lek = 1.0

        chI = chi[1:-1]
        dT = (chI/2)*lapT - np.einsum("ij,ij->i", hmol, wdot)/(rho*cp)
        dY = (chI[:, None]/(2*Lek))*lapY + wdot*self.MW[None, :]/rho[:, None]
        return np.concatenate([dT, dY.ravel()])

    def wnorm(self, F):
        Fr = F.copy(); Fr[:self.NI] /= 1000.0
        return np.linalg.norm(Fr)/np.sqrt(F.size)

    def clip(self, u):
        """Physical box: T in [300, 4200] K, Y in [0, 1]."""
        u[:self.NI] = np.clip(u[:self.NI], 300.0, 4200.0)
        u[self.NI:] = np.clip(u[self.NI:], 0.0, 1.0)
        return u

    def solve(self, chi_st, u0, maxit=400, tol=3e-3, jac_every=8,
              dt0=1e-8, verbose=False):
        """Pseudo-transient continuation (PTC): implicit-Euler steps
        (I/dt - J) du = F with STEP-SIZE control (max |dT| per step), not
        residual-descent control. Residual-monotone line search deadlocks
        on the stiff transient path (nt==nrm as lam->0), while raw Newton
        with unbounded dt runs T away -- bounding the physical step and
        clipping the state is the classical robust middle ground."""
        u = self.clip(u0.copy()); dt = dt0
        Jlu = None; jage = 999; jdt = dt
        f = lambda v: self.rhs(chi_st, v)
        t0 = time.time()
        for it in range(maxit):
            Fu = f(u); nrm = self.wnorm(Fu)
            if nrm < tol:
                T, _ = self.unpack(u)
                return u, True, T.max(), it, time.time()-t0
            if not np.isfinite(nrm) or nrm > 1e7:
                return u, False, self.unpack(u)[0].max(), it, time.time()-t0
            if jage >= jac_every or dt < 0.3*jdt or dt > 8*jdt:
                J = approx_derivative(f, u, method="2-point",
                                      sparsity=self.S)
                Jlu = (self.Imat/dt - J).tocsc()
                jdt = dt; jage = 0
            try:
                du = spsolve(Jlu, Fu)
            except Exception:
                return u, False, self.unpack(u)[0].max(), it, time.time()-t0

            mxT = np.abs(du[:self.NI]).max()
            if mxT > 120.0:
                # step too large for this dt: retry smaller (no state update)
                dt = max(dt*0.3, 1e-11)
                jage = 999
                if verbose:
                    log(f"      it {it:3d}: |F|w={nrm:.3e} dt={dt:.1e} "
                        f"RETRY mxT={mxT:.0f}")
                continue

            u = self.clip(u + du)
            jage += 1
            if mxT < 25.0:
                dt = min(dt*1.7, 3.0)
            if verbose and it % 5 == 0:
                log(f"      it {it:3d}: |F|w={nrm:.3e} dt={dt:.1e} "
                    f"mxT={mxT:.1f} Tmax={self.unpack(u)[0].max():.0f}")
        T, _ = self.unpack(u)
        Fu = f(u)
        return u, self.wnorm(Fu) < 10*tol, T.max(), maxit, time.time()-t0

    def solve_transient(self, chi_st, u0, ndiff=40.0, verbose=False):
        """Integrate the physical transient to (near-)steady state with a
        proper stiff integrator (BDF + block-tridiagonal sparsity). The
        custom pseudo-transient Newton wanders super-equilibrium at 52.5 bar;
        BDF follows the true relaxation, and extinction is detected
        PHYSICALLY: a burning initial state quenches to the mixing solution
        when chi_st exceeds chi_ext.  ndiff = integration span in diffusion
        times 1/chi_st."""
        t0 = time.time()
        self._tlast = [time.time()]
        def f(t, v):
            if time.time() - self._tlast[0] > 60:
                self._tlast[0] = time.time()
                log(f"      BDF progress: t={t:.3e}/{ndiff/chi_st:.3e} "
                    f"nfev={self.neval} ({time.time()-t0:.0f}s)")
            return self.rhs(chi_st, self.clip(v.copy()))
        tend = ndiff/chi_st
        sol = solve_ivp(f, (0.0, tend), self.clip(u0.copy()),
                        method="BDF", jac_sparsity=self.S,
                        rtol=2e-4, atol=1e-9, first_step=1e-8/chi_st)
        u = self.clip(sol.y[:, -1].copy())
        T, _ = self.unpack(u)
        nrm = self.wnorm(self.rhs(chi_st, u))
        ok = sol.success
        if verbose:
            log(f"      BDF: {sol.status} nsteps={sol.t.size} "
                f"|F|w_end={nrm:.2e} Tmax={T.max():.0f} "
                f"({time.time()-t0:.0f}s, nfev={self.neval})")
        return u, ok, T.max(), sol.t.size, time.time()-t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("dd", "ule"), required=True)
    ap.add_argument("--n", type=int, default=41)
    ap.add_argument("--out", default=str(HERE/"data/rf_scurve"))
    ap.add_argument("--chi0", type=float, default=100.0)
    ap.add_argument("--chi-max", type=float, default=2.0e5)
    ap.add_argument("--fac", type=float, default=2.0,
                    help="chi multiplier per continuation step")
    ap.add_argument("--nbisect", type=int, default=5)
    ap.add_argument("--single", type=float, default=None,
                    help="debug: solve one chi verbosely and exit")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fl = Flamelet(args.mode, args.n)
    log(f"[{args.mode}] P={P_BAR}bar N={args.n} ns={fl.ns} "
        f"unknowns={fl.NI*fl.nv}")

    u = fl.init_equilibrium()
    T0, _ = fl.unpack(u)
    log(f"[{args.mode}] equilibrium init T_max={T0.max():.0f}K")

    if args.single is not None:
        u_try, ok, Tmax, its, secs = fl.solve_transient(args.single, u, verbose=True)
        log(f"[{args.mode}] SINGLE chi={args.single}: "
            f"{'CONV' if ok else 'FAIL'} T_max={Tmax:.1f} it={its} "
            f"{secs:.0f}s nfev={fl.neval}")
        return

    results = []           # (chi, Tmax, burning, iters, secs)
    chi = args.chi0
    u_burn, chi_burn = None, None
    chi_ext_lo, chi_ext_hi = None, None

    first = True
    while chi <= args.chi_max:
        u_try, ok, Tmax, its, secs = fl.solve_transient(chi, u)
        if first and not ok:
            log(f"[{args.mode}] first point chi={chi} failed -- retry with "
                "smaller dt0 / more iterations")
            u_try, ok, Tmax, its, secs = fl.solve_transient(chi, u, ndiff=80.0)
        first = False
        burning = ok and (Tmax > BURN_T)
        log(f"[{args.mode}] chi={chi:10.1f}  T_max={Tmax:7.1f}K "
            f"{'BURN' if burning else ('ext ' if ok else 'FAIL')} "
            f"(it={its}, {secs:.0f}s, nfev={fl.neval})")
        results.append((chi, float(Tmax), bool(burning), its, secs))
        if burning:
            u = u_try
            u_burn, chi_burn = u.copy(), chi
            np.savez(out/f"fl_{args.mode}_chi{chi:g}.npz",
                     Z=fl.Z, T=fl.unpack(u)[0], Y=fl.unpack(u)[1],
                     chi_st=chi, P=fl.P, mode=args.mode)
            chi *= args.fac
        else:
            chi_ext_lo, chi_ext_hi = chi_burn, chi
            break

    # bisection refine of extinction chi
    if chi_ext_lo is not None and chi_ext_hi is not None:
        for k in range(args.nbisect):
            mid = np.sqrt(chi_ext_lo*chi_ext_hi)   # log bisection
            u_try, ok, Tmax, its, secs = fl.solve_transient(mid, u_burn, ndiff=60.0)
            burning = ok and (Tmax > BURN_T)
            log(f"[{args.mode}] bisect {k}: chi={mid:10.1f} "
                f"T_max={Tmax:7.1f}K {'BURN' if burning else 'ext'} "
                f"(it={its}, {secs:.0f}s)")
            results.append((mid, float(Tmax), bool(burning), its, secs))
            if burning:
                chi_ext_lo = mid; u_burn = u_try
                np.savez(out/f"fl_{args.mode}_chi{mid:g}.npz",
                         Z=fl.Z, T=fl.unpack(u_try)[0],
                         Y=fl.unpack(u_try)[1],
                         chi_st=mid, P=fl.P, mode=args.mode)
            else:
                chi_ext_hi = mid
        log(f"[{args.mode}] EXTINCTION chi_st in "
            f"[{chi_ext_lo:.1f}, {chi_ext_hi:.1f}] 1/s")
    else:
        log(f"[{args.mode}] no extinction found up to chi={args.chi_max}")

    with open(out/f"scurve_{args.mode}.json", "w") as fh:
        json.dump({"mode": args.mode, "P_bar": P_BAR, "N": args.n,
                   "results": results,
                   "chi_ext_lo": chi_ext_lo, "chi_ext_hi": chi_ext_hi},
                  fh, indent=1)
    log(f"[{args.mode}] saved {out}/scurve_{args.mode}.json")


if __name__ == "__main__":
    main()
