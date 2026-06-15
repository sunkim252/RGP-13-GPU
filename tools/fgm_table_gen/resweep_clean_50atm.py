"""Clean re-solve of the noisy mid-strain 50 atm flamelets (idx 19..27).

The dual-gas stage-E sweep (05_flamelet_sweep_dualgas.run) solves each strain
point with `solve(auto=False)` capped at max_time_step_count=150 and ACCEPTS the
time-stepped result whenever Tmax>1500 K ("accept-if-burning"). At the mid-mdot
points (0.42..1.0) 150 steps is not enough to reach steady, so half-transitioned
states were saved (e.g. idx22: Tmax=3239 K, Z@Tmax=0.145, T(Zst)=2789 K, chi=87 --
a cool, lean-shifted, NON-physical outlier). Those break the FPV manifold: the
envelope-normalization picks the high-C cool state as the c=1 closure.

This script keeps the clean idx0..18 (T(Zst)~3740-3790, Z@Tmax~0.245) and
re-solves idx19..27 by CONTINUATION from idx18's checkpoint with strict
convergence: multi-round auto=False settle (max_time_step_count=2000) until Tmax
is stationary, plus a quality gate (Z@Tmax in [0.18,0.33], T(Zst)>3650 -- all
these points have chi<<extinction so they must be near-equilibrium). It
overwrites flamelet_0NN.npz and stageE_idx_0NN.yaml in place.
"""
import importlib.util, time
from pathlib import Path
import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dg", HERE / "05_flamelet_sweep_dualgas.py")
dg = importlib.util.module_from_spec(spec); spec.loader.exec_module(dg)
swp = dg.swp

DATA = dg.DATA
OUT  = DATA / "flamelets_dualgas"
CKPT = DATA / "ckpt_dualgas"
Zst  = 0.2255

# Existing mdot grid (matches the deployed table's strain axis).
MDOTS = {19: 0.417, 20: 0.466, 21: 0.521, 22: 0.582,
         23: 0.651, 24: 0.728, 25: 0.813, 26: 0.909, 27: 1.017}
START_IDX = 18   # last clean checkpoint to continue from


def _metrics(flame):
    Zb = flame.mixture_fraction("Bilger"); o = np.argsort(Zb)
    T = flame.T
    Tzst = float(np.interp(Zst, Zb[o], T[o]))
    ztmax = float(Zb[T.argmax()])
    return Tzst, ztmax, float(T.max())


def _gate(flame):
    Zb = flame.mixture_fraction("Bilger"); o = np.argsort(Zb)
    Tzst = float(np.interp(Zst, Zb[o], flame.T[o]))
    ztmax = float(Zb[flame.T.argmax()])
    return (0.18 <= ztmax <= 0.33) and (Tzst > 3650.0), Tzst, ztmax


def _robust_settle(flame, log, rounds=5, step_cap=400):
    """Bounded multi-round auto=False. A single round at max_time_step_count=2000
    is ~400 s here (one round time-steps until the cap, since Cantera rarely
    DECLARES steady at these conditions). So cap the steps LOW per round and do
    several rounds, breaking as soon as Tmax is stationary (<2 K) AND the quality
    gate passes (the gate distinguishes a real near-equilibrium steady state from
    a half-transitioned one -- the exact failure mode of the 150-step sweep)."""
    flame.max_time_step_count = step_cap
    Tprev = None
    for rnd in range(1, rounds + 1):
        declared = True
        try:
            flame.solve(loglevel=log, auto=False)
        except ct.CanteraError:
            if flame.T.max() < 1500.0:
                raise
            declared = False
        Tm = float(flame.T.max())
        ok, Tzst, ztmax = _gate(flame)
        dg._log(f"    round {rnd}: declared={declared} Tmax={Tm:.1f} "
                f"T(Zst)={Tzst:.0f} Z@Tmax={ztmax:.3f} gate={'OK' if ok else 'FAIL'}")
        stationary = (Tprev is not None and abs(Tm - Tprev) < 2.0)
        if (declared and ok) or (stationary and ok):
            return declared, rnd
        Tprev = Tm
    return False, rounds


def main(log=0):
    igas, _ = dg._load(dg.IDEAL_YAML)
    sgas, _ = dg._load(dg.SRK_YAMLS)
    prev_ckpt = CKPT / f"stageE_idx_{START_IDX:03d}.yaml"
    flame = dg._restore_ideal(igas, prev_ckpt, "E")
    cur_mdot = float(flame.fuel_inlet.mdot)
    dg._log(f"restored idx{START_IDX} mdot={cur_mdot:.3f} npts={flame.grid.size}")

    for idx in range(START_IDX + 1, 28):
        target = MDOTS[idx]
        factor = target / cur_mdot
        swp._scale_apply(flame, factor, dg.A_EXP)
        t0 = time.time()
        declared, rnds = _robust_settle(flame, log)
        Tzst, ztmax, Tmax = _metrics(flame)
        ok = (0.18 <= ztmax <= 0.33) and (Tzst > 3650.0)
        flag = "OK" if ok else "QUALITY-GATE-FAIL"
        dg._log(f"  idx{idx:02d} mdot={float(flame.fuel_inlet.mdot):.3f} "
                f"npts={flame.grid.size} Tmax={Tmax:.0f} Z@Tmax={ztmax:.3f} "
                f"T(Zst)={Tzst:.0f} declared={declared} rnds={rnds} "
                f"{time.time()-t0:.0f}s [{flag}]")
        # Save (overwrite) regardless, but the flag tells us if a point needs
        # a finer continuation step.
        dg._save_flamelet_realfluid(flame, sgas, OUT, idx)
        swp._ckpt_save(flame, CKPT / f"stageE_idx_{idx:03d}.yaml", "E",
                       f"ideal strain idx={idx} (clean re-solve)")
        cur_mdot = float(flame.fuel_inlet.mdot)
    dg._log("clean re-sweep done (idx19..27)")


if __name__ == "__main__":
    import sys
    main(log=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
