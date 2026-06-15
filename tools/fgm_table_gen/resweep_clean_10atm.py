"""Clean re-solve of the under-converged 10 atm flamelets that create the
cool T(c) dip near c~0.9 (Wang-2015 10 atm validation).

Diagnosis (quality scan of flamelets_dualgas_P10atm, Zst=0.2255):
  - idx5,6,7,8 are the LOWEST-strain points (chi<1) yet T(Zst)~3300-3390 K,
    i.e. ~150-230 K below the family max 3531 -- a low-strain flame must sit
    near equilibrium (hottest), so these are half-transitioned (the dual-gas
    sweep's accept-if-burning at max_time_step_count=150 stopped early).
  - idx13 (Z@Tmax=0.207, T(Zst)=3181) and idx38 (Z@Tmax=0.187, T(Zst)=3096)
    fail the lean/cool gate.
These cool high-c states are what the envelope c-normalisation drops at c~0.9,
producing the non-monotonic T(c) dip the FPV solver then turns into spurious
dilatation.

Strategy (per redo_idx12_P10atm / resweep_clean_50atm): restore a HEALTHY
neighbour's checkpoint, scale mdot to the target, robust multi-round settle,
gate on Z@Tmax in [0.18,0.33] & T(Zst) > 3400, overwrite flamelet_0NN.npz.
"""
import importlib.util, time
from pathlib import Path
import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dg", HERE / "05_flamelet_sweep_dualgas.py")
dg = importlib.util.module_from_spec(spec); spec.loader.exec_module(dg)
swp = dg.swp

CKPT = HERE / "data/ckpt_dualgas_P10atm"
OUT  = HERE / "data/flamelets_dualgas_P10atm"
Zst  = 0.2255
TZST_MIN = 3400.0

# bad idx -> (healthy source idx for the checkpoint, target mdot)
TARGETS = {
    5:  (4,  0.063),
    6:  (4,  0.067),
    7:  (4,  0.070),
    8:  (4,  0.073),
    13: (11, 0.093),
    38: (37, 0.306),
}


def _metrics(flame):
    Zb = swp._bilger_Z(flame); o = np.argsort(Zb); T = flame.T
    Tzst = float(np.interp(Zst, Zb[o], T[o]))
    ztmax = float(Zb[T.argmax()])
    return Tzst, ztmax, float(T.max())


def _robust_settle(flame, rounds=6, step_cap=400):
    flame.max_time_step_count = step_cap
    Tprev = None
    for rnd in range(1, rounds + 1):
        declared = True
        try:
            flame.solve(loglevel=0, auto=False)
        except ct.CanteraError:
            if flame.T.max() < 1500.0:
                raise
            declared = False
        Tzst, ztmax, Tm = _metrics(flame)
        ok = (0.18 <= ztmax <= 0.33) and (Tzst > TZST_MIN)
        print(f"      round {rnd}: declared={declared} Tmax={Tm:.0f} "
              f"T(Zst)={Tzst:.0f} Z@Tmax={ztmax:.3f} gate={'OK' if ok else '..'}",
              flush=True)
        stationary = (Tprev is not None and abs(Tm - Tprev) < 2.0)
        if (declared and ok) or (stationary and ok):
            return True
        Tprev = Tm
    return False


def main():
    igas, _ = dg._load(dg.IDEAL_YAML)
    sgas, _ = dg._load(dg.SRK_YAMLS)
    for idx in sorted(TARGETS):
        src, target = TARGETS[idx]
        ck = CKPT / f"stageE_idx_{src:03d}.yaml"
        flame = swp._ckpt_restore(igas, ck, "E", transport=dg.TRANSPORT_IDEAL)
        cur = float(flame.fuel_inlet.mdot)
        Tzst0, zt0, Tm0 = _metrics(flame)
        print(f"\nidx{idx}: restored src idx{src} mdot={cur:.4f} "
              f"T(Zst)={Tzst0:.0f} -> scale to mdot={target:.4f}", flush=True)
        swp._scale_apply(flame, target / cur, dg.A_EXP)
        t0 = time.time()
        ok = _robust_settle(flame)
        Tzst, ztmax, Tm = _metrics(flame)
        spikes = int((np.abs(np.diff(flame.T, 2)) > 500.0).sum())
        print(f"  idx{idx}: Tmax={Tm:.0f} Z@Tmax={ztmax:.3f} T(Zst)={Tzst:.0f} "
              f"spikes={spikes} gate={'OK' if ok else 'FAIL'} {time.time()-t0:.0f}s",
              flush=True)
        if ok and not spikes:
            dg._save_flamelet_realfluid(flame, sgas, OUT, idx)
            print(f"  flamelet_{idx:03d}.npz OVERWRITTEN (healthy)", flush=True)
        else:
            print(f"  idx{idx} NOT overwritten (gate/spike fail)", flush=True)
    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
