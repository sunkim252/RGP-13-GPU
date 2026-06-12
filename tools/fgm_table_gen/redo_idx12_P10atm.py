"""Re-converge the corrupted 10 atm strain point idx=12 (Z@Tmax=0.046,
T-profile spikes) from the healthy idx=11 checkpoint, with a MUCH larger
settle budget than the sweep's accept-if-burning shortcut, then overwrite
flamelet_012.npz."""
import importlib.util
from pathlib import Path
import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "dg", HERE / "05_flamelet_sweep_dualgas.py")
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)
swp = dg.swp

CKPT = HERE / "data/ckpt_dualgas_P10atm/stageE_idx_011.yaml"
OUT = HERE / "data/flamelets_dualgas_P10atm"

igas, _ = dg._load(dg.IDEAL_YAML)
sgas, _ = dg._load(dg.SRK_YAMLS)

flame = swp._ckpt_restore(igas, CKPT, "E", transport=dg.TRANSPORT_IDEAL)
print(f"restored idx11: mdot={flame.fuel_inlet.mdot:.4f} "
      f"Tmax={flame.T.max():.1f} npts={flame.grid.size}", flush=True)

# one strain step x1.10 (same as the sweep)
swp._scale_apply(flame, 1.10, dg.A_EXP)
print(f"scaled to mdot={flame.fuel_inlet.mdot:.4f}; robust settle...",
      flush=True)

flame.max_time_step_count = 400
Tprev = None
for rnd in range(1, 7):
    declared = True
    try:
        flame.solve(loglevel=0, auto=False)
    except ct.CanteraError:
        declared = False
    Tm = float(flame.T.max())
    print(f"  round {rnd}: declared={declared} Tmax={Tm:.1f}", flush=True)
    if declared:
        break
    if Tprev is not None and abs(Tm - Tprev) < 2.0:
        print("  Tmax stationary across rounds -> converged in practice",
              flush=True)
        break
    Tprev = Tm

# health check before overwriting
T = np.asarray(flame.T)
Z = swp._bilger_Z(flame)
izm = int(np.argmax(T))
spikes = int((np.abs(np.diff(T, 2)) > 500.0).sum())
print(f"health: Tmax={T.max():.1f} @Z={Z[izm]:.3f}, T-spikes={spikes}",
      flush=True)
if not (0.15 <= Z[izm] <= 0.35) or spikes:
    raise SystemExit("REDO FAILED health check -- not overwriting")

dg._save_flamelet_realfluid(flame, sgas, OUT, 12)
print("flamelet_012.npz OVERWRITTEN (healthy)", flush=True)
