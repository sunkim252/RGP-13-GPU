"""Diagnose why the strain sweep stalls at idx-4: restore idx-3, apply one
Fiala-Sattelmayer strain step (x1.25), and solve with loglevel=1 to watch."""
import numpy as np, cantera as ct, time

gi = ct.Solution("data/wang2011_ideal_v32.yaml")
fi = ct.CounterflowDiffusionFlame(gi, width=0.02)
fi.transport_model = "mixture-averaged"
fi.restore("data/ckpt_dualgas/stageE_idx_003.yaml", name="E")
print("restored idx-3: mdot=%.4f npts=%d Tmax=%.1f"
      % (fi.fuel_inlet.mdot, fi.grid.size, fi.T.max()), flush=True)
fi.max_time_step_count = 2000

A = dict(grid=-1/2, mdot=1/2, u=1/2, V=1.0, lam=2.0)
r = 1.25
fi.flame.grid = fi.flame.grid * (r ** A["grid"])
fi.fuel_inlet.mdot *= r ** A["mdot"]
fi.oxidizer_inlet.mdot *= r ** A["mdot"]
g = np.asarray(fi.grid); span = max(g[-1] - g[0], 1e-30); nm = g / span
fi.set_profile("velocity",    nm, np.asarray(fi.velocity) * (r ** A["u"]))
fi.set_profile("spread_rate", nm, np.asarray(fi.spread_rate) * (r ** A["V"]))
fi.set_profile("lambda",      nm, np.asarray(fi.L) * (r ** A["lam"]))
print("scaled to mdot=%.4f, solving auto=False loglevel=1 ..."
      % fi.fuel_inlet.mdot, flush=True)
t0 = time.time()
try:
    fi.solve(loglevel=1, auto=False)
    print("AUTO=FALSE OK npts=%d Tmax=%.1f dt=%.0fs"
          % (fi.grid.size, fi.T.max(), time.time() - t0), flush=True)
except Exception as e:
    print("AUTO=FALSE FAIL: %s dt=%.0fs"
          % (str(e).splitlines()[-1][:60], time.time() - t0), flush=True)
