"""Real-fluid flamelet at a single strain, SESSION GEOMETRY (matches the prior
dual-gas Fig19 work so results overlay on the same strain axis):
  - DOMAIN_WIDTH = 0.02 m,  EQUAL mdot on both inlets (as 05_flamelet_sweep)
  - strain a = 100 * U_ox = 100 * mdot / rho_ox   (a = 2*U_ox/L, L=0.02)
  - a=1000  <->  mdot = 10 * rho_ox
Solved DIRECTLY with SRK + high-pressure-Chung via auto=True grid sequencing
(the proven-robust path; avoids the auto=False SRK Newton stall the session hit).

Usage: python rf_flamelet_sg.py P_atm [a]
"""
import sys, time, json
import numpy as np, cantera as ct

P_ATM = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
A = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
P = P_ATM * 101325.0
TIN = 800.0
WIDTH = 0.02
FUEL = {'NC10H22': 0.74, 'PHC3H7': 0.15, 'CYC9H18': 0.11}
OXID = {'O2': 1.0}
MECH = 'data/wang2011_srk_v32.yaml'
MAX_GRID = 600
def log(m): print(m, flush=True)

gas = ct.Solution(MECH); gas.transport_model = 'high-pressure-Chung'
gas.TPX = TIN, P, OXID; rho_ox = gas.density
gas.TPX = TIN, P, FUEL; rho_f = gas.density
mdot = A * rho_ox / 100.0            # a = 100*U_ox = 100*mdot/rho_ox
U_ox = mdot / rho_ox; U_f = mdot / rho_f
log(f"=== P={P_ATM}atm a={A:g}  rho_ox={rho_ox:.3f} rho_f={rho_f:.3f}  "
    f"mdot={mdot:.3f} kg/m2s  U_ox={U_ox:.3f} U_f={U_f:.3f} m/s  width={WIDTH*1e3:.0f}mm ===")

def fwhm(x, y, base):
    yy = y - base; pk = yy.max()
    if pk <= 0: return float('nan')
    half = base + 0.5 * pk; idx = np.where(y >= half)[0]
    if len(idx) < 2: return float('nan')
    iL, iR = idx[0], idx[-1]
    xL = x[iL] if iL == 0 else np.interp(half, [y[iL-1], y[iL]], [x[iL-1], x[iL]])
    xR = x[iR] if iR == len(x)-1 else np.interp(half, [y[iR], y[iR+1]], [x[iR], x[iR+1]])
    return xR - xL

f = ct.CounterflowDiffusionFlame(gas, width=WIDTH)
f.P = P
f.fuel_inlet.mdot = mdot;      f.fuel_inlet.X = FUEL;  f.fuel_inlet.T = TIN
f.oxidizer_inlet.mdot = mdot;  f.oxidizer_inlet.X = OXID; f.oxidizer_inlet.T = TIN
f.set_max_grid_points(f.flame, MAX_GRID)
f.set_refine_criteria(ratio=3.0, slope=0.12, curve=0.24, prune=0.03)

t0 = time.time()
# Stage 1: ignite + resolve with mixture-averaged transport (thicker flame ->
# survives coarse grids; high-pressure-Chung's thin Takahashi flame extinguishes
# on coarse grids on the wide session domain). EOS stays SRK throughout.
f.transport_model = 'mixture-averaged'
f.set_initial_guess()
try:
    f.solve(loglevel=1, auto=True)
except Exception as e:
    log(f"STAGE1 (mixture-averaged) FAILED: {e}"); sys.exit(1)
log(f"  stage1 mixture-averaged: Tmax={f.T.max():.0f}K npts={f.grid.size} ({time.time()-t0:.0f}s)")
# Stage 2: switch transport to real-fluid high-pressure-Chung (Chung mu/kappa +
# Takahashi diffusion), re-solve from the burning MA solution.
f.transport_model = 'high-pressure-Chung'
try:
    f.solve(loglevel=1, auto=True)
except Exception as e:
    log(f"STAGE2 (high-pressure-Chung) FAILED: {e}"); sys.exit(1)
Tmax = float(f.T.max())
x = f.grid.copy(); T = f.T.copy(); hrr = f.heat_release_rate.copy()
try: Z = f.mixture_fraction('Bilger')
except Exception: Z = np.zeros_like(x)
dT = fwhm(x, T, TIN); dQ = fwhm(x, hrr, 0.0)
q_int = float(np.trapz(hrr, x))
sr = {}
for d in ('max', 'mean', 'stoichiometric', 'potential_flow_fuel', 'potential_flow_oxidizer'):
    try: sr[d] = float(f.strain_rate(d))
    except Exception: sr[d] = float('nan')
rec = dict(P_atm=P_ATM, a=A, mdot=float(mdot), U_ox=float(U_ox), U_f=float(U_f),
           Tmax=Tmax, delta_T_mm=float(dT*1e3), delta_hrr_mm=float(dQ*1e3),
           q_int=q_int, npts=len(x), strain=sr, burning=bool(Tmax > 1500), dt=time.time()-t0)
log(f"\n=== P={P_ATM}atm a={A:g}: Tmax={Tmax:.0f}K  deltaT={dT*1e3:.4f}mm  q_int={q_int:.3e}  "
    f"sr_max={sr['max']:.0f} sr_pf_ox={sr['potential_flow_oxidizer']:.0f}  npts={len(x)} "
    f"({rec['dt']:.0f}s) {'BURNING' if Tmax>1500 else 'EXTINGUISHED'}")
np.savez(f'data/rf_sg_P{P_ATM:g}_a{A:g}.npz', x=x, T=T, Y=f.Y, hrr=hrr, Z=Z, P=P, a=A,
         Tmax=Tmax, delta_T_mm=dT*1e3, q_int=q_int, mdot=mdot, U_ox=U_ox,
         species=np.array(gas.species_names), **{f'sr_{k}': v for k, v in sr.items()})
with open(f'data/rf_sg_P{P_ATM:g}_a{A:g}.json', 'w') as fp:
    json.dump(rec, fp, indent=2)
log(f"saved data/rf_sg_P{P_ATM:g}_a{A:g}.npz")
