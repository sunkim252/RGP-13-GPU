"""CRYOGENIC / transcritical real-fluid counterflow flamelet:
  oxidizer = LOx at T_OX (cryogenic, dense transcritical at rocket P)
  fuel     = kerosene surrogate at T_FUEL (liquid-like)
SRK (Redlich-Kwong) EOS + high-pressure-Chung transport, two-stage auto solve.

Purpose: see whether the dense->gas (pseudo-boiling) transition across the flame
is tractable, and how far the adiabatic flame temperature drops vs the 800K-inlet
table. ** CAVEAT: NASA-7 thermo for all species floors at ~300K; below that
(cold O2 stream) cp/h are EXTRAPOLATED -> cold-stream sensible enthalpy is
approximate. EOS density is physical (RK), thermo below 300K is not. **

Usage: python rf_flamelet_cryo.py [P_atm] [a] [T_ox] [T_fuel]
"""
import sys, time, json
import numpy as np, cantera as ct

P_ATM = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
A     = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
T_OX  = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
T_FUEL= float(sys.argv[4]) if len(sys.argv) > 4 else 300.0
P = P_ATM * 101325.0
WIDTH = 0.02
FUEL = {'NC10H22': 0.74, 'PHC3H7': 0.15, 'CYC9H18': 0.11}
OXID = {'O2': 1.0}
MECH = 'data/wang2011_srk_v32.yaml'
MAX_GRID = 600
def log(m): print(m, flush=True)

gas = ct.Solution(MECH); gas.transport_model = 'high-pressure-Chung'
gas.TPX = T_OX, P, OXID;  rho_ox = gas.density
gas.TPX = T_FUEL, P, FUEL; rho_f = gas.density
mdot = A * rho_ox / 100.0
U_ox = mdot / rho_ox; U_f = mdot / rho_f
log(f"=== CRYO P={P_ATM}atm a={A:g}  T_ox={T_OX}K T_fuel={T_FUEL}K ===")
log(f"  rho_ox={rho_ox:.1f} (LOx dense!) rho_f={rho_f:.1f} kg/m3  mdot={mdot:.1f}  U_ox={U_ox:.2f} U_f={U_f:.2f} m/s")
# adiabatic flame T at stoich for reference (HP equilibrium of cold mix)
g2 = ct.Solution(MECH)
g2.set_mixture_fraction(0.2255, FUEL, OXID)   # stoich-ish by Bilger? use eq ratio
g2.set_equivalence_ratio(1.0, FUEL, OXID)
# mass-weighted inlet enthalpy: blend cold streams
g2.TP = T_OX, P
log(f"  (ref) cold O2 rho range across flame: 100K={rho_ox:.0f} -> hot ~50 kg/m3")

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
f.fuel_inlet.mdot = mdot;      f.fuel_inlet.X = FUEL;  f.fuel_inlet.T = T_FUEL
f.oxidizer_inlet.mdot = mdot;  f.oxidizer_inlet.X = OXID; f.oxidizer_inlet.T = T_OX
f.set_max_grid_points(f.flame, MAX_GRID)
f.set_refine_criteria(ratio=3.0, slope=0.12, curve=0.24, prune=0.03)

t0 = time.time()
f.transport_model = 'mixture-averaged'
f.set_initial_guess()
try:
    f.solve(loglevel=1, auto=True)
except Exception as e:
    log(f"STAGE1 (mixture-averaged) FAILED: {e}"); sys.exit(1)
log(f"  stage1 MA: Tmax={f.T.max():.0f}K npts={f.grid.size} ({time.time()-t0:.0f}s)")
f.transport_model = 'high-pressure-Chung'
try:
    f.solve(loglevel=1, auto=True)
except Exception as e:
    log(f"STAGE2 (Chung) FAILED: {e} -- keeping MA result")
Tmax = float(f.T.max())
x = f.grid.copy(); T = f.T.copy(); hrr = f.heat_release_rate.copy()
rho = f.density.copy()
try: Z = f.mixture_fraction('Bilger')
except Exception: Z = np.zeros_like(x)
iZst = int(np.argmin(np.abs(Z - 0.2255)))
dT = fwhm(x, T, T.min())
log(f"\n=== CRYO P={P_ATM}atm: Tmax={Tmax:.0f}K  T@Zst={T[iZst]:.0f}K  deltaT={dT*1e3:.4f}mm")
log(f"    density across flame: ox-side={rho[0 if Z[0]<Z[-1] else -1]:.0f} -> peak-T={rho[np.argmax(T)]:.1f} kg/m3 (transcritical span)")
log(f"    npts={len(x)} ({time.time()-t0:.0f}s) {'BURNING' if Tmax>1500 else 'WEAK/EXTINCT'}")
np.savez(f'data/rf_cryo_P{P_ATM:g}_Tox{T_OX:g}.npz', x=x, T=T, Y=f.Y, hrr=hrr, Z=Z, rho=rho,
         P=P, a=A, T_ox=T_OX, T_fuel=T_FUEL, Tmax=Tmax, species=np.array(gas.species_names))
log(f"saved data/rf_cryo_P{P_ATM:g}_Tox{T_OX:g}.npz")
