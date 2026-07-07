"""VALIDATION-FIRST real-fluid flamelets: stable-branch strain continuation.

Cantera CounterflowDiffusionFlame (the validated core FPVgen wraps) with
SRK EOS + high-pressure-Chung (= Chung mu/kappa + Takahashi diffusion).
We avoid FPVgen's two-point S-curve continuation (which stalled); instead we
warm-step the GLOBAL strain a = 2*v_tot/width up the stable branch until the
target strain (a=1000) or extinction. Per pressure -> T_max, delta(FWHM), q_dot.

Usage: python rf_strain_sweep.py P_atm [a1,a2,...]
"""
import sys, time, json
import numpy as np, cantera as ct

P_ATM = float(sys.argv[1]) if len(sys.argv) > 1 else 25.0
A_TARGETS = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [200, 600, 1000]
P = P_ATM * 101325.0
TIN = 800.0
FUEL = {'NC10H22': 0.74, 'PHC3H7': 0.15, 'CYC9H18': 0.11}
OXID = {'O2': 1.0}
MECH = 'data/wang2011_srk_v32.yaml'
def log(m): print(m, flush=True)

gas = ct.Solution(MECH); gas.transport_model = 'high-pressure-Chung'
gas.TPX = TIN, P, FUEL; rho_f = gas.density
gas.TPX = TIN, P, OXID; rho_ox = gas.density
# burned-gas thermal diffusivity for flame-thickness / width estimate
gas.set_equivalence_ratio(1.0, FUEL, OXID); gas.TP = TIN, P; gas.equilibrate('HP')
Tad = gas.T; alpha_b = gas.thermal_conductivity / (gas.density * gas.cp_mass)
a_low = min(A_TARGETS)
delta_est = np.sqrt(2 * alpha_b / a_low)
width = 8.0 * delta_est
log(f"=== P={P_ATM}atm  rho_ox={rho_ox:.3f} rho_f={rho_f:.3f}  Tad={Tad:.0f}K "
    f"alpha_b={alpha_b:.2e}  delta_est(a={a_low:g})={delta_est*1e3:.4f}mm  width={width*1e3:.3f}mm ===")

def mdots(a):
    v_tot = a * width / 2.0
    v_f = v_tot / (1.0 + np.sqrt(rho_f / rho_ox))
    v_o = v_tot - v_f
    return rho_f * v_f, rho_ox * v_o

def fwhm(x, y, base):
    """Full width at half max of (y-base)."""
    yy = y - base
    pk = yy.max()
    if pk <= 0: return float('nan')
    half = base + 0.5 * pk
    idx = np.where(y >= half)[0]
    if len(idx) < 2: return float('nan')
    # linear-interp crossings
    iL, iR = idx[0], idx[-1]
    xL = x[iL] if iL == 0 else np.interp(half, [y[iL-1], y[iL]], [x[iL-1], x[iL]])
    xR = x[iR] if iR == len(x)-1 else np.interp(half, [y[iR], y[iR+1]], [x[iR], x[iR+1]])
    return xR - xL

f = ct.CounterflowDiffusionFlame(gas, width=width)
f.transport_model = 'high-pressure-Chung'
f.P = P
f.fuel_inlet.X = FUEL; f.fuel_inlet.T = TIN
f.oxidizer_inlet.X = OXID; f.oxidizer_inlet.T = TIN
mf, mo = mdots(a_low); f.fuel_inlet.mdot = mf; f.oxidizer_inlet.mdot = mo
f.set_refine_criteria(ratio=3.0, slope=0.18, curve=0.30, prune=0.04)
f.set_initial_guess()

results = []
for k, a in enumerate(A_TARGETS):
    mf, mo = mdots(a); f.fuel_inlet.mdot = mf; f.oxidizer_inlet.mdot = mo
    t0 = time.time()
    try:
        f.solve(loglevel=1, auto=(k == 0))
    except Exception as e:
        log(f"  a={a:g}: SOLVE FAILED ({e}) -> extinction near here"); break
    Tmax = float(f.T.max())
    if Tmax < 1500.0:
        log(f"  a={a:g}: EXTINGUISHED (Tmax={Tmax:.0f}K)"); break
    x = f.grid.copy(); T = f.T.copy(); hrr = f.heat_release_rate.copy()
    Zmix = np.array([0.0]*len(x))
    try: Zmix = f.mixture_fraction('Bilger')
    except Exception: pass
    dT = fwhm(x, T, TIN); dQ = fwhm(x, hrr, 0.0)
    q_int = float(np.trapz(hrr, x))
    sr = {}
    for d in ('max', 'mean', 'stoichiometric', 'potential_flow_fuel', 'potential_flow_oxidizer'):
        try: sr[d] = float(f.strain_rate(d))
        except Exception: sr[d] = float('nan')
    U_f = mf / rho_f; U_ox = mo / rho_ox          # inlet velocities [m/s]
    rec = dict(a_set=a, Tmax=Tmax, delta_T_mm=float(dT*1e3), delta_hrr_mm=float(dQ*1e3),
               q_int=q_int, npts=len(x), strain=sr, U_f=float(U_f), U_ox=float(U_ox),
               width_mm=float(width*1e3), dt=time.time()-t0)
    results.append(rec)
    log(f"  a={a:g}: Tmax={Tmax:.0f}K  deltaT={dT*1e3:.4f}mm  q_int={q_int:.3e}  "
        f"sr_max={sr['max']:.0f} sr_mean={sr['mean']:.0f}  npts={len(x)} ({rec['dt']:.0f}s)")
    np.savez(f'data/rf_cf_P{P_ATM:g}_a{a:g}.npz',
             x=x, T=T, Y=f.Y, hrr=hrr, Z=Zmix, P=P, a_set=a, Tmax=Tmax,
             delta_T_mm=dT*1e3, delta_hrr_mm=dQ*1e3, q_int=q_int,
             U_ox=U_ox, U_f=U_f, width_mm=width*1e3,
             sr_max=sr['max'], sr_mean=sr['mean'],
             sr_pf_ox=sr['potential_flow_oxidizer'], sr_pf_fuel=sr['potential_flow_fuel'],
             species=np.array(gas.species_names))
    # per-(P,a) summary so parallel single-strain runs don't clobber each other
    with open(f'data/rf_cf_P{P_ATM:g}_a{a:g}.json', 'w') as fp:
        json.dump(rec, fp, indent=2)

log(f"\n=== SUMMARY P={P_ATM}atm ===")
for r in results:
    log(f"  a={r['a_set']:g}: Tmax={r['Tmax']:.0f}K deltaT={r['delta_T_mm']:.4f}mm q={r['q_int']:.3e} "
        f"pf_ox={r['strain']['potential_flow_oxidizer']:.0f}")
log(f"done P={P_ATM}atm ({len(results)} strains reached)")
