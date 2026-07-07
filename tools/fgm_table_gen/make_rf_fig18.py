"""Fig18 — 10 atm flame structure (T + species mole fractions vs axial location)
from the real-fluid (SRK + high-pressure-Chung) flamelet nearest pf_ox=1000.
"""
import sys, glob
import numpy as np, cantera as ct
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P_ATM = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
PF_TARGET = 1000.0
MECH = 'data/wang2011_srk_v32.yaml'

# pick the flamelet at this pressure whose pf_ox is closest to the target
best = None
for f in glob.glob(f'data/rf_cf_P{P_ATM:g}_a*.npz'):
    d = np.load(f, allow_pickle=True)
    if 'sr_pf_ox' not in d.files: continue
    pf = float(d['sr_pf_ox'])
    if best is None or abs(pf - PF_TARGET) < abs(best[1] - PF_TARGET):
        best = (f, pf, d)
fpath, pf, d = best
print(f"using {fpath}  pf_ox={pf:.0f}  Tmax={float(d['Tmax']):.0f}K")

x = d['x'] * 1e3                     # mm
T = d['T']; Y = d['Y']; sp = list(d['species'])
W = ct.Solution(MECH).molecular_weights
# mass -> mole fractions
Xmole = (Y / W[:, None]); Xmole /= Xmole.sum(axis=0, keepdims=True)
xc = x - x[np.argmax(T)]            # center on peak-T location

# --- Wang et al. 2015 Fig18 (n-heptane, 10 atm, 800K, a=1000) digitized (approx).
#     axial location [cm]; T-peak at ~0.945 cm -> shift to mm-from-peak.
WX0 = 0.945
def wmm(cm): return [(c - WX0) * 10.0 for c in cm]
WANG18 = {
 'T':    (wmm([0.88,0.90,0.91,0.92,0.93,0.94,0.945,0.95,0.955,0.96,0.97,0.98,1.00]),
          [900,1150,1450,1850,2400,3050,3380,3350,3150,2800,2150,1600,950]),
 'O2':   (wmm([0.86,0.90,0.93,0.94,0.95,0.96,0.97,0.98,1.00]),
          [0.02,0.03,0.08,0.18,0.40,0.65,0.82,0.92,0.98]),
 'NC10H22':(wmm([0.85,0.88,0.90,0.91,0.92,0.925,0.93,0.94,0.95]),   # Wang fuel = n-C7H16
          [0.97,0.92,0.78,0.60,0.38,0.25,0.12,0.03,0.0]),
 'CO':   (wmm([0.89,0.90,0.91,0.92,0.925,0.93,0.935,0.94,0.95,0.96,0.97]),
          [0.05,0.12,0.25,0.40,0.48,0.50,0.46,0.38,0.22,0.10,0.03]),
 'H2O':  (wmm([0.91,0.92,0.93,0.94,0.95,0.955,0.96,0.97,0.98]),
          [0.03,0.08,0.15,0.23,0.28,0.27,0.22,0.13,0.05]),
 'CO2':  (wmm([0.92,0.93,0.94,0.95,0.955,0.96,0.97,0.98]),
          [0.02,0.05,0.10,0.14,0.15,0.13,0.08,0.03]),
}

SPECIES = ['O2', 'NC10H22', 'CO2', 'H2O', 'CO', 'H2', 'OH']
colors = {}
fig, ax = plt.subplots(figsize=(9.5, 6.0))
axT = ax.twinx()
l, = axT.plot(xc, T, 'k-', lw=2.2, label='T (this work)')
axT.set_ylabel('T [K]', color='k')
for s in SPECIES:
    if s in sp:
        ln, = ax.plot(xc, Xmole[sp.index(s)], lw=1.8, label=s)
        colors[s] = ln.get_color()
# Wang overlay: open markers in matching colors (T on axT in black)
axT.plot(WANG18['T'][0], WANG18['T'][1], 'kD', mfc='none', ms=6, mew=1.2)
for s in ('O2', 'NC10H22', 'CO', 'H2O', 'CO2'):
    if s in colors:
        ax.plot(WANG18[s][0], WANG18[s][1], 'D', mfc='none', mec=colors[s], ms=6, mew=1.2)
ax.plot([], [], 'kD', mfc='none', label='Wang 2015 (n-heptane, ◇)')
ax.set_xlabel('axial location (centered on $T_{max}$) [mm]')
ax.set_ylabel('mole fraction')
ax.set_title(f'Fig18 — {P_ATM:g} atm flame structure: this work (kerosene surr., lines) vs '
             f'Wang 2015 (n-heptane, ◇)\npf_ox={pf:.0f}/s, 800K, a=1000')
ax.legend(loc='upper left', fontsize=8, ncol=2)
ax.grid(True, ls=':', alpha=0.4)
# zoom around the flame
w = 6.0 * float(d['delta_T_mm'])
ax.set_xlim(-w, w); axT.set_ylim(700, 3700)
fig.tight_layout()
out = f'data/rf_fig18_P{P_ATM:g}.png'
fig.savefig(out, dpi=130)
print(f"saved {out}")
