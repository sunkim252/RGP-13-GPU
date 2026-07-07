"""Fig19 from real-fluid (SRK + high-pressure-Chung) narrow-domain flamelets.
Per pressure, collect burning flamelets, interpolate T_max / delta(FWHM) / q_dot
to the physical strain pf_ox = A_TARGET (Wang a=1000 == imposed oxidizer strain),
and plot vs pressure. Overlay dual-gas (MA structure) for the contrast.

Usage: python make_rf_fig19.py [a_target=1000]
"""
import sys, glob, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

A_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1000.0
PS = [1, 10, 25, 50, 100, 150]   # 200 atm excluded (direct high-P solve intractable)
DATA = 'data'

# --- Wang et al. (2015) "Counterflow Diffusion Flames of Oxygen and N-Alkanes",
#     Fig 19, 800 K inlet, a=1000/s, OXYGEN / n-HEPTANE.  Visually digitized
#     (±5-10%). NOTE fuel = n-heptane (C7H16); our flamelets = kerosene surrogate.
WANG_P     = [1, 10, 25, 50, 100, 150, 200]
WANG_TMAX  = [3080, 3440, 3550, 3650, 3740, 3810, 3860]      # K
WANG_DELTA = [1.8, 0.55, 0.38, 0.27, 0.18, 0.15, 0.13]        # mm
WANG_Q_M2  = [q * 1e4 for q in [250, 850, 1400, 2000, 2800, 3700, 4300]]  # W/cm2 -> W/m2

def interp_to(xs, ys, x0):
    """Interpolate ys(xs) at x0; clamp to nearest endpoint if x0 out of range (flagged)."""
    xs = np.asarray(xs); ys = np.asarray(ys)
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    if x0 < xs[0] or x0 > xs[-1]:
        j = int(np.argmin(np.abs(xs - x0)))
        return float(ys[j]), False        # nearest endpoint, flagged as not-bracketed
    return float(np.interp(x0, xs, ys)), True

rf = {}   # P -> dict(Tmax, delta, q, n, extrap, pfmax)
for P in PS:
    rows = []
    for fp in sorted(glob.glob(f'{DATA}/rf_cf_P{P}_a*.npz')):
        d = np.load(fp, allow_pickle=True)
        if 'sr_pf_ox' not in d.files: continue   # skip stale npz from earlier tests
        Tm = float(d['Tmax'])
        if Tm < 1500: continue            # extinguished
        rows.append((float(d['sr_pf_ox']), Tm, float(d['delta_T_mm']), float(d['q_int'])))
    if not rows:
        continue
    rows.sort()
    pf = [r[0] for r in rows]
    Tmax, ok1 = interp_to(pf, [r[1] for r in rows], A_TARGET)
    dlt, _    = interp_to(pf, [r[2] for r in rows], A_TARGET)
    q, _      = interp_to(pf, [r[3] for r in rows], A_TARGET)
    rf[P] = dict(Tmax=Tmax, delta=dlt, q=q, n=len(rows), inrange=ok1,
                 pfmin=min(pf), pfmax=max(pf))
    print(f"P={P:4g}atm: {len(rows)} burning flamelets, pf_ox in [{min(pf):.0f},{max(pf):.0f}]  "
          f"@pf={A_TARGET:g}: Tmax={Tmax:.0f}K delta={dlt:.4f}mm q={q:.3e} "
          f"{'' if ok1 else '(EXTRAPOLATED - pf_ox range did not reach target)'}")

Pg = sorted(rf.keys())
if not Pg:
    print("no burning flamelets found yet"); sys.exit(0)
Tmax = [rf[P]['Tmax'] for P in Pg]
delta = [rf[P]['delta'] for P in Pg]
q = [rf[P]['q'] for P in Pg]

br = [rf[P]['inrange'] for P in Pg]          # True = clean bracket, False = nearest-point
def mark(axis, ys, color, m):
    axis.plot(Pg, ys, '-', color=color, lw=2, zorder=1)
    for P, y, b in zip(Pg, ys, br):
        axis.plot([P], [y], m, color=color, mfc=(color if b else 'white'),
                  mec=color, ms=8, mew=1.6, zorder=2)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
mark(ax[0], Tmax, 'C3', 'o'); ax[0].set_ylabel('$T_{max}$ [K]')
mark(ax[1], delta, 'C0', 's'); ax[1].set_ylabel('flame thickness $\\delta$ (FWHM) [mm]')
mark(ax[2], q, 'C2', '^'); ax[2].set_ylabel('$\\dot q$ (integrated HRR) [W/m$^2$]')
# Wang et al. 2015 reference (n-heptane, 800K, a=1000) — black open diamonds
ax[0].plot(WANG_P, WANG_TMAX, 'D', mfc='none', mec='k', ms=7, mew=1.3, label='Wang 2015 (n-heptane)')
ax[1].plot(WANG_P, WANG_DELTA, 'D', mfc='none', mec='k', ms=7, mew=1.3, label='Wang 2015 (n-heptane)')
ax[2].plot(WANG_P, WANG_Q_M2, 'D', mfc='none', mec='k', ms=7, mew=1.3, label='Wang 2015 (n-heptane)')
ax[0].plot([], [], 'o-', color='C3', label='this work (kerosene surr.)')  # legend proxy
for a in ax:
    a.set_xlabel('pressure [atm]'); a.set_xscale('log'); a.grid(True, which='both', ls=':', alpha=0.4)
    a.legend(fontsize=8, loc='best')
ax[2].set_yscale('log')
fig.suptitle(f'Fig19 @ a=1000/s : real-fluid flamelet (this work, kerosene surrogate) vs Wang 2015 (n-heptane)\n'
             f'filled=bracket-interp, open=nearest-pf_ox, black ◇=Wang digitized', fontsize=11)
fig.tight_layout()
out = f'{DATA}/rf_fig19_a{A_TARGET:g}.png'
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
json.dump({str(P): rf[P] for P in Pg}, open(f'{DATA}/rf_fig19_a{A_TARGET:g}.json', 'w'), indent=2)
print(f"saved {DATA}/rf_fig19_a{A_TARGET:g}.json")
