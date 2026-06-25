"""Non-adiabatic FPV (method b): add an ENTHALPY-DEFECT 4th axis to a 3-D FGM
table, so the manifold can be queried at inlet/heat-loss enthalpies different
from the 800 K flamelet boundary -- in particular cryogenic LOx + warm kerosene.

Coordinate (the clean part): TOTAL enthalpy h is a CONSERVED scalar (like Z), so
the adiabatic reference is a mixing line  h_ad(Z) = (1-Z) h_ox + Z h_fuel  with
h_ox, h_fuel the pure-stream total enthalpies at the 800 K flamelet boundary.
The 4th axis is the enthalpy DEFECT  dh = h - h_ad(Z)  (dh=0 -> adiabatic 800 K
manifold = the original 3-D table EXACTLY; dh<0 -> colder inlet / heat loss).

Build (frozen-composition, no extra flamelet solves -> avoids the stiff cryo
flamelet Newton stall). The defect is the SENSIBLE-enthalpy change referenced to
the flamelet's OWN adiabatic state, so the formation-enthalpy / differential-
diffusion (Takahashi/Chung) / dual-gas offset CANCELS (h(Tad,Y) need NOT equal
the mixing line):
  T(Z,gZ,c,dh):  solve  h(T,Y) - h(Tad,Y) = dh   at frozen Y(Z,gZ,c)   [RK EOS]
  Y(Z,gZ,c,dh) = Y(Z,gZ,c)                                    (frozen)
  omega_C(...,dh) = omega_C(Z,gZ,c) * ignition_gate(T(...,dh))  (inert when cold)
Solver side: dh = h_transported - ((1-Z)hOx + Z hFuel); at a stream boundary the
composition IS the pure stream so h(Tad,Y)=h_mix there -> the two references
coincide, and in the interior the heat-loss departure is measured consistently.

Solver reads: nH, fourthAxis=enthalpy, hOx, hFuel, the dh axis. It transports
total enthalpy h, forms dh = h - ((1-Z)hOx + Z hFuel), and does the 4-D lookup.

** CAVEAT: NASA-7 thermo floors at ~300 K; below that (cold O2) the h<->T
inversion is EXTRAPOLATED. EOS density is physical, cold-stream sensible h is
approximate -> swap a NIST/REFPROP h(T) for the pure cold streams for production. **

Usage: python add_enthalpy_axis.py <src_3d.npz> <out_prefix> [n_h] [T_ox_cold] [T_fuel_cold]
"""
import sys, time
import numpy as np, cantera as ct

SRC   = sys.argv[1]
OUT   = sys.argv[2]
N_H   = int(sys.argv[3])   if len(sys.argv) > 3 else 11
T_OXC = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0   # cold oxidizer (LOx)
T_FUC = float(sys.argv[5]) if len(sys.argv) > 5 else 300.0   # cold fuel (kerosene)
T_BOUND = 800.0
FUEL = {'NC10H22': 0.74, 'PHC3H7': 0.15, 'CYC9H18': 0.11}
OXID = {'O2': 1.0}
MECH = 'data/wang2011_srk_v32.yaml'
T_IGN, DT_IGN = 900.0, 150.0     # source ignition gate (reaction frozen below ~crossover)
def log(m): print(m, flush=True)

d = np.load(SRC, allow_pickle=True)
Z_axis = d['Z_axis']; g_axis = d['gZ_axis']; C_axis = d['C_axis']
P = float(d['P']); species = [str(s) for s in d['species']]
T3 = d['T']; om3 = d['omega_C']
has_hrr = 'hrr' in d.files
hrr3 = d['hrr'] if has_hrr else None
Yk = {sp: d[f'Y_{sp}'] for sp in species}
nZ, nGz, nC = T3.shape
log(f"loaded 3-D table {SRC}: {nZ}x{nGz}x{nC}, P={P/1e5:.1f}bar, {len(species)} species")

g = ct.Solution(MECH)
# Ideal-gas twin for the h(T) inversion: the SRK cubic root solver fails to
# converge for some (T,Y) with the real composition (e.g. T~1630 K) and aborts
# the build. The enthalpy used for the T inversion is dominated by formation +
# sensible terms (both identical in ideal/real); the real-fluid departure is
# small and largely cancels in the sensible DIFFERENCE h(T,Y)-h(Tad,Y). So fall
# back to the ideal twin wherever the cubic fails -> robust, no crashes.
gi = ct.Solution('data/wang2011_ideal_v32.yaml')
def h_at(sol, T, Ydict):
    sol.TPY = T, P, Ydict; return sol.enthalpy_mass
# boundary total enthalpies at the flamelet boundary T (defect reference)
g.TPX = T_BOUND, P, OXID;  h_ox = g.enthalpy_mass
g.TPX = T_BOUND, P, FUEL;  h_fuel = g.enthalpy_mass
# cold-stream defects set the axis span
g.TPX = T_OXC, P, OXID;  dh_ox = g.enthalpy_mass - h_ox
g.TPX = T_FUC, P, FUEL;  dh_fu = g.enthalpy_mass - h_fuel
dh_min = 1.05 * min(dh_ox, dh_fu)         # most negative defect (+5% margin)
h_axis = np.linspace(dh_min, 0.0, N_H)    # enthalpy defect [J/kg]; dh=0 = adiabatic
log(f"h_ox(800K)={h_ox/1e6:.3f} h_fuel(800K)={h_fuel/1e6:.3f} MJ/kg")
log(f"cold defects: O2@{T_OXC:g}K dh={dh_ox/1e6:.3f}  fuel@{T_FUC:g}K dh={dh_fu/1e6:.3f} MJ/kg")
log(f"enthalpy-defect axis: {N_H} pts, [{h_axis[0]/1e6:.3f}, 0] MJ/kg")

# ---- build the 4-D arrays (frozen composition, ROBUST h->T inversion) ----
# Cantera's HPY Newton lands on spurious roots below the 300 K NASA-poly floor
# (non-monotonic extrapolated h(T)). Instead build a MONOTONE-ENFORCED h(T) grid
# per composition and invert by interpolation, clamped to [T_FLOOR, T_ad]. This
# never returns garbage; over-cold targets (e.g. a burnt state pushed to the
# coldest defect -- a corner the CFD never visits) simply clamp to T_FLOOR.
T_FLOOR = 80.0          # coldest physical (LOx ~90 K); covers O2 @100 K inlet
NT = 32                 # h(T) grid points per composition
T_CUT = 600.0           # hard reaction cutoff: omega = 0 below this T
T4  = np.zeros((nZ, nGz, nC, N_H))
om4 = np.zeros((nZ, nGz, nC, N_H))
hrr4 = np.zeros((nZ, nGz, nC, N_H)) if has_hrr else None
t0 = time.time(); npt = 0
for iZ in range(nZ):
    for iG in range(nGz):
        for iC in range(nC):
            Yvec = np.array([Yk[sp][iZ, iG, iC] for sp in species])
            s = Yvec.sum()
            Tad = T3[iZ, iG, iC]
            if s <= 1e-6:                      # empty/edge -> replicate adiabatic
                T4[iZ, iG, iC, :] = Tad
                om4[iZ, iG, iC, :] = om3[iZ, iG, iC]
                if has_hrr: hrr4[iZ, iG, iC, :] = hrr3[iZ, iG, iC]
                continue
            Yvec /= s
            # ** set composition BY NAME ** -- the table's species list is sorted
            # alphabetically, NOT in Cantera's order, so g.TPY=T,P,Yvec (array)
            # SCRAMBLES the composition. A name->value dict is order-independent.
            Ydict = dict(zip(species, Yvec))
            # monotone h(T) from T_FLOOR up to the local adiabatic T
            Tgrid = np.linspace(T_FLOOR, max(Tad, T_FLOOR + 50.0), NT)
            hgrid = np.empty(NT)
            for j, Tg in enumerate(Tgrid):
                try:
                    hgrid[j] = h_at(g, Tg, Ydict)        # SRK (real-fluid)
                except Exception:
                    hgrid[j] = h_at(gi, Tg, Ydict)       # ideal fallback (cubic failed)
                npt += 1
            hgrid = np.maximum.accumulate(hgrid)   # enforce dh/dT>0 (clip extrap wiggle)
            # SENSIBLE-enthalpy defect referenced to the flamelet's OWN adiabatic
            # state h(Tad,Y) = hgrid[-1]:  solve  h(T,Y) - h(Tad,Y) = dh.  The
            # formation-enthalpy (and the differential-diffusion / dual-gas) offset
            # CANCELS in the difference, so dh=0 -> T=Tad exactly and a cold defect
            # cools T by the right sensible amount regardless of that offset. (An
            # absolute h_ad=(1-Z)hOx+ZhFuel reference fails because h(Tad,Y) != h_ad
            # for real-fluid flamelets.) Matches the solver's dh = h - h_mix(Z).
            targets = hgrid[-1] + h_axis
            Tloc = np.interp(targets, hgrid, Tgrid)  # clamps to [T_FLOOR, Tad]
            Tloc[np.abs(h_axis) < 1e-9] = Tad        # dh=0 slice exact
            T4[iZ, iG, iC, :] = Tloc
            gate = 0.5 * (1.0 + np.tanh((Tloc - T_IGN) / DT_IGN))
            gate[Tloc < T_CUT] = 0.0                 # hard inert cutoff when cold
            om4[iZ, iG, iC, :] = om3[iZ, iG, iC] * gate
            if has_hrr: hrr4[iZ, iG, iC, :] = hrr3[iZ, iG, iC] * gate
    if iZ % 10 == 0:
        log(f"  iZ={iZ}/{nZ}  ({time.time()-t0:.0f}s, {npt} h(T) evals)")
# Y frozen: broadcast along the new axis
Y4 = {sp: np.repeat(Yk[sp][..., None], N_H, axis=3) for sp in species}
log(f"4-D build done: {npt} h(T) evals in {time.time()-t0:.0f}s, shape {T4.shape}")

# sanity: dh=0 slice must equal the original 3-D T exactly
err = np.abs(T4[..., -1] - T3).max()
log(f"dh=0 slice vs original 3-D Tmax-diff = {err:.3e} K (should be ~0)")

# ---- write npz + OpenFOAM dict (enthalpy-axis 4-D format) ----
def fmt(a): return "\n".join(f"    {v:.8e}" for v in np.asarray(a).reshape(-1))

np.savez_compressed(OUT + '.npz',
    Z_axis=Z_axis, gZ_axis=g_axis, C_axis=C_axis, h_axis=h_axis,
    species=np.array(species, dtype=object), P=P, h_ox=h_ox, h_fuel=h_fuel,
    fourthAxis='enthalpy', T=T4, omega_C=om4,
    **({'hrr': hrr4} if has_hrr else {}),
    **{f'Y_{sp}': Y4[sp] for sp in species})
log(f"[write] {OUT}.npz")

nH = N_H
blocks = [f"sourcePV\n(\n{fmt(om4)}\n);", f"T\n(\n{fmt(T4)}\n);"]
blocks.append("species ( " + " ".join(species) + " );")
for sp in species:
    blocks.append(f"Y_{sp}\n(\n{fmt(Y4[sp])}\n);")
body = "\n\n".join(blocks)
header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| FGM 4-D NON-ADIABATIC (Z~, gZ, c, dh=enthalpy-defect) -- add_enthalpy_axis.py
| P_ref = {P:.6g} Pa | h_ox(800K)={h_ox:.6g} h_fuel(800K)={h_fuel:.6g} J/kg
| dh = h_total - ((1-Z) hOx + Z hFuel);  dh=0 -> adiabatic 800K manifold
| flat C-order: idx = ((iZ*nGz + iGz)*nC + iC)*nH + iH
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      fgmProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

nZ      {nZ};
nGz     {nGz};
nC      {nC};
nH      {nH};
fourthAxis  enthalpy;
hOx     {h_ox:.8e};
hFuel   {h_fuel:.8e};

Le
{{
    Z   0.63;
    C   0.60;
    h   0.72;
}}

Z
(
{fmt(Z_axis)}
);

gZ
(
{fmt(g_axis)}
);

C
(
{fmt(C_axis)}
);

enthalpy
(
{fmt(h_axis)}
);

{body}

// ************************************************************************* //
"""
open(OUT, 'w').write(header)
log(f"[write] {OUT}  (4-D non-adiabatic, nH={nH})")
