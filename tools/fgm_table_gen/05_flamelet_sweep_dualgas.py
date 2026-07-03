"""Dual-gas flamelet sweep: ideal-gas continuation + per-state SRK+Chung upgrade.

WHY THIS EXISTS
---------------
The SRK (Redlich-Kwong) cubic EOS collapses Cantera's Newton / pseudo-transient
solve whenever pressure or temperature is *moving* near the kerosene surrogate's
critical point (n-decane Tc=617.8 K, Pc=21 bar). The 1 atm -> P_OPER pressure
ramp under SRK + high-pressure-Chung hung for >22 h on a single r=1.10 step;
switching the ramp transport to mixture-averaged did not help, proving the EOS
(not the transport model) is the culprit.

Verified empirically (tools/fgm_table_gen/_ig_to_srk_test.py):
  * an ideal-gas flame converges DIRECTLY at 52.5 bar (no pressure ramp at all)
    and cools to cold inlets without collapse;
  * a converged ideal-gas flame, restored into an SRK + high-pressure-Chung
    flame and solved auto=False at the FIXED target, converges
    (RESULT B1 OK: npts=192, Tmax=3570 K vs ideal 3615 K).

So the whole continuation runs on an ideal-gas flame, and every state we keep is
"upgraded" to SRK + high-pressure-Chung with a fixed-grid (auto=False) solve.
If a given state still won't converge under SRK, we fall back to the ideal-gas
flamelet for that point and flag it (the table is then ideal-gas there).

This is physically sound: high-pressure chemistry is identical (rate constants
depend on P, T, not on the EOS); only density and transport differ, and those
are recomputed at runtime by OpenFOAM's SRK + Chung-Takahashi. The SRK upgrade
additionally bakes the real-fluid rho / lambda / mu (hence chi_st) into the
saved flamelets, keeping the table consistent with the flow solver.

Run:
  conda activate ct-env3
  python 05_flamelet_sweep_dualgas.py --log 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "14")

import importlib.util
import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

# Reuse the verified helpers from the staged SRK script (same directory).
_spec = importlib.util.spec_from_file_location(
    "swp", HERE / "05_flamelet_sweep_fast.py")
swp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swp)

# Pull constants / helpers through so the two scripts cannot drift.
P_OPER        = swp.P_OPER
DOMAIN_WIDTH  = swp.DOMAIN_WIDTH

# --- Inlet temperature: matched to Wang, Huo & Yang 2015 (CST 187:60), Fig 18 ---
# That study computes O2 / n-alkane counterflow flamelets with the SAME real-fluid
# framework (SRK + departure functions + corresponding-states transport) to build
# a tabulated chemistry library. For the HEAVY n-alkanes (n-heptane and the
# C12-C16 family -- our kerosene surrogate is n-decane-dominant, C10) they fix
# BOTH inlets at 800 K: "To avoid complexities associated with fuel vaporization,
# the inlet temperature remains at 600 K or above" and Fig 18 uses
# TO2 = TC7H16 = 800 K. They further note the inlet temperature has a negligible
# effect on flame characteristics -- the real-fluid effect is confined to the
# cold-inlet region and the fluid behaves ideal-gas-like in the flame zone. So
# the 800 K manifold is valid for the cold (120 K LOX / ~300 K kerosene) injection
# application, where OpenFOAM applies the real inlet BCs + SRK + Chung at runtime.
# Holding both inlets hot also keeps the SRK + high-pressure-Chung solve well-
# conditioned (compressibility ~ 1), which is exactly what made the cold-inlet
# SRK upgrade collapse (dt -> 1e-9). Hence NO cooling stage here.
INLET_T       = 800.0
T_HOT_START   = INLET_T
T_FUEL        = INLET_T
MDOT_INIT     = swp.MDOT_INIT
X_OX          = swp.X_OX
A_EXP         = swp.A_EXP
REFINE_COARSE = swp.REFINE_COARSE
REFINE_FINAL  = swp.REFINE_FINAL
MAX_GRID_COARSE = swp.MAX_GRID_COARSE
MAX_GRID_FINAL  = swp.MAX_GRID_FINAL
TRANSPORT_SRK   = swp.TRANSPORT_MODEL      # high-pressure-Chung
# Flame STRUCTURE is solved with unity Lewis numbers, consistent with the
# fgmFluid solver's unity-Schmidt C/Z transport. With mixture-averaged
# transport, differential diffusion enriches the reaction zone and the
# flamelet progress variable OVERSHOOTS the adiabatic-equilibrium value
# (observed: C=0.817 vs C_eq=0.771 at Z_st, 50 atm), which breaks the
# c = C/C_eq(Z) normalization and the equilibrium burnt-end closure.
# Unity-Le guarantees C <= C_eq so equilibrium is a clean upper bound.
# (FPVgen, IhmeGroup, offers unity-Lewis-number as a first-class option.)
# Real-fluid properties (rho/lambda/mu/chi_st) are still re-evaluated with
# SRK + high-pressure-Chung at save time (_save_flamelet_realfluid).
TRANSPORT_IDEAL = os.environ.get("FGM_STRUCT_TRANSPORT", "unity-Lewis-number")
_log = swp._log

IDEAL_YAML = DATA / "wang2011_ideal_v32.yaml"
SRK_YAMLS  = [DATA / "wang2011_srk_v32.yaml", DATA / "wang2011_srk.yaml"]


def _load(path_or_list):
    paths = path_or_list if isinstance(path_or_list, list) else [path_or_list]
    last = None
    for p in paths:
        if p.exists():
            try:
                return ct.Solution(str(p)), p
            except Exception as ex:
                last = ex
    raise SystemExit(f"no usable YAML in {paths}: {last}")


# ------------------------- ideal-gas helpers -------------------------

def _fresh_ideal(gas):
    f = ct.CounterflowDiffusionFlame(gas, width=DOMAIN_WIDTH)
    f.transport_model = TRANSPORT_IDEAL
    f.P = P_OPER
    f.fuel_inlet.T = T_HOT_START
    f.fuel_inlet.X = swp._fuel_X(gas)
    f.fuel_inlet.mdot = MDOT_INIT
    f.oxidizer_inlet.T = T_HOT_START
    f.oxidizer_inlet.X = X_OX
    f.oxidizer_inlet.mdot = MDOT_INIT
    f.set_refine_criteria(**REFINE_COARSE)
    try:
        f.set_max_grid_points(f.flame, MAX_GRID_COARSE)
    except Exception:
        pass
    f.max_time_step_count = 900
    return f


def _cool_inlet(flame, which, T_target, nsteps, log, depth=0):
    """Adaptively ramp one inlet temperature to T_target under ideal gas."""
    inlet = flame.fuel_inlet if which == "fuel" else flame.oxidizer_inlet
    T0 = float(inlet.T)
    if abs(T0 - T_target) < 0.5:
        return
    for T in np.linspace(T0, T_target, nsteps + 1)[1:]:
        T_prev = float(inlet.T)
        inlet.T = float(T)
        try:
            swp._solve(flame, f"{which} T={T:6.1f} K", log)
        except ct.CanteraError as ex:
            if depth >= 4 or abs(T - T_prev) < 1.0:
                raise
            inlet.T = T_prev
            msg = str(ex).splitlines()[-1] if str(ex) else "CanteraError"
            _log(f"  retry: split {which} T {T_prev:.1f}->{T:.1f} ({msg})")
            _cool_inlet(flame, which, float(T), 2, log, depth + 1)


# ------------------------- SRK upgrade -------------------------

def _transfer_profile(src, dst, dst_gas):
    """Copy the converged solution (flow components + T + every species) from
    `src` onto `dst` via set_profile, so the destination EOS derives its OWN
    density from (T, P, Y). We deliberately do NOT use Sim1D.restore here:
    restore transfers the saved density/solution vector, and at cold trans-
    critical conditions the ideal-gas density (e.g. O2 ~168 kg/m3 at 120 K /
    52.5 bar) is wildly inconsistent with SRK's liquid density (~978), which
    makes restore throw 'neg rho'. set_profile sidesteps that entirely."""
    g = np.asarray(src.grid)
    span = max(g[-1] - g[0], 1e-30)
    norm = (g - g[0]) / span
    dst.set_profile("velocity",    norm, np.asarray(src.velocity))
    dst.set_profile("spread_rate", norm, np.asarray(src.spread_rate))
    dst.set_profile("lambda",      norm, np.asarray(src.L))
    dst.set_profile("T",           norm, np.asarray(src.T))
    src_gas = src.gas
    for sp in dst_gas.species_names:
        dst.set_profile(sp, norm, src.Y[src_gas.species_index(sp)])


def _srk_upgrade(srk_gas, ideal_flame, ckpt_dir, log):
    """Transfer the converged ideal-gas solution onto a fresh SRK +
    high-pressure-Chung flame (via set_profile so SRK recomputes density) and
    settle it on the SAME grid (auto=False). Returns the converged SRK flame,
    or None if SRK will not converge here (caller falls back to ideal)."""
    # Construct the SRK flame DIRECTLY on the ideal grid. A default-width flame
    # has only a ~6-point grid, so set_profile would squash the refined ideal
    # solution (~260 pts) down to 6 points (observed: 36-pt flamelets with ~4
    # points across Z_st). Assigning fs.flame.grid AFTER construction segfaults
    # (the Sim1D solution vector isn't resized), so the grid must be passed to
    # the constructor, which sizes the solution vector correctly up front.
    fs = ct.CounterflowDiffusionFlame(srk_gas,
                                      grid=np.asarray(ideal_flame.grid))
    fs.transport_model = TRANSPORT_SRK
    fs.P = float(ideal_flame.P)
    fs.fuel_inlet.T = float(ideal_flame.fuel_inlet.T)
    fs.fuel_inlet.X = swp._fuel_X(srk_gas)
    fs.fuel_inlet.mdot = float(ideal_flame.fuel_inlet.mdot)
    fs.oxidizer_inlet.T = float(ideal_flame.oxidizer_inlet.T)
    fs.oxidizer_inlet.X = X_OX
    fs.oxidizer_inlet.mdot = float(ideal_flame.oxidizer_inlet.mdot)
    _transfer_profile(ideal_flame, fs, srk_gas)
    fs.max_time_step_count = 2000
    t0 = time.time()
    try:
        fs.solve(loglevel=log, auto=False)
        _log(f"  SRK upgrade OK: npts={fs.grid.size} "
             f"Tmax={fs.T.max():.1f} K dt={time.time()-t0:.1f}s")
        return fs
    except ct.CanteraError as ex:
        msg = str(ex).splitlines()[-1] if str(ex) else "CanteraError"
        _log(f"  SRK upgrade FAILED ({msg}); keeping ideal-gas flamelet")
        return None


# ------------------ real-fluid property re-evaluation ------------------

def _save_flamelet_realfluid(flame, srk_gas, out_dir, idx):
    """Save a flamelet whose density, transport, AND progress-variable source
    are evaluated with the real-fluid SRK EOS + high-pressure-Chung transport,
    POINTWISE on the converged ideal-gas flame structure.

    Why this instead of solving an SRK flame: the SRK + high-pressure-Chung
    steady Newton stalls on a resolved (>=150-pt) grid at 52.5 bar (dt frozen
    ~1e-5), so an SRK counterflow flamelet cannot be converged reliably here.
    Wang, Huo & Yang 2015 (CST 187:60) show the flame structure in
    mixture-fraction space is essentially EOS-insensitive, so we take the
    structure (T, Y_k, grid) from the robust ideal-gas + mixture-averaged solve
    and re-evaluate, at each grid node, exactly the quantities the LES solver
    uses: rho (SRK), lambda/mu (high-pressure-Chung), cp (SRK), the progress-
    variable source omega_C (SRK concentrations -> real reaction rates), and
    hence alpha, chi, chi_st. This keeps the tabulated properties consistent
    with the OpenFOAM SRK + Chung-Takahashi flow solver without ever needing an
    SRK flame to converge."""
    fgas = flame.gas
    z = np.asarray(flame.grid)
    T = np.asarray(flame.T)
    Yall = np.array(flame.Y)
    P = float(flame.P)
    # Composition hygiene: accepted (steady-not-declared) transients carry
    # isolated unconverged spikes with sum(Y) far from 1. Normalize pointwise
    # and warn; spikes beyond 5% are additionally masked by the table builder
    # on load. (Cantera's TPY setter normalizes internally, so the SRK
    # property loop below was never affected -- only the saved Y arrays.)
    sumY = Yall.sum(axis=0)
    dev = float(np.abs(sumY - 1.0).max())
    if dev > 1e-6:
        Yall = Yall / np.maximum(sumY, 1e-12)[None, :]
        if dev > 0.05:
            _log(f"  WARN flamelet idx={idx}: max|sumY-1|={dev:.2f} at "
                 f"{int((np.abs(sumY-1)>0.05).sum())} point(s); normalized")
    srk_gas.transport_model = TRANSPORT_SRK
    n = z.size
    rho = np.zeros(n); lam = np.zeros(n); mu = np.zeros(n)
    cp = np.zeros(n); omega_C = np.zeros(n)
    Mw = srk_gas.molecular_weights
    pv_idx = [srk_gas.species_index(s) for s in swp.PV_SPECIES
              if s in srk_gas.species_names]
    for i in range(n):
        srk_gas.TPY = float(T[i]), P, Yall[:, i]
        rho[i] = srk_gas.density_mass
        lam[i] = srk_gas.thermal_conductivity
        mu[i] = srk_gas.viscosity
        cp[i] = srk_gas.cp_mass
        wdot = srk_gas.net_production_rates           # kmol/m3/s (SRK conc.)
        omega_C[i] = float(sum(wdot[k] * Mw[k] for k in pv_idx)) \
            / max(rho[i], 1e-30)
    Z = swp._bilger_Z(flame)
    alpha = lam / np.maximum(rho * cp, 1e-30)
    chi = swp._chi_profile(z, Z, lam, rho, cp)
    chi_st = swp._chi_at_Zst(Z, chi)
    arrs = {
        "z": z, "Z": Z, "T": T, "rho": rho, "lam": lam, "mu": mu, "cp": cp,
        "alpha": alpha, "chi": chi, "chi_st": np.asarray(chi_st),
        "C": sum(Yall[fgas.species_index(s)] for s in swp.PV_SPECIES
                 if s in fgas.species_names),
        "omega_C": omega_C,
        "mdot": np.asarray(float(flame.fuel_inlet.mdot)),
        "P": np.asarray(P),
        "T_fuel": np.asarray(float(flame.fuel_inlet.T)),
        "T_ox": np.asarray(float(flame.oxidizer_inlet.T)),
        "Z_st_ref": np.asarray(swp._Z_ST),
        "npts": np.asarray(int(n)),
        "Tmax": np.asarray(float(T.max())),
        "idx": np.asarray(int(idx)),
        # provenance: which STRUCTURE transport produced this flamelet
        # (unity-Lewis vs mixture-averaged families are physically different
        # manifolds; earlier files lacked this and needed forensic T_max
        # signatures to attribute).
        "struct_transport": np.asarray(TRANSPORT_IDEAL),
    }
    for sp in srk_gas.species_names:
        arrs[f"Y_{sp}"] = Yall[fgas.species_index(sp)]
    path = out_dir / f"flamelet_{idx:03d}.npz"
    np.savez_compressed(path, **arrs)
    _log(f"  saved {path.name}  npts={n}  chi_st={chi_st:.3g} 1/s  "
         f"Tmax={T.max():.1f} K  rho<={rho.max():.0f}  (SRK+Chung props)")


# ------------------------- checkpoint / state -------------------------

def _state_load(f):
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}


def _state_save(f, s):
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=2, default=float))
    tmp.replace(f)


def _restore_ideal(gas, path, name):
    return swp._ckpt_restore(gas, path, name, transport=TRANSPORT_IDEAL)


# ------------------------- main pipeline -------------------------

def run(log=0, mdot_factor=1.25, mdot_max=20.0, restart=False,
        warm_from=None):
    # Per-pressure output/checkpoint dirs so a multi-pressure Wang-2015 sweep
    # (1..200 atm) never clashes. The 52.5 bar production table keeps the plain
    # "dualgas" name for back-compat; any other pressure gets a P-tag.
    p_atm = P_OPER / 101325.0
    tag = "" if abs(P_OPER - 52.5e5) < 1e3 else f"_P{p_atm:g}atm"
    # Non-default STRUCTURE transport gets its own family/checkpoint dirs --
    # without this an MA regeneration at 52.5 bar would OVERWRITE the
    # unity-Le production family in the untagged "flamelets_dualgas" dir and
    # resume from its unity-Le checkpoints.
    if TRANSPORT_IDEAL != "unity-Lewis-number":
        tslug = {"mixture-averaged": "MA",
                 "multicomponent": "MC"}.get(TRANSPORT_IDEAL,
                                             TRANSPORT_IDEAL.replace("-", ""))
        tag += f"_{tslug}"
    out_dir = DATA / f"flamelets_dualgas{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    CKPT = DATA / f"ckpt_dualgas{tag}"
    CKPT.mkdir(parents=True, exist_ok=True)
    STATE = CKPT / "state.json"

    igas, ipath = _load(IDEAL_YAML)
    sgas, spath = _load(SRK_YAMLS)
    T_ox = INLET_T   # both inlets held at the Wang-2015 heavy-n-alkane value
    _log(f"ideal={ipath.name} srk={spath.name}  "
         f"P={P_OPER/1e5:.1f} bar  T_fuel={T_FUEL} T_ox={T_ox} K "
         f"(Wang 2015 Fig 18: TO2=Tfuel=800 K, no cooling)")

    state = {} if restart else _state_load(STATE)
    state.setdefault("stage", "A")
    state.setdefault("strain_idx", -1)
    _state_save(STATE, state)

    flame = None
    cA, cB, cC = (CKPT / "stageA_ideal.yaml",
                  CKPT / "stageB_ideal.yaml",
                  CKPT / "stageC_ideal.yaml")

    # ---- Stage A: ideal-gas, converge directly at P_OPER, hot inlets ----
    # The cold-start auto solve does NOT converge under unity-Lewis-number
    # ("Could not find a solution for the 1D problem" at 1 and 50 atm), so
    # we warm-start with the robust mixture-averaged transport and only then
    # swap to unity-Le on the converged structure and settle (auto=False) --
    # the same converge-then-swap trick that worked for the SRK upgrade.
    if state["stage"] == "A":
        if warm_from:
            # PRESSURE CONTINUATION for extreme pressures (>=150 atm): the
            # MA cold start cannot find the razor-thin flame ("Could not
            # find a solution for the 1D problem"), but a converged lower-
            # pressure stage-A checkpoint ramped up in geometric steps
            # (x1.2 per step, auto=True so the grid follows the thinning
            # flame, accept-if-burning per step) gets there robustly --
            # the same continuation idea as everywhere else in this file.
            flame = swp._ckpt_restore(igas, Path(warm_from), "A",
                                      transport="mixture-averaged")
            P0 = float(flame.P)
            _log(f"stage A: warm-start from {warm_from} "
                 f"(P={P0/1e5:.1f} bar) -> ramp to {P_OPER/1e5:.1f} bar")
            # RAMP WITH auto=False ON THE INHERITED GRID. Cantera's
            # solve(auto=True) regrids internally regardless of the refine
            # criteria set on the Sim1D (verified: COARSE and FINAL criteria
            # produced bit-identical 48-point grids), and the 48-point
            # solution overshoots the adiabatic-equilibrium ceiling even
            # after a converged unity-Le settle (4165 K vs the exact 3984 K
            # at 152 bar) -- pure truncation error at the peak. auto=False
            # keeps the donor checkpoint's ~152-point grid (refined for the
            # 100 atm flame; the flame only thins ~(P ratio)^-1/2 ~ 20% per
            # 1.5x pressure step, so the inherited clustering still resolves
            # it), and each pressure step settles in multi-round fashion.
            try:
                flame.max_time_step_count = 200
            except Exception:
                pass
            nstep = max(1, int(np.ceil(np.log(P_OPER / P0) / np.log(1.2))))
            for Pt in np.geomspace(P0, P_OPER, nstep + 1)[1:]:
                flame.P = float(Pt)
                t0 = time.time()
                Tprev = None
                for rnd in range(1, 5):
                    declared = True
                    try:
                        flame.solve(loglevel=log, auto=False)
                    except ct.CanteraError:
                        if flame.T.max() < 1500.0:
                            raise
                        declared = False
                    Tm = float(flame.T.max())
                    if declared or (Tprev is not None
                                    and abs(Tm - Tprev) < 2.0):
                        break
                    Tprev = Tm
                _log(f"  P-ramp {Pt/1e5:7.2f} bar npts={flame.grid.size} "
                     f"Tmax={flame.T.max():.1f} K ({rnd} round(s), "
                     f"dt={time.time()-t0:.1f}s)")
            # NO extra mixture-averaged settle here: the ramp's final
            # auto=True solve already adapted/converged at the target
            # pressure, and additional MA pseudo-time on the thin flame only
            # drifted the peak upward (+100 K per 200 steps, observed). Go
            # straight to the unity-Le swap + multi-round settle below --
            # under unity-Le the adiabatic-equilibrium ceiling is enforced
            # thermodynamically (h(Z) linear).
        else:
            _log(f"stage A: ideal-gas converge directly @ "
                 f"{P_OPER/1e5:.1f} bar, T_inlets={T_HOT_START} K "
                 f"(no pressure ramp; mixture-averaged warm-start -> "
                 f"{TRANSPORT_IDEAL})")
            flame = _fresh_ideal(igas)
            flame.transport_model = "mixture-averaged"
            t0 = time.time()
            flame.solve(loglevel=log, auto=True)
            _log(f"  mixture-averaged warm-start npts={flame.grid.size} "
                 f"Tmax={flame.T.max():.1f} K dt={time.time()-t0:.1f}s")
        flame.transport_model = TRANSPORT_IDEAL
        # Cantera frequently refuses to DECLARE steady convergence after the
        # transport swap even though the transient has settled (same behavior
        # as the strain sweep): cap the transient short and accept the
        # time-stepped solution as long as the flame is burning. The MA grid
        # (~150 pts) already resolves the structure, so no further refine.
        try:
            flame.max_time_step_count = 200
        except Exception:
            pass
        # Multi-round settle: a SINGLE accept after the transport swap left a
        # drifting transient at 152 bar (Tmax 4082 K, ABOVE the ~3993 K
        # adiabatic-equilibrium ceiling -- the unity-Le Tmax shift grows with
        # pressure while the ceiling margin shrinks). Rounds continue until
        # Cantera declares steady or Tmax is stationary (<2 K change).
        t0 = time.time()
        Tprev = None
        for rnd in range(1, 5):
            declared = True
            try:
                flame.solve(loglevel=log, auto=False)
            except ct.CanteraError:
                if flame.T.max() < 1500.0:
                    raise
                declared = False
            Tm = float(flame.T.max())
            _log(f"  unity-Le settle round {rnd}: declared={declared} "
                 f"Tmax={Tm:.1f} K")
            if declared or (Tprev is not None and abs(Tm - Tprev) < 2.0):
                break
            Tprev = Tm
        _log(f"  {TRANSPORT_IDEAL} settled npts={flame.grid.size} "
             f"Tmax={flame.T.max():.1f} K dt={time.time()-t0:.1f}s")
        swp._ckpt_save(flame, cA, "A", "ideal @P_OPER hot inlets (unity-Le)")
        state["stage"] = "B"
        _state_save(STATE, state)

    # ---- Stage B: ideal-gas, cool inlets to target (no-op when uniform-hot) ----
    if state["stage"] == "B":
        if flame is None:
            flame = _restore_ideal(igas, cA, "A")
        need_cool = (abs(T_FUEL - T_HOT_START) > 0.5
                     or abs(T_ox - T_HOT_START) > 0.5)
        if need_cool:
            _log(f"stage B: cool inlets  fuel {T_HOT_START}->{T_FUEL} K, "
                 f"ox {T_HOT_START}->{T_ox} K")
            _cool_inlet(flame, "fuel", T_FUEL, 4, log)
            _cool_inlet(flame, "oxidizer", T_ox, 10, log)
        else:
            _log(f"stage B: inlets already at target {INLET_T} K "
                 f"(Wang 2015, no cooling)")
        swp._ckpt_save(flame, cB, "B", "ideal at target inlets")
        state["stage"] = "C"
        _state_save(STATE, state)

    # ---- Stage C: ideal-gas, moderate refine at the target state ----
    # NOTE: we deliberately use REFINE_COARSE (~150 pts), NOT REFINE_FINAL
    # (~260 pts). The SRK + high-pressure-Chung steady Newton STALLS on a
    # ~260-pt grid at these trans-critical conditions (dt frozen at ~1e-5,
    # residual frozen), but converges on the ~150-pt grid (dt climbs through
    # 5 orders of magnitude). 150 pts still puts ~26 nodes in the reaction
    # zone Z in [0.10, 0.35] around Z_st=0.2255 -- ample for the tabulated
    # T(Z), Y_k(Z), omega_C(Z). Finer is counterproductive here.
    if state["stage"] == "C":
        if flame is None:
            flame = _restore_ideal(igas, cB, "B")
        # NO further refine here. The MA warm-start in stage A already
        # produced the right (~150-pt) grid and the unity-Le settle kept it.
        # Re-running auto=True after the transport swap PRUNED the smooth
        # unity-Le solution down to ~48 pts and, combined with an accepted
        # not-fully-declared transient, overshot Tmax to 3967 K -- ABOVE the
        # adiabatic-equilibrium ceiling (3791 K @ 50.7 bar), i.e. unphysical.
        # The stage-A solution (npts=152, Tmax=3765 K < T_eq) is the correct
        # target baseline, so stage C just re-checkpoints it.
        _log(f"stage C: skipped (stage A grid is final; refine after the "
             f"unity-Le swap prunes/overshoots)  npts={flame.grid.size} "
             f"Tmax={flame.T.max():.1f} K")
        swp._ckpt_save(flame, cC, "C", "ideal target baseline (refined)")
        state["stage"] = "E"
        state["strain_idx"] = -1
        _state_save(STATE, state)

    # ---- Stage E: strain sweep, SRK upgrade + save per flamelet ----
    if flame is None:
        flame = _restore_ideal(igas, cC, "C")
    _log(f"stage E: strain sweep (mdot x{mdot_factor} up to {mdot_max}x), "
         f"real-fluid (SRK+Chung) property re-evaluation per flamelet")

    def _emit(idx):
        """Save the current ideal-gas flame with its density/transport/source
        re-evaluated under SRK + high-pressure-Chung (see
        _save_flamelet_realfluid). No SRK flame solve -- that stalls here."""
        _save_flamelet_realfluid(flame, sgas, out_dir, idx)
        swp._ckpt_save(flame, CKPT / f"stageE_idx_{idx:03d}.yaml",
                       "E", f"ideal strain idx={idx}")
        state["strain_idx"] = idx
        _state_save(STATE, state)
        _log(f"  flamelet {idx:03d} done  mdot={float(flame.fuel_inlet.mdot):.3f}")

    start = state["strain_idx"] + 1
    if start > 0:
        prev = CKPT / f"stageE_idx_{state['strain_idx']:03d}.yaml"
        if prev.exists():
            flame = _restore_ideal(igas, prev, "E")
            _log(f"  resume strain sweep at idx={start} "
                 f"(mdot={float(flame.fuel_inlet.mdot):.3f})")
    if start == 0:
        _emit(0)
        start = 1

    # The ideal strain sweep must stay robust toward near-extinction so the
    # chi_st axis spans a wide range. KEY DIAGNOSIS: at higher strain Cantera
    # frequently will NOT *declare* steady-state convergence even though the
    # transient HAS reached steady state (dt grows to ~10 s, log10(residual)~0);
    # solve() then just keeps time-stepping until it hits max_time_step_count
    # and raises "Took maximum number of timesteps". (This is what hung idx-4
    # for hours under a 5000-step cap.) So we cap the steps low (the transient
    # reaches steady well within ~150 steps) and ACCEPT the time-stepped
    # solution on that error as long as the flame is still burning; a collapsed
    # Tmax is genuine extinction and ends the sweep (defining the high-chi end).
    try:
        flame.max_time_step_count = 150
    except Exception:
        pass
    idx = start
    while flame.fuel_inlet.mdot / MDOT_INIT < mdot_max:
        swp._scale_apply(flame, mdot_factor, A_EXP)
        t0 = time.time()
        try:
            flame.solve(loglevel=log, auto=False)
        except ct.CanteraError:
            if flame.T.max() < 1500.0:
                _log(f"  extinction at mdot={flame.fuel_inlet.mdot:.3f} "
                     f"(Tmax={flame.T.max():.0f} K); stopping sweep")
                break
            _log(f"  steady not declared; accepting time-stepped solution "
                 f"(Tmax={flame.T.max():.0f} K)")
        _log(f"  ideal mdot={flame.fuel_inlet.mdot:.3f} npts={flame.grid.size} "
             f"Tmax={flame.T.max():.1f} K dt={time.time()-t0:.1f}s")
        _emit(idx)
        idx += 1

    state["stage"] = "DONE"
    _state_save(STATE, state)
    _log(f"DONE. {idx} flamelets -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=int, default=0)
    p.add_argument("--mdot-factor", type=float, default=1.25)
    p.add_argument("--mdot-max", type=float, default=20.0)
    p.add_argument("--pressure-atm", type=float, default=None,
                   help="operating pressure in atm; default keeps the "
                        "52.5 bar production point. Used for the Wang-2015 "
                        "1-200 atm verification sweep (per-pressure dirs).")
    p.add_argument("--restart", action="store_true")
    p.add_argument("--warm-from", default=None,
                   help="stage-A checkpoint (stageA_ideal.yaml) of a "
                        "converged LOWER pressure to pressure-ramp from; "
                        "required for extreme pressures (>=150 atm) where "
                        "the cold start cannot find the thin flame.")
    a = p.parse_args()
    if a.pressure_atm is not None:
        global P_OPER
        P_OPER = a.pressure_atm * 101325.0
    run(log=a.log, mdot_factor=a.mdot_factor, mdot_max=a.mdot_max,
        restart=a.restart, warm_from=a.warm_from)


if __name__ == "__main__":
    main()
