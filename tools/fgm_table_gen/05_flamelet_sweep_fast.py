"""Fast flamelet table generation using Fiala-Sattelmayer (2014) scaling laws.

Adapted from Cantera's diffusion_flame_batch.py example
(https://cantera.org/stable/examples/python/onedim/diffusion_flame_batch.html).

Strategy (huge speed-up vs. the naive auto-refine path used in 03_*.py):

  1. Solve ONE cheap baseline at (P=1 atm, T_fuel=T_ox=700 K, mdot=mdot_0).
  2. P ramp 1 atm -> P_OPER via Fiala-Sattelmayer pressure scaling:
       grid    *= r^(-5/4)
       mdot    *= r^( 5/4)
       u       *= r^( 1/4)
       V       *= r^( 3/2)
       Lambda  *= r^( 4 )
     (r = P_new / P_old). Each step needs only a few Newton iterations.
  3. Cool T_fuel -> T_FUEL and T_ox -> T_OX_TARGET (adaptive linear ramps).
  4. mdot (strain) sweep at target (P_OPER, T_FUEL, T_OX_TARGET) using the
     analogous strain scaling exponents (-1/2, 1/2, 1/2, 1, 2).
  5. Save each strain-swept flamelet -> data/flamelets_wang2011_fast/.

WSL run:
  python3 05_flamelet_sweep_fast.py --mech wang2011
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Activate multi-thread BLAS *before* importing cantera. Cantera's own
# kinetics/transport code is single-threaded (conda-forge build), but the
# Sundials Lapack-band linear solver (libsundials_sunlinsollapackband) used
# inside Sim1D.solve() picks up these OpenBLAS/MKL thread counts. Empirical
# speed-up on the 14700KF + 28 vCPU WSL2 setup is ~1.3-1.5x; the 1D band
# is narrow, so don't expect linear scaling.
#
# Override with e.g. OPENBLAS_NUM_THREADS=1 on the command line if you want
# to compare against single-thread.
for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "14")

import numpy as np
import cantera as ct

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True, parents=True)

# --- target operating point (Ahn et al. 2012 CST 184:323, KARI bi-swirl) ---
# Validation case: 19-element LOX/Jet-A1 swirl-coaxial combustor. The design
# operating point (and the densest cluster of hot-fire data, HeadB = 24 tests)
# is P_c = 52.5 bar, OFR = 2.77. 52.5 bar sits just above O2's critical
# pressure (50.4 bar), so the table covers the supercritical-O2 regime that
# motivates the real-fluid SRK + Chung treatment. Previous target was 100 bar
# (Wang & Yang 2017 RD-0110); the gentler 1 atm -> 52.5 bar ramp also makes
# the trans-critical pressure continuation far more robust.
P_OPER = 52.5e5
T_OX_TARGET = 120.0
T_OX_CLAMP_MIN = 200.0       # NASA-7 polynomial safety
T_FUEL = 300.0
T_HOT_START = 700.0
DOMAIN_WIDTH = 0.02
MDOT_INIT = 0.05
X_OX = "O2:1.0"

# Cantera 3.1 accepts "Soave-Redlich-Kwong" in the YAML phase block;
# Cantera 3.2 only accepts "Redlich-Kwong" (Soave correction applied via
# acentric-factor). We keep two YAML variants and pick whichever Cantera
# can actually load at runtime.
MECHS = {
    "wang2011": [DATA / "wang2011_srk.yaml",
                 DATA / "wang2011_srk_v32.yaml"],
    "kero4s":   [DATA / "kero4s_srk.yaml"],
    "kero2s":   [DATA / "kero2s_srk.yaml"],
}


def _load_gas(mech):
    """Try the candidate YAMLs in order, returning the first that loads."""
    last_err = None
    for p in MECHS[mech]:
        if not p.exists():
            continue
        try:
            return ct.Solution(str(p)), p
        except Exception as ex:
            last_err = ex
    raise SystemExit(f"no usable YAML for mech={mech}\n  last error: {last_err}")
PV_SPECIES = ("CO2", "CO", "H2O", "H2")
DUMP_SPECIES = PV_SPECIES + ("O2", "NC10H22", "PHC3H7", "CYC9H18", "N2", "OH", "KERO")

X_FUEL_SURROGATE = "NC10H22:0.74, PHC3H7:0.15, CYC9H18:0.11"
X_FUEL_KERO = "KERO:1.0"

# --- Fiala-Sattelmayer scaling exponents (Cantera example) ---
P_EXP = dict(grid=-5/4, mdot=5/4, u=1/4, V=3/2, lam=4.0)
A_EXP = dict(grid=-1/2, mdot=1/2, u=1/2, V=1.0, lam=2.0)

# Transport model used everywhere. Cantera 3.2's "high-pressure-Chung" applies
# the Chung-Takahashi (1988) dense-gas corrections to mu, lambda, and binary
# diffusion using each species' critical-parameters block (present in
# wang2011_srk*.yaml). At dilute conditions it falls back to standard Chung
# kinetic theory, so the 1 atm baseline is also consistent. See the README
# for the impact study: at 100 bar / 120 K LOX the dense-gas correction
# raises mu by ~20x and lambda by ~7x relative to dilute Chapman-Enskog.
TRANSPORT_MODEL = "high-pressure-Chung"
TRANSPORT_FALLBACK = "mixture-averaged"

# --- solver acceleration knobs ---
# Newton Jacobian aging: re-evaluate the Jacobian every N solver steps.
# We tried 60 (failed at P=3 bar) and 20 (failed at P=3.6 bar) under
# high-pressure-Chung + Fiala-Sattelmayer scaling. The combination of
# dense-gas transport, SRK EOS, and aggressive grid rescaling apparently
# needs a fresh Jacobian almost every step, so we fall back to the
# Cantera default of 5. Acceleration comes from refine relaxation and BLAS
# multithreading instead.
JAC_AGE_SS = 5
JAC_AGE_TS = 5

# Refine criteria: lightly relaxed vs the original 3.0/0.10/0.20/0.03.
# The aggressive 4.0/0.15/0.30 produced only 126 grid points at 1 atm,
# which was not enough to ride the trans-critical pressure ramp.
REFINE_COARSE = dict(ratio=3.5, slope=0.12, curve=0.24, prune=0.04)
REFINE_FINAL  = dict(ratio=2.2, slope=0.07, curve=0.14, prune=0.022)
MAX_GRID_COARSE = 1200
MAX_GRID_FINAL  = 3500


def _log(*a):
    print("[fast]", *a, flush=True)


def _resolve_T_ox(gas):
    for T_try in (T_OX_TARGET, T_OX_CLAMP_MIN):
        try:
            gas.TPX = T_try, P_OPER, X_OX
            _ = gas.cp_mass
            return T_try
        except Exception as ex:
            _log(f"T_ox={T_try} K rejected: {ex}")
    raise RuntimeError("no valid T_ox")


def _fuel_X(gas):
    return X_FUEL_KERO if "KERO" in gas.species_names else X_FUEL_SURROGATE


# ---------- scaling-based step (Cantera diffusion_flame_batch.py) ----------

def _scale_apply(flame, ratio, exp):
    """Apply Fiala-Sattelmayer scaling to the current solution in-place
    using ratio = new/old and the supplied exponent dictionary.

    Supports both Cantera 3.1 (Sim1D.set_profile) and Cantera 3.2
    (Domain.set_values) APIs.
    """
    if ratio == 1.0:
        return
    # 1) grid scaling
    flame.flame.grid = flame.flame.grid * (ratio ** exp["grid"])
    # 2) mass-flow rate scaling
    flame.fuel_inlet.mdot *= ratio ** exp["mdot"]
    flame.oxidizer_inlet.mdot *= ratio ** exp["mdot"]

    # 3) velocity / spread_rate / Lambda profile scaling
    # --- try Cantera 3.1 API first (Sim1D.set_profile + normalized grid) ---
    try:
        g = np.asarray(flame.grid)
        span = max(g[-1] - g[0], 1e-30)
        norm = g / span
        u = np.asarray(flame.velocity) * (ratio ** exp["u"])
        V = np.asarray(flame.spread_rate) * (ratio ** exp["V"])
        L = np.asarray(flame.L) * (ratio ** exp["lam"])
        flame.set_profile("velocity",   norm, u)
        flame.set_profile("spread_rate", norm, V)
        flame.set_profile("lambda",     norm, L)
        return
    except Exception as ex_31:
        pass

    # --- Cantera 3.2 fallback (Domain.set_values, no normalization) ---
    try:
        flame.flame.set_values("velocity",
                               flame.flame.velocity * (ratio ** exp["u"]))
        flame.flame.set_values("spreadRate",
                               flame.flame.spread_rate * (ratio ** exp["V"]))
        flame.flame.set_values("Lambda",
                               flame.flame.radial_pressure_gradient
                               * (ratio ** exp["lam"]))
        return
    except Exception as ex_32:
        _log(f"WARN: profile update failed "
             f"(3.1 path: {ex_31}; 3.2 path: {ex_32})")


def _solve(flame, label, log=0):
    t0 = time.time()
    flame.solve(loglevel=log, auto=False)
    _log(f"  {label}: npts={flame.grid.size}  Tmax={flame.T.max():.1f} K  "
         f"dt={time.time()-t0:.1f} s")


# ---------- checkpoint / resume infrastructure ----------
# state.json schema:
#   {
#     "mech": "wang2011",
#     "stage": int,            # last fully-completed stage (0..6)
#     "p_idx": int,            # stage 2: last completed P index (-1 = none)
#     "tf":    float | null,   # stage 3: last completed T_fuel
#     "tox":   float | null,   # stage 4: last completed T_ox
#     "strain_idx": int,       # stage 6: last completed flamelet idx
#     "P_OPER": float, "T_FUEL": float, "T_OX_TARGET": float
#   }
# Per-step yaml files live alongside state.json:
#   stage1.yaml, stage2_P_<idx>.yaml, stage3_Tf_<T>K.yaml,
#   stage4_Tox_<T>K.yaml, stage5_target.yaml, stage6_idx_<i>.yaml

def _state_load(state_f):
    if state_f.exists():
        try:
            return json.loads(state_f.read_text())
        except Exception:
            return {}
    return {}


def _state_save(state_f, state):
    tmp = state_f.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=float))
    tmp.replace(state_f)


def _ckpt_save(flame, path, name, desc):
    try:
        flame.save(str(path), name=name, description=desc, overwrite=True)
    except Exception as ex:
        _log(f"  WARN: checkpoint save failed at {path.name}: {ex}")


def _ckpt_restore(gas, path, name, transport=None):
    """Reconstruct a CounterflowDiffusionFlame and restore from yaml.

    `transport` selects the transport model assigned to the rebuilt flame;
    None means TRANSPORT_MODEL (dense-gas high-pressure-Chung). The
    pressure/temperature continuation (stages 2-4) restores under the cheap
    TRANSPORT_FALLBACK because dense-gas Chung collapses Newton on a moving
    trans-critical state; Chung is reinstated at the fixed target (stage 5)."""
    flame = ct.CounterflowDiffusionFlame(gas, width=DOMAIN_WIDTH)
    flame.transport_model = transport or TRANSPORT_MODEL
    flame.restore(str(path), name=name)
    try:
        flame.set_max_grid_points(flame.flame, MAX_GRID_COARSE)
    except Exception:
        pass
    try:
        flame.set_max_jac_age(JAC_AGE_SS, JAC_AGE_TS)
    except Exception:
        pass
    return flame


# ---------- main pipeline ----------

def build_and_sweep(mech="wang2011", log=0,
                   mdot_sweep_factor=1.25,
                   mdot_max_factor=20.0,
                   restart=False):
    out_dir = DATA / f"flamelets_{mech}_fast"
    out_dir.mkdir(exist_ok=True, parents=True)

    gas, yaml_path = _load_gas(mech)
    _log(f"loaded {yaml_path.name}  (thermo_model={gas.thermo_model})")
    T_ox_target = _resolve_T_ox(gas)
    X_FUEL = _fuel_X(gas)
    _log(f"mech={mech}, T_ox_target={T_ox_target} K, X_FUEL={X_FUEL}")

    # Checkpoint directory shared by every stage.
    CKPT = DATA / f"ckpt_{mech}_fast"
    CKPT.mkdir(exist_ok=True, parents=True)
    STATE_F = CKPT / "state.json"

    if restart:
        _log("--restart given; ignoring any existing checkpoints/state")
        state = {}
    else:
        state = _state_load(STATE_F)
        if state:
            _log(f"resume: state.json found  stage={state.get('stage',0)}, "
                 f"p_idx={state.get('p_idx',-1)}, "
                 f"tf={state.get('tf')}, tox={state.get('tox')}, "
                 f"strain_idx={state.get('strain_idx',-1)}")

    # Schema version + run params (used to detect mismatch with checkpoints).
    state.setdefault("mech", mech)
    state.setdefault("stage", 0)
    state.setdefault("p_idx", -1)
    state.setdefault("tf", None)
    state.setdefault("tox", None)
    state.setdefault("strain_idx", -1)
    state["P_OPER"] = P_OPER
    state["T_FUEL"] = T_FUEL
    state["T_OX_TARGET"] = T_ox_target
    if state.get("mech") != mech:
        _log(f"WARN: state.json mech={state.get('mech')} != requested {mech}; "
             f"falling back to restart")
        state = {"mech": mech, "stage": 0, "p_idx": -1, "tf": None,
                 "tox": None, "strain_idx": -1, "P_OPER": P_OPER,
                 "T_FUEL": T_FUEL, "T_OX_TARGET": T_ox_target}
    _state_save(STATE_F, state)

    flame = None

    # ------- stage 1: baseline at 1 atm hot inlets -------
    s1_path = CKPT / "stage1.yaml"
    if state["stage"] < 1:
        flame = ct.CounterflowDiffusionFlame(gas, width=DOMAIN_WIDTH)
        flame.transport_model = TRANSPORT_MODEL
        flame.P = 1.0e5
        flame.fuel_inlet.T = T_HOT_START
        flame.fuel_inlet.X = X_FUEL
        flame.fuel_inlet.mdot = MDOT_INIT
        flame.oxidizer_inlet.T = T_HOT_START
        flame.oxidizer_inlet.X = X_OX
        flame.oxidizer_inlet.mdot = MDOT_INIT
        flame.set_refine_criteria(**REFINE_COARSE)
        try:
            flame.set_max_grid_points(flame.flame, MAX_GRID_COARSE)
        except Exception:
            pass
        try:
            flame.set_max_jac_age(JAC_AGE_SS, JAC_AGE_TS)
        except Exception:
            pass

        _log(f"stage 1: P=1 atm, T_fuel=T_ox={T_HOT_START} K  "
             f"(transport={TRANSPORT_MODEL}, "
             f"jac_age={JAC_AGE_SS}, refine={REFINE_COARSE['ratio']:.1f}/"
             f"{REFINE_COARSE['slope']:.2f})")
        t0 = time.time()
        try:
            flame.solve(loglevel=log, auto=True)
        except ct.CanteraError as ex:
            # high-pressure-Chung occasionally chokes on the auto-refine
            # cold-start at 1 atm; fall back to dilute kinetic theory just for
            # this initial guess, then switch back to dense-gas before stage 2.
            _log(f"  stage1 auto-solve failed under {TRANSPORT_MODEL}: "
                 f"{str(ex).splitlines()[-1] if str(ex) else '?'}")
            _log(f"  retrying with {TRANSPORT_FALLBACK} for the warm-start...")
            flame.transport_model = TRANSPORT_FALLBACK
            flame.solve(loglevel=log, auto=True)
            flame.transport_model = TRANSPORT_MODEL
            # one more non-auto solve to settle under the dense-gas transport
            flame.solve(loglevel=log, auto=False)
        _log(f"  done npts={flame.grid.size} Tmax={flame.T.max():.1f} K "
             f"dt={time.time()-t0:.1f} s")
        _ckpt_save(flame, s1_path, "stage1",
                   f"P=1atm T_fuel=T_ox={T_HOT_START} K "
                   f"transport={TRANSPORT_MODEL}")
        state["stage"] = 1
        _state_save(STATE_F, state)
    else:
        flame = _ckpt_restore(gas, s1_path, "stage1")
        _log(f"stage 1: skipped (loaded {s1_path.name}, "
             f"npts={flame.grid.size}, P={flame.P/1e5:.2f} bar)")

    # ------- stage 2: medium-step pressure ramp -------
    # History of attempts:
    #   - 20-step Fiala-Sattelmayer scaling ramp: collapsed at P=3 bar
    #   - 35/50-step scaling ramp:               collapsed at P~4 bar
    #   - 1-step single jump (1 atm -> 100 bar): single solve never returned
    #     (Newton + auto-refine on a 100x pressure jump is intractable)
    # 8 geometric steps (r ~ 1.7) without Fiala-Sattelmayer rescaling, with
    # auto=True refinement per step and snapshot+halve retry as a safety
    # net. Total wallclock target: ~30-45 min for stage 2.
    P_seq = list(np.geomspace(2.0e5, P_OPER, 8))
    last_p_idx = len(P_seq) - 1
    if state["stage"] < 2:
        start_idx = state["p_idx"] + 1
        if start_idx > 0:
            # mid-stage 2 resume
            prev_ck = CKPT / f"stage2_P_{state['p_idx']:02d}.yaml"
            flame = _ckpt_restore(gas, prev_ck, "stage2",
                                  transport=TRANSPORT_FALLBACK)
            _log(f"stage 2: resuming at P_idx={start_idx}/{len(P_seq)} from "
                 f"{prev_ck.name}  (P={flame.P/1e5:.2f} bar)")
        else:
            _log(f"stage 2: P ramp via scaling, {len(P_seq)} steps "
                 f"to {P_OPER/1e5:.0f} bar")
        # Continuation transport: run the trans-critical pressure ramp (and the
        # downstream temperature cooling in stages 3-4) under the cheap
        # TRANSPORT_FALLBACK. Dense-gas high-pressure-Chung collapses the Newton
        # solve whenever P or T moves at trans-critical LOX conditions (it hung
        # for >22 h on a single r=1.10 step). Chung is reinstated at the fixed
        # target state in stage 5 before any flamelet is saved.
        flame.transport_model = TRANSPORT_FALLBACK
        _log(f"stage 2-4: continuation under {TRANSPORT_FALLBACK} "
             f"({TRANSPORT_MODEL} reinstated at target in stage 5)")

        # Snapshot used by retry logic (overwritten before every P-step).
        snap_path = CKPT / "stage2_snap.yaml"

        # Allow many more transient timesteps before the solver gives up.
        # Default is 500; cubic EOS + Chung pseudo-critical needs more.
        try:
            flame.max_time_step_count = 5000
        except Exception:
            pass

        def _step_pressure(P_target, depth=0):
            """Bump pressure and let Cantera's auto-refine handle the new
            state. We previously tried Fiala-Sattelmayer scaling here, but
            it accumulates drift under dense-gas (SRK + high-pressure-Chung)
            and the Newton step eventually collapses (~P=4 bar) with no
            recovery even after deep retries. Plain `flame.P = P_new`
            followed by `solve(auto=True)` is slower but robust.

            On failure we still snapshot+halve as a safety net."""
            nonlocal flame
            P_prev_local = float(flame.P)
            flame.P = float(P_target)
            t0 = time.time()
            try:
                # auto=True so Cantera can refine the grid to follow the
                # thinning flame as pressure climbs.
                flame.solve(loglevel=log, auto=True)
                _log(f"  P={P_target/1e5:8.2f} bar: "
                     f"npts={flame.grid.size}  "
                     f"Tmax={flame.T.max():.1f} K  "
                     f"dt={time.time()-t0:.1f} s")
                return
            except ct.CanteraError as ex:
                msg = (str(ex).splitlines()[-1]
                       if str(ex) else "CanteraError")
                r = P_target / P_prev_local
                if depth >= 4 or r < 1.01:
                    _log(f"  GIVE UP at P={P_target/1e5:.2f} bar: {msg}")
                    raise
                flame = _ckpt_restore(gas, snap_path, "stage2_snap",
                                      transport=TRANSPORT_FALLBACK)
                try:
                    flame.max_time_step_count = 5000
                except Exception:
                    pass
                P_mid = (P_prev_local * P_target) ** 0.5
                _log(f"  retry: split P {P_prev_local/1e5:.2f} -> "
                     f"{P_target/1e5:.2f} via {P_mid/1e5:.2f} bar "
                     f"({msg})")
                _step_pressure(P_mid, depth + 1)
                _step_pressure(P_target, depth + 1)

        for i in range(start_idx, len(P_seq)):
            # Snapshot the pre-step state so retry can roll back.
            _ckpt_save(flame, snap_path, "stage2_snap",
                       f"pre-step snapshot, P_idx={i}")
            P_new = P_seq[i]
            _step_pressure(P_new)
            _ckpt_save(flame, CKPT / f"stage2_P_{i:02d}.yaml",
                       "stage2", f"stage2 P_idx={i} P={P_new:.3e} Pa")
            state["p_idx"] = i
            _state_save(STATE_F, state)
        # snapshot of hot-inlet @ P_OPER for downstream tools
        warmup = DATA / f"warmup_baseline_{mech}_fast.yaml"
        _ckpt_save(flame, warmup, "hot_baseline",
                   f"P={P_OPER:.0f} Pa T_fuel=T_ox={T_HOT_START} K")
        _log(f"saved hot-inlet baseline @ {P_OPER/1e5:.0f} bar -> "
             f"{warmup.name}")
        state["stage"] = 2
        _state_save(STATE_F, state)
    else:
        flame = _ckpt_restore(gas, CKPT / f"stage2_P_{last_p_idx:02d}.yaml",
                              "stage2", transport=TRANSPORT_FALLBACK)
        _log(f"stage 2: skipped (loaded final P_idx={last_p_idx}, "
             f"P={flame.P/1e5:.2f} bar)")

    # ------- stage 3: cool T_fuel -> T_FUEL -------
    if T_FUEL < T_HOT_START - 1.0:
        Tf_seq = list(np.linspace(T_HOT_START, T_FUEL, 5)[1:])
        if state["stage"] < 3:
            done_tf = state["tf"]
            # determine which Tf step we resume at
            start_j = 0
            if done_tf is not None:
                for j, T in enumerate(Tf_seq):
                    if abs(T - done_tf) < 0.5:
                        start_j = j + 1
                        break
                if start_j > 0:
                    prev_ck = CKPT / f"stage3_Tf_{done_tf:06.2f}K.yaml"
                    flame = _ckpt_restore(gas, prev_ck, "stage3",
                                          transport=TRANSPORT_FALLBACK)
                    _log(f"stage 3: resuming at Tf_idx={start_j} from "
                         f"{prev_ck.name}")
            if start_j == 0:
                _log(f"stage 3: cool fuel inlet {T_HOT_START} -> {T_FUEL} K "
                     f"({len(Tf_seq)} steps)")
            for j in range(start_j, len(Tf_seq)):
                T = Tf_seq[j]
                flame.fuel_inlet.T = float(T)
                _solve(flame, f"T_fuel={T:6.1f} K", log)
                _ckpt_save(flame, CKPT / f"stage3_Tf_{T:06.2f}K.yaml",
                           "stage3", f"stage3 T_fuel={T} K")
                state["tf"] = float(T)
                _state_save(STATE_F, state)
            state["stage"] = 3
            _state_save(STATE_F, state)
        else:
            final_Tf = Tf_seq[-1]
            flame = _ckpt_restore(gas,
                                  CKPT / f"stage3_Tf_{final_Tf:06.2f}K.yaml",
                                  "stage3", transport=TRANSPORT_FALLBACK)
            _log(f"stage 3: skipped (loaded T_fuel={final_Tf:.1f} K)")
    else:
        if state["stage"] < 3:
            state["stage"] = 3
            _state_save(STATE_F, state)

    # ------- stage 4: cool T_ox -> T_OX_TARGET (fine steps near 155 K) -------
    if T_ox_target < T_HOT_START - 1.0:
        Tox_sched = [600, 500, 400, 300, 250, 200, 180, 170,
                     165, 162, 160, 158, 156, 154, 152, 150, 148,
                     145, 140, 135, 130, 128, T_ox_target]
        Tox_sched = [T for T in Tox_sched if T >= T_ox_target - 0.5]
        if Tox_sched[-1] > T_ox_target + 0.5:
            Tox_sched.append(T_ox_target)

        if state["stage"] < 4:
            done_tox = state["tox"]
            start_k = 0
            if done_tox is not None:
                for k, T in enumerate(Tox_sched):
                    if abs(T - done_tox) < 0.5:
                        start_k = k + 1
                        break
                if start_k > 0:
                    prev_ck = CKPT / f"stage4_Tox_{done_tox:06.2f}K.yaml"
                    flame = _ckpt_restore(gas, prev_ck, "stage4",
                                          transport=TRANSPORT_FALLBACK)
                    _log(f"stage 4: resuming at Tox_idx={start_k} from "
                         f"{prev_ck.name}")
            if start_k == 0:
                _log(f"stage 4: cool ox inlet {T_HOT_START} -> "
                     f"{T_ox_target} K  ({len(Tox_sched)} steps; "
                     f"2 K granularity 162-148 K)")

            def _solve_T_ox(T, depth=0):
                """Set oxidizer T and solve; on cubic failure, halve dT
                recursively up to depth=4 (min step ≈ current_dT/16)."""
                T_prev = float(flame.oxidizer_inlet.T)
                flame.oxidizer_inlet.T = float(T)
                try:
                    _solve(flame, f"T_ox={T:6.1f} K", log)
                    return
                except ct.CanteraError as ex:
                    msg = (str(ex).splitlines()[-1]
                           if str(ex) else "CanteraError")
                    if depth >= 4 or abs(T - T_prev) < 0.25:
                        _log(f"  GIVE UP at T_ox={T:.2f} K: {msg}")
                        raise
                    T_mid = 0.5 * (T_prev + T)
                    _log(f"  retry: split T_ox {T_prev:.2f} -> "
                         f"{T:.2f} via {T_mid:.2f} K  ({msg})")
                    flame.oxidizer_inlet.T = float(T_prev)
                    _solve_T_ox(T_mid, depth + 1)
                    _solve_T_ox(T, depth + 1)

            for k in range(start_k, len(Tox_sched)):
                T = Tox_sched[k]
                _solve_T_ox(T)
                _ckpt_save(flame, CKPT / f"stage4_Tox_{T:06.2f}K.yaml",
                           "stage4", f"stage4 T_ox={T} K")
                state["tox"] = float(T)
                _state_save(STATE_F, state)
            state["stage"] = 4
            _state_save(STATE_F, state)
        else:
            final_Tox = Tox_sched[-1]
            flame = _ckpt_restore(gas,
                                  CKPT / f"stage4_Tox_{final_Tox:06.2f}K.yaml",
                                  "stage4", transport=TRANSPORT_FALLBACK)
            _log(f"stage 4: skipped (loaded T_ox={final_Tox:.1f} K)")
    else:
        if state["stage"] < 4:
            state["stage"] = 4
            _state_save(STATE_F, state)

    # ------- stage 5: final tight refine at the production state -------
    s5_path = CKPT / "stage5_target.yaml"
    target_baseline = DATA / f"baseline_{mech}_fast.yaml"
    if state["stage"] < 5:
        _log(f"stage 5: final refine "
             f"(ratio={REFINE_FINAL['ratio']:.1f}, "
             f"slope={REFINE_FINAL['slope']:.2f})")
        flame.set_refine_criteria(**REFINE_FINAL)
        try:
            flame.set_max_grid_points(flame.flame, MAX_GRID_FINAL)
        except Exception:
            pass
        # Reinstate dense-gas Chung at the *fixed* target state (no P/T motion
        # now, so the Newton solve is far more forgiving than the continuation
        # ramp was). Settle on the existing grid first (auto=False) to absorb
        # the property jump from the fallback transport, then refine. If Chung
        # still collapses, keep the mixture-averaged flamelets and flag it: the
        # table is then transport-fallback, which the paper notes as a caveat.
        _solver_transport = TRANSPORT_FALLBACK
        t0 = time.time()
        try:
            flame.transport_model = TRANSPORT_MODEL
            _log(f"  reinstating {TRANSPORT_MODEL} at target "
                 f"(P={P_OPER/1e5:.1f} bar, T_ox={T_ox_target:.0f} K)")
            flame.solve(loglevel=log, auto=False)
            flame.solve(loglevel=log, auto=True)
            _solver_transport = TRANSPORT_MODEL
        except ct.CanteraError as ex:
            msg = str(ex).splitlines()[-1] if str(ex) else "CanteraError"
            _log(f"  WARN: {TRANSPORT_MODEL} failed at target ({msg}); "
                 f"keeping {TRANSPORT_FALLBACK} for flamelets")
            flame = _ckpt_restore(gas,
                                  CKPT / f"stage4_Tox_{T_ox_target:06.2f}K.yaml",
                                  "stage4", transport=TRANSPORT_FALLBACK)
            flame.set_refine_criteria(**REFINE_FINAL)
            try:
                flame.set_max_grid_points(flame.flame, MAX_GRID_FINAL)
            except Exception:
                pass
            flame.solve(loglevel=log, auto=True)
        _log(f"  done npts={flame.grid.size} Tmax={flame.T.max():.1f} K "
             f"transport={_solver_transport} dt={time.time()-t0:.1f} s")
        desc = (f"P={P_OPER:.0f} Pa T_fuel={T_FUEL} K "
                f"T_ox={T_ox_target} K")
        _ckpt_save(flame, s5_path, "stage5", desc)
        _ckpt_save(flame, target_baseline, "target_baseline", desc)
        _log(f"saved TARGET baseline -> {target_baseline.name}")
        state["stage"] = 5
        _state_save(STATE_F, state)
    else:
        flame = _ckpt_restore(gas, s5_path, "stage5")
        _log(f"stage 5: skipped (loaded {s5_path.name})")

    # ------- stage 6: strain (mdot) sweep at target -------
    _log(f"stage 6: strain sweep, mdot factor {mdot_sweep_factor} until "
         f"{mdot_max_factor}×")
    start_idx = state["strain_idx"] + 1
    if start_idx > 0:
        prev_ck = CKPT / f"stage6_idx_{state['strain_idx']:03d}.yaml"
        if prev_ck.exists():
            flame = _ckpt_restore(gas, prev_ck, "stage6")
            _log(f"  resuming strain sweep at idx={start_idx} from "
                 f"{prev_ck.name}  (mdot={flame.fuel_inlet.mdot:.3f})")
    if start_idx == 0:
        _save_flamelet(flame, gas, out_dir, idx=0)
        _ckpt_save(flame, CKPT / "stage6_idx_000.yaml",
                   "stage6", "stage6 idx=0 (target baseline)")
        state["strain_idx"] = 0
        _state_save(STATE_F, state)
        start_idx = 1

    idx = start_idx
    while flame.fuel_inlet.mdot / MDOT_INIT < mdot_max_factor:
        _scale_apply(flame, mdot_sweep_factor, A_EXP)
        try:
            _solve(flame, f"mdot={flame.fuel_inlet.mdot:.3f} kg/m2/s", log)
        except ct.CanteraError as ex:
            _log(f"extinction or solver failure ({ex}); stopping sweep")
            break
        _save_flamelet(flame, gas, out_dir, idx=idx)
        _ckpt_save(flame, CKPT / f"stage6_idx_{idx:03d}.yaml",
                   "stage6", f"stage6 idx={idx} "
                              f"mdot={flame.fuel_inlet.mdot:.3f}")
        state["strain_idx"] = idx
        _state_save(STATE_F, state)
        idx += 1

    state["stage"] = 6
    _state_save(STATE_F, state)
    _log(f"saved {idx} flamelets -> {out_dir}")
    _log(f"DONE. Checkpoint dir: {CKPT}")


# ---------- flamelet IO ----------

def _bilger_Z(flame):
    try:
        return np.asarray(flame.mixture_fraction("Bilger"))
    except Exception:
        z = np.asarray(flame.grid)
        return (z - z[0]) / max(z[-1] - z[0], 1e-30)


def _progress_variable(flame, gas):
    Y = np.zeros(flame.grid.size)
    for sp in PV_SPECIES:
        if sp in gas.species_names:
            Y += flame.Y[gas.species_index(sp)]
    return Y


def _source_pv(flame, gas):
    omega_mol = flame.net_production_rates
    Mw = gas.molecular_weights
    src = np.zeros(flame.grid.size)
    for sp in PV_SPECIES:
        if sp in gas.species_names:
            k = gas.species_index(sp)
            src += omega_mol[k] * Mw[k]
    rho = np.asarray(getattr(flame, "density_mass", flame.density))
    return src / np.maximum(rho, 1e-30)


_Z_ST = 0.2255  # LOX / kerosene-surrogate stoichiometric Bilger Z_st.
# Computed from Cantera set_equivalence_ratio(1.0, surrogate, O2) +
# mixture_fraction(..., element="Bilger") for the Wang-2011 surrogate
# (0.74 NC10H22 / 0.15 PHC3H7 / 0.11 CYC9H18); cross-checked against the
# flamelet T_max location (Bilger Z = 0.233). The previous value 0.0625 was
# wrong by ~3.6x and put the χ_st probe in the lean (non-reacting) region.


def _chi_profile(z, Z, lam, rho, cp):
    """Scalar dissipation rate profile χ(x) = 2 α (dZ/dx)² along the
    1-D counterflow grid, with α = λ / (ρ·cp) the thermal diffusivity.
    Returned in 1/s. Required by the 4-axis FPV table builder."""
    alpha = np.asarray(lam) / (np.maximum(np.asarray(rho), 1e-30)
                                * np.maximum(np.asarray(cp), 1e-30))
    dZdx = np.gradient(np.asarray(Z), np.asarray(z))
    return 2.0 * alpha * dZdx * dZdx


def _chi_at_Zst(Z, chi, Z_st=_Z_ST):
    """Pull χ at Z = Z_st by 1-D interpolation along the flame profile.
    Falls back to χ_max if Z_st is outside the [Z_min, Z_max] of the
    flamelet (e.g. a near-extinct branch that doesn't reach Z_st)."""
    Z = np.asarray(Z); chi = np.asarray(chi)
    if Z_st < Z.min() or Z_st > Z.max():
        return float(np.max(chi))
    # Sort by Z because counterflow grid is monotonic in x but Z is not
    # guaranteed to be (it's still monotonic in practice, but be safe).
    order = np.argsort(Z)
    return float(np.interp(Z_st, Z[order], chi[order]))


def _save_flamelet(flame, gas, out_dir, idx):
    path = out_dir / f"flamelet_{idx:03d}.npz"
    rho  = np.asarray(getattr(flame, "density_mass", flame.density))
    T    = np.asarray(flame.T)
    Z    = _bilger_Z(flame)
    # Transport profiles needed by the 4-axis (Z, gZ, C, χ_st) builder.
    lam  = np.asarray(flame.thermal_conductivity)
    mu   = np.asarray(flame.viscosity)
    cp   = np.asarray(flame.cp_mass)
    alpha = lam / np.maximum(rho * cp, 1e-30)
    chi   = _chi_profile(flame.grid, Z, lam, rho, cp)
    chi_st = _chi_at_Zst(Z, chi)
    arrs = {
        "z":       np.asarray(flame.grid),
        "Z":       Z,
        "T":       T,
        "rho":     rho,
        "lam":     lam,
        "mu":      mu,
        "cp":      cp,
        "alpha":   alpha,
        "chi":     chi,
        "chi_st":  np.asarray(chi_st),
        "C":       _progress_variable(flame, gas),
        "omega_C": _source_pv(flame, gas),
        "mdot":    np.asarray(float(flame.fuel_inlet.mdot)),
        "P":       np.asarray(float(flame.P)),
        "T_fuel":  np.asarray(float(flame.fuel_inlet.T)),
        "T_ox":    np.asarray(float(flame.oxidizer_inlet.T)),
        "Z_st_ref": np.asarray(_Z_ST),
        "npts":    np.asarray(int(flame.grid.size)),
        "Tmax":    np.asarray(float(flame.T.max())),
        "idx":     np.asarray(int(idx)),
    }
    for sp in gas.species_names:
        arrs[f"Y_{sp}"] = flame.Y[gas.species_index(sp)]
    np.savez_compressed(path, **arrs)
    _log(f"  saved {path.name}  (mdot={float(flame.fuel_inlet.mdot):.3f}, "
         f"chi_st={chi_st:.3g} 1/s, Tmax={flame.T.max():.1f} K)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mech", choices=list(MECHS), default="wang2011")
    p.add_argument("--log", type=int, default=0,
                   help="Cantera solver loglevel")
    p.add_argument("--mdot-factor", type=float, default=1.25,
                   help="strain (mdot) sweep ratio per step")
    p.add_argument("--mdot-max", type=float, default=20.0,
                   help="terminate sweep when mdot/mdot_0 exceeds this")
    p.add_argument("--restart", action="store_true",
                   help="ignore existing checkpoints and run from scratch")
    args = p.parse_args()
    build_and_sweep(mech=args.mech, log=args.log,
                    mdot_sweep_factor=args.mdot_factor,
                    mdot_max_factor=args.mdot_max,
                    restart=args.restart)


if __name__ == "__main__":
    main()
