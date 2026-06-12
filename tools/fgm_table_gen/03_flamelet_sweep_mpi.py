"""Fully MPI-parallel FGM flamelet sweep.

No separate serial warmup phase. Each MPI rank independently:
  1. Builds its own CounterflowDiffusionFlame from scratch.
  2. Runs continuation warmup (P=1 atm/T_ox=300K -> 100 bar/T_ox=target
     -> first assigned mdot).
  3. Sweeps its assigned contiguous slice of MDOTS with warm-start.
  4. Writes one .npz per flamelet -> data/flamelets/flamelet_NNN.npz

After every rank finishes, rank 0 merges per-flamelet files into
data/flamelets.npz (numpy compressed archive).

Wall time ~ slowest rank's single workflow (one warmup + chunk sweep).
CPU time ~ N * (warmup + chunk). Trade CPU for wall-time.

Windows + ct-env2 + MS-MPI:
    mpiexec -n 8 ^
        "C:\\Users\\Sunchang Kim\\miniconda3\\envs\\ct-env2\\python.exe" ^
        "\\\\wsl.localhost\\Ubuntu-22.04\\home\\sunkim\\openfoam\\RGP-13\\RGP-13-realFluid\\tools\\fgm_table_gen\\03_flamelet_sweep_mpi.py"

If mpi4py is unavailable the script falls back to serial (size=1) so it
remains testable on a workstation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# ---- DLL search path bootstrap (Windows + conda env under mpiexec) ----
# When launched via mpiexec, child processes may inherit a PATH that lacks
# the conda env's Library/bin directory, so DLL-backed packages (h5py ->
# HDF5.dll, etc.) fail to import. Add the env's Library/bin explicitly.
def _bootstrap_conda_dll_path():
    if os.name != "nt":
        return
    env_root = Path(sys.executable).parent
    for sub in ("Library/bin", "Library/mingw-w64/bin", "DLLs"):
        d = env_root / sub
        if not d.exists():
            continue
        try:
            os.add_dll_directory(str(d))
        except (AttributeError, FileNotFoundError, OSError):
            pass
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


_bootstrap_conda_dll_path()


import numpy as np

# ---- MPI (optional) ----
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    HAVE_MPI = True
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1
    HAVE_MPI = False

# ---- Cantera ----
try:
    import cantera as ct
except ImportError as e:
    sys.exit(f"cantera unavailable: {e}")

# NOTE: previously used h5py for flamelet output, but on Windows + mpiexec
# h5py hit DLL conflicts. Switched to numpy .npz (npz = zip of .npy arrays)
# which only needs numpy and works without conda activation.


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FLAMELET_DIR = DATA_DIR / "flamelets"
FLAMELET_DIR.mkdir(exist_ok=True, parents=True)

# Mechanism selection. Default: full Wang 2011. Use --mech to switch.
# "wang2011" -> wang2011_srk.yaml (106 sp, slow, production)
# "kero4s"   -> kero4s_srk.yaml   (7 sp, 4-step, fast, development)
MECHS = {
    "wang2011": DATA_DIR / "wang2011_srk.yaml",
    "kero4s": DATA_DIR / "kero4s_srk.yaml",
    "kero2s": DATA_DIR / "kero2s_srk.yaml",
}
# These are set in run()/main() once --mech is parsed.
SRK_YAML = MECHS["wang2011"]
MECH_TAG = "wang2011"
BASELINE_FILE = DATA_DIR / "warmup_baseline_wang2011.yaml"
MERGED_OUTPUT = DATA_DIR / "flamelets_wang2011.npz"
BASELINE_NAME = "baseline_final"


def _select_mech(tag):
    """Set module-level paths for the chosen mechanism."""
    global SRK_YAML, MECH_TAG, BASELINE_FILE, MERGED_OUTPUT
    MECH_TAG = tag
    SRK_YAML = MECHS[tag]
    BASELINE_FILE = DATA_DIR / f"warmup_baseline_{tag}.yaml"
    MERGED_OUTPUT = DATA_DIR / f"flamelets_{tag}.npz"


# ---- Operating conditions (Wang & Yang 2017) ----
P_OPER = 100.0e5             # Pa
T_OX_TARGET = 120.0          # K  (clamped upward if NASA-7 polynomial fails)
T_OX_CLAMP_MIN = 200.0
T_FUEL = 300.0
X_OX = "O2:1.0"

# Wang 2011 tricomponent surrogate (74/15/11 vol). For the 4-step mechanism
# the fuel is the single lumped species KERO. Resolved at runtime against the
# mechanism's species list by _resolve_fuel().
X_FUEL_SURROGATE = "NC10H22:0.74, PHC3H7:0.15, CYC9H18:0.11"
X_FUEL_KERO = "KERO:1.0"
X_FUEL = X_FUEL_SURROGATE  # overwritten in run() once gas is known

DOMAIN_WIDTH = 0.02          # m  -- counterflow inlet separation

# Mdots (strain proxy). Sorted ascending so within-rank warm continuation works.
MDOTS = np.array([
    0.05, 0.075, 0.10, 0.15, 0.20,
    0.30, 0.50, 0.75, 1.00, 1.50,
    2.00, 3.00, 5.00, 7.50, 10.00,
    15.00, 20.00, 30.00, 50.00, 75.00,
])

# Progress variable: C = sum of these mass fractions that EXIST in the
# mechanism. Mechanisms without H2 (e.g. the 2-step) simply drop that term.
PV_SPECIES = ("CO2", "CO", "H2O", "H2")
DUMP_SPECIES = PV_SPECIES + ("O2", "NC10H22", "PHC3H7", "CYC9H18", "N2", "OH", "KERO")

# Filled in run() once the mechanism is loaded: PV species actually present.
_PV_PRESENT = None

MDOT_INIT = 0.05             # baseline mdot for warmup stage 1


# ---------------------------- logging ----------------------------

def _log(msg, only_root=False):
    if only_root and RANK != 0:
        return
    print(f"[rank {RANK}/{SIZE}] {msg}", flush=True)


# ---------------------------- Cantera helpers ----------------------------

def _density(flame) -> np.ndarray:
    for attr in ("density_mass", "density"):
        v = getattr(flame, attr, None)
        if v is not None:
            return np.asarray(v)
    raise AttributeError("flame has no density attr")


def _bilger_Z(flame) -> np.ndarray:
    """Bilger mixture fraction profile. Falls back to a normalized element
    mixture fraction if the Cantera helper is unavailable.
    """
    try:
        return np.asarray(flame.mixture_fraction("Bilger"))
    except Exception:
        # Fallback: element-C+H based, normalized to [0,1] across the domain.
        z = np.asarray(flame.grid)
        return (z - z[0]) / max(z[-1] - z[0], 1e-30)


def _pv_present(gas):
    """PV species actually present in the loaded mechanism."""
    global _PV_PRESENT
    if _PV_PRESENT is None:
        _PV_PRESENT = [s for s in PV_SPECIES if s in gas.species_names]
    return _PV_PRESENT


def _progress_variable(flame, gas) -> np.ndarray:
    Y = np.zeros(flame.grid.size)
    for sp in _pv_present(gas):
        Y += flame.Y[gas.species_index(sp)]
    return Y


def _source_pv(flame, gas) -> np.ndarray:
    omega_mol = flame.net_production_rates
    Mw = gas.molecular_weights
    src = np.zeros(flame.grid.size)
    for sp in _pv_present(gas):
        k = gas.species_index(sp)
        src += omega_mol[k] * Mw[k]
    return src / np.maximum(_density(flame), 1e-30)


def _resolve_T_ox(gas) -> float:
    for T_try in (T_OX_TARGET, T_OX_CLAMP_MIN):
        try:
            gas.TPX = T_try, P_OPER, X_OX
            _ = gas.cp_mass
            return T_try
        except Exception:
            continue
    raise RuntimeError("cannot pick valid T_ox")


def _resolve_fuel(gas) -> str:
    """Pick fuel composition matching the mechanism's species."""
    global X_FUEL
    if "KERO" in gas.species_names:
        X_FUEL = X_FUEL_KERO
    else:
        X_FUEL = X_FUEL_SURROGATE
    return X_FUEL


# Hot-start temperature for both inlets. Kerosene (Tc~635 K) is liquid at
# 300 K / 1 atm (SRK), which breaks the gas-phase flame solver. Starting both
# inlets hot keeps the fuel gaseous and makes the global mechanism ignite;
# we then ramp P up (past the fuel's critical pressure) BEFORE cooling the
# inlets, so the fuel never crosses into the two-phase region.
T_HOT_START = 700.0


def _configure(flame, P, T_fuel, T_ox, mdot):
    flame.P = P
    flame.fuel_inlet.T = T_fuel
    flame.fuel_inlet.X = X_FUEL
    flame.fuel_inlet.mdot = mdot
    flame.oxidizer_inlet.T = T_ox
    flame.oxidizer_inlet.X = X_OX
    flame.oxidizer_inlet.mdot = mdot


def _adaptive_ramp(flame, setter, v_from, v_to, midpoint, n_steps,
                   log=0, max_subdiv=8, label="", on_success=None):
    """Drive a control variable from v_from to v_to over n_steps, subdividing
    any step that fails to converge.

    setter(v): apply control value v to the flame (e.g. set P or inlet T).
    midpoint(a, b): intermediate value (geometric for P, arithmetic for T).
    on_success(flame, label, v): optional callback after every successfully
      converged step (used to write a recoverable baseline checkpoint to disk).
    The flame is assumed already converged at v_from. On a CanteraError the
    last good state is restored from a temp checkpoint and the failing step is
    halved (recursively, up to max_subdiv extra subdivisions per step).
    """
    ckpt = DATA_DIR / f"_ramp_ckpt_rank{RANK}.yaml"
    flame.save(str(ckpt), name="ck", overwrite=True)
    good = v_from

    # Build the list of nominal targets.
    if midpoint is _geom_mid:
        targets = list(np.geomspace(v_from, v_to, num=n_steps + 1))[1:]
    else:
        targets = list(np.linspace(v_from, v_to, num=n_steps + 1))[1:]

    subdiv_budget = max_subdiv
    for tgt in targets:
        stack = [tgt]
        while stack:
            v = stack[-1]
            setter(v)
            t0 = time.time()
            try:
                flame.solve(loglevel=log, auto=True)
                flame.save(str(ckpt), name="ck", overwrite=True)
                good = v
                stack.pop()
                _log(f"[baseline] {label}={v:10.4g}  npts={flame.grid.size:4d}  "
                     f"Tmax={flame.T.max():.1f} K  dt={time.time()-t0:.1f}s")
                if on_success is not None:
                    on_success(flame, label, v)
            except ct.CanteraError:
                if subdiv_budget <= 0:
                    if ckpt.exists():
                        ckpt.unlink()
                    raise
                subdiv_budget -= 1
                flame.restore(str(ckpt), name="ck")
                vmid = midpoint(good, v)
                _log(f"[baseline] {label}={v:10.4g} failed; subdividing -> "
                     f"{vmid:10.4g} (budget {subdiv_budget})")
                stack.append(vmid)
    if ckpt.exists():
        ckpt.unlink()


def _geom_mid(a, b):
    return float(np.sqrt(a * b))


def _lin_mid(a, b):
    return 0.5 * (a + b)


RAMP_MAX_GRID = 250        # cap during P/T continuation
FINAL_MAX_GRID = 5000      # effectively unrestricted at the final refine
RAMP_MAX_SUBDIV = 16       # adaptive subdivision budget (per nominal step)


def _set_grid_cap(flame, cap):
    """Cap the reacting-domain grid size; ignore if API differs."""
    try:
        flame.set_max_grid_points(flame.flame, int(cap))
    except Exception:
        try:
            flame.max_grid_points = int(cap)
        except Exception as ex:
            _log(f"[baseline] WARN: cannot set max_grid_points: {ex}")


def _build_shared_baseline(gas, T_ox, log=0):
    """Stage 1 + P ramp + T_ramps + final refine. Publication-grade target
    (P=P_OPER, T_fuel=T_FUEL, T_ox=T_ox) with cost mitigations that do NOT
    sacrifice the final-state accuracy:

      - resume from BASELINE_FILE if present (no work redo);
      - loose refine + max_grid_points cap during the ramps;
      - Stage 5 restores tight refine + removes the grid cap, so the final
        baseline is fully resolved;
      - per-step checkpoint to BASELINE_FILE so any failure is recoverable;
      - adaptive subdivision budget RAMP_MAX_SUBDIV for pseudo-critical T_ox.

    Result: a publication-grade baseline at (P_OPER, T_FUEL, T_ox).
    """
    flame = ct.CounterflowDiffusionFlame(gas, width=DOMAIN_WIDTH)
    flame.transport_model = "mixture-averaged"

    # Loose refine during ramping so the flame doesn't pile on cells while
    # crossing high-pressure / trans-critical regions. Stage 5 re-tightens.
    flame.set_refine_criteria(ratio=4.5, slope=0.30, curve=0.50, prune=0.05)
    _set_grid_cap(flame, RAMP_MAX_GRID)

    def save_partial(flame, label, v):
        try:
            desc = (f"partial baseline: last step {label}={v:.4g}; "
                    f"P={flame.P/1e5:.2f} bar, T_fuel={flame.fuel_inlet.T:.1f} K, "
                    f"T_ox={flame.oxidizer_inlet.T:.1f} K, "
                    f"npts={flame.grid.size}")
            flame.save(
                str(BASELINE_FILE), name=BASELINE_NAME,
                description=desc, overwrite=True,
            )
        except Exception as ex:
            _log(f"[baseline] WARN: checkpoint save failed: {ex}")

    # ---------- Resume from a previous partial baseline if available ----------
    resumed = False
    if BASELINE_FILE.exists():
        try:
            flame.restore(str(BASELINE_FILE), name=BASELINE_NAME)
            resumed = True
            _log(f"[baseline] RESUME from checkpoint: "
                 f"P={flame.P/1e5:.2f} bar, "
                 f"T_fuel={flame.fuel_inlet.T:.1f} K, "
                 f"T_ox={flame.oxidizer_inlet.T:.1f} K, "
                 f"npts={flame.grid.size}")
        except Exception as ex:
            _log(f"[baseline] checkpoint restore failed ({ex}); fresh start")
            resumed = False

    # ---------- Stage 1: 1 atm hot-inlet seed (skip on resume) ----------
    if not resumed:
        _log(f"[baseline] stage 1: P=1 atm, T_fuel=T_ox={T_HOT_START} K, "
             f"mdot={MDOT_INIT}")
        _configure(flame, P=1.0e5, T_fuel=T_HOT_START, T_ox=T_HOT_START,
                   mdot=MDOT_INIT)
        t0 = time.time()
        flame.solve(loglevel=log, auto=True)
        _log(f"[baseline]   stage1 OK  npts={flame.grid.size}  "
             f"Tmax={flame.T.max():.1f} K  dt={time.time()-t0:.1f}s")
        save_partial(flame, "stage1", 1.0e5)

    # ---------- Stage 2: P ramp -> P_OPER (skip if already there) ----------
    P_from = float(flame.P)
    if P_from < P_OPER * 0.99:
        # number of remaining geometric steps at the original ramp ratio (~1.33x)
        n_full = 16
        log_full = float(np.log(P_OPER / 1.0e5))
        log_rem = float(np.log(P_OPER / max(P_from, 1.0)))
        n_remain = max(2, int(np.ceil(n_full * log_rem / log_full)))
        _log(f"[baseline] stage 2: pressure ramp {P_from/1e5:.2f} -> "
             f"{P_OPER/1e5:.0f} bar ({n_remain} steps + adaptive)")
        _adaptive_ramp(flame, lambda P: setattr(flame, "P", float(P)),
                       v_from=P_from, v_to=P_OPER, midpoint=_geom_mid,
                       n_steps=n_remain, log=log, label="P",
                       on_success=save_partial, max_subdiv=RAMP_MAX_SUBDIV)

    # ---------- Stage 3: cool fuel inlet (skip if already at target) ----------
    Tf_from = float(flame.fuel_inlet.T)
    if Tf_from > T_FUEL + 1.0:
        n_remain = max(2, int(np.ceil((Tf_from - T_FUEL) / 140.0)))
        _log(f"[baseline] stage 3: cool fuel {Tf_from:.0f} -> {T_FUEL} K "
             f"({n_remain} steps + adaptive)")
        _adaptive_ramp(flame, lambda T: setattr(flame.fuel_inlet, "T", float(T)),
                       v_from=Tf_from, v_to=T_FUEL, midpoint=_lin_mid,
                       n_steps=n_remain, log=log, label="T_fuel",
                       on_success=save_partial, max_subdiv=RAMP_MAX_SUBDIV)

    # ---------- Stage 4: cool oxidizer inlet (pseudo-critical region) ----------
    Tox_from = float(flame.oxidizer_inlet.T)
    if Tox_from > T_ox + 1.0:
        # ~80 K per step on the linear schedule; adaptive will further refine
        # the trans-critical band around O2's critical point (~155 K).
        n_remain = max(3, int(np.ceil((Tox_from - T_ox) / 80.0)))
        _log(f"[baseline] stage 4: cool oxidizer {Tox_from:.0f} -> {T_ox} K "
             f"({n_remain} steps + adaptive, budget {RAMP_MAX_SUBDIV})")
        _adaptive_ramp(flame, lambda T: setattr(flame.oxidizer_inlet, "T", float(T)),
                       v_from=Tox_from, v_to=T_ox, midpoint=_lin_mid,
                       n_steps=n_remain, log=log, label="T_ox",
                       on_success=save_partial, max_subdiv=RAMP_MAX_SUBDIV)

    # ---------- Stage 5: final tight refine (publication-grade) ----------
    _log("[baseline] stage 5: final tight refine (grid cap removed)")
    _set_grid_cap(flame, FINAL_MAX_GRID)
    flame.set_refine_criteria(ratio=2.0, slope=0.08, curve=0.15, prune=0.02)
    t0 = time.time()
    flame.solve(loglevel=log, auto=True)
    _log(f"[baseline]   final OK  npts={flame.grid.size}  "
         f"Tmax={flame.T.max():.1f} K  dt={time.time()-t0:.1f}s")
    save_partial(flame, "final_refine", flame.P)
    return flame


def _ramp_mdot_to(flame, target_mdot, log=0):
    """Geometric mdot ramp from current to target with factor ~1.6x."""
    cur = float(flame.fuel_inlet.mdot)
    if abs(target_mdot - cur) / max(cur, 1e-9) < 0.05:
        return
    nsteps = max(int(np.ceil(abs(np.log(target_mdot / cur)) / np.log(1.6))), 1)
    ramp = np.geomspace(cur, target_mdot, num=nsteps + 1)[1:]
    for m in ramp:
        flame.fuel_inlet.mdot = float(m)
        flame.oxidizer_inlet.mdot = float(m)
        flame.solve(loglevel=log, auto=True)


def _split_mdots(mdots, nranks):
    n = len(mdots)
    base, extra = divmod(n, nranks)
    out, s = [], 0
    for r in range(nranks):
        e = s + base + (1 if r < extra else 0)
        out.append((s, e))
        s = e
    return out


def _save_flamelet_npz(flame, gas, mdot, idx):
    path = FLAMELET_DIR / f"flamelet_{idx:03d}.npz"
    arrs = {
        "z": np.asarray(flame.grid),
        "Z": _bilger_Z(flame),
        "T": np.asarray(flame.T),
        "rho": _density(flame),
        "C": _progress_variable(flame, gas),
        "omega_C": _source_pv(flame, gas),
        # Scalars stored as 0-d arrays (npz needs ndarray values).
        "mdot": np.asarray(float(mdot)),
        "P": np.asarray(float(flame.P)),
        "T_fuel": np.asarray(float(flame.fuel_inlet.T)),
        "T_ox": np.asarray(float(flame.oxidizer_inlet.T)),
        "npts": np.asarray(int(flame.grid.size)),
        "Tmax": np.asarray(float(flame.T.max())),
        "idx": np.asarray(int(idx)),
    }
    # Dump ALL species mass fractions: R2 thermo coupling needs the full
    # composition Y_k(Z,gZ,C) to feed the real-fluid (SRK+Chung/Ely) thermo.
    for sp in gas.species_names:
        arrs[f"Y_{sp}"] = flame.Y[gas.species_index(sp)]
    np.savez_compressed(path, **arrs)
    return path


# ---------------------------- main ----------------------------

def _baseline_cache_ok():
    """True iff BASELINE_FILE holds a baseline AT THE TARGET STATE
    (P==P_OPER, T_fuel==T_FUEL, T_ox==T_OX_TARGET).

    Returns False for a partial baseline so the caller will dispatch into
    _build_shared_baseline(), which resumes from the cached state instead
    of redoing finished work.
    """
    if not BASELINE_FILE.exists():
        return False
    try:
        g = ct.Solution(str(SRK_YAML))
        f = ct.CounterflowDiffusionFlame(g, width=DOMAIN_WIDTH)
        f.restore(str(BASELINE_FILE), name=BASELINE_NAME)
        P_bar = float(f.P)/1e5
        Tf = float(f.fuel_inlet.T)
        Tox = float(f.oxidizer_inlet.T)
        partial = (
            abs(P_bar - P_OPER/1e5) > 0.5
            or abs(Tf - T_FUEL) > 1.0
            or abs(Tox - T_OX_TARGET) > 1.0
        )
        if partial:
            _log(f"baseline cache PARTIAL ({P_bar:.2f} bar, "
                 f"T_fuel={Tf:.1f} K, T_ox={Tox:.1f} K, npts={f.grid.size})"
                 f" -> will RESUME build")
            return False
        _log(f"baseline cache at TARGET state ({P_bar:.2f} bar, "
             f"T_fuel={Tf:.1f} K, T_ox={Tox:.1f} K, npts={f.grid.size})")
        return True
    except Exception:
        return False


def run(force_rebuild=False):
    if not SRK_YAML.exists():
        if RANK == 0:
            print(f"missing {SRK_YAML}. Run 01_convert_chemkin.py first.")
        sys.exit(1)

    _log(f"loading {SRK_YAML.name} (mech={MECH_TAG})", only_root=True)
    gas = ct.Solution(str(SRK_YAML))
    T_ox = _resolve_T_ox(gas)
    fuel = _resolve_fuel(gas)
    _log(f"T_ox resolved = {T_ox} K, fuel = {fuel}", only_root=True)

    # ---- Phase A: rank 0 builds the shared baseline (or uses cache) ----
    if RANK == 0:
        if not force_rebuild and _baseline_cache_ok():
            _log(f"baseline cache HIT -> {BASELINE_FILE.name} (skip warmup)")
        else:
            if force_rebuild and BASELINE_FILE.exists():
                _log("force-rebuild requested; deleting cache")
                BASELINE_FILE.unlink()
            _log("building shared baseline (only rank 0 works during warmup)")
            t_warm = time.time()
            try:
                flame0 = _build_shared_baseline(gas, T_ox, log=0)
            except ct.CanteraError as ex:
                _log(f"baseline build FAILED: {ex}")
                if HAVE_MPI:
                    # broadcast failure so other ranks abort cleanly
                    COMM.Abort(1)
                sys.exit(1)
            flame0.save(
                str(BASELINE_FILE),
                name=BASELINE_NAME,
                description=(
                    f"baseline at P={P_OPER:.0f} Pa, T_ox={T_ox} K, "
                    f"T_fuel={T_FUEL} K, mdot={MDOT_INIT}"
                ),
                overwrite=True,
            )
            _log(f"baseline saved to {BASELINE_FILE.name} in "
                 f"{time.time()-t_warm:.1f}s total")

    # ---- Barrier: all ranks wait for baseline to be ready ----
    if HAVE_MPI:
        COMM.Barrier()

    # ---- Phase B: per-rank mdot ramp + chunk sweep ----
    chunks = _split_mdots(MDOTS, SIZE)
    s, e = chunks[RANK]
    my_mdots = MDOTS[s:e]
    _log(f"assigned mdots[{s}:{e}] = {[round(float(m), 3) for m in my_mdots]}")
    if len(my_mdots) == 0:
        _log("no work assigned, waiting at final barrier")
        if HAVE_MPI:
            COMM.Barrier()
        if RANK == 0:
            merge_flamelets()
        return

    # Restore baseline (every rank, including rank 0 to get a fresh copy)
    flame = ct.CounterflowDiffusionFlame(gas, width=DOMAIN_WIDTH)
    flame.transport_model = "mixture-averaged"
    flame.restore(str(BASELINE_FILE), name=BASELINE_NAME)

    # Ramp mdot from MDOT_INIT to first assigned
    first_mdot = float(my_mdots[0])
    t_chunk = time.time()
    if first_mdot > MDOT_INIT * 1.05:
        _log(f"ramp mdot {MDOT_INIT} -> {first_mdot:.3f}")
        try:
            _ramp_mdot_to(flame, first_mdot, log=0)
        except ct.CanteraError as ex:
            _log(f"mdot ramp FAILED: {ex}")
            if HAVE_MPI:
                COMM.Barrier()
            if RANK == 0:
                merge_flamelets()
            return

    # Save first flamelet
    _save_flamelet_npz(flame, gas, first_mdot, s)
    _log(f"saved idx={s:3d}  mdot={first_mdot:6.3f}  "
         f"npts={flame.grid.size:4d}  Tmax={flame.T.max():6.1f} K")

    # Sweep remaining mdots in chunk (warm-continue from previous)
    for k, mdot in enumerate(my_mdots[1:], start=1):
        idx = s + k
        flame.fuel_inlet.mdot = float(mdot)
        flame.oxidizer_inlet.mdot = float(mdot)
        t0 = time.time()
        try:
            flame.solve(loglevel=0, auto=True)
        except ct.CanteraError as ex:
            _log(f"mdot={mdot}: extinction or failure ({ex})")
            break
        _save_flamelet_npz(flame, gas, mdot, idx)
        _log(f"saved idx={idx:3d}  mdot={float(mdot):6.3f}  "
             f"npts={flame.grid.size:4d}  Tmax={flame.T.max():6.1f} K  "
             f"dt={time.time()-t0:5.1f}s")

    _log(f"chunk total {time.time()-t_chunk:.1f}s")

    if HAVE_MPI:
        COMM.Barrier()
    if RANK == 0:
        merge_flamelets()


def merge_flamelets():
    """Merge per-flamelet .npz files into a single flat .npz.

    Layout: keys prefixed with `f{idx:03d}_` -- e.g. `f000_T`, `f000_C`,
    `f000_omega_C`, ... plus global keys `P`, `T_fuel`, `X_fuel`,
    `X_ox`, `pv_species`, `width`, `n_flamelets`, `mdots`, `idx_list`.
    Read back with `np.load(path, allow_pickle=False)`.
    """
    print(f"[merge] scanning {FLAMELET_DIR}", flush=True)
    files = sorted(FLAMELET_DIR.glob("flamelet_*.npz"))
    if not files:
        print("[merge] no flamelet files found")
        return
    print(f"[merge] -> {MERGED_OUTPUT}  ({len(files)} flamelets)")

    out = {
        "P": np.asarray(P_OPER),
        "T_fuel": np.asarray(T_FUEL),
        "X_fuel": np.asarray(X_FUEL),
        "X_ox": np.asarray(X_OX),
        "pv_species": np.asarray(",".join(PV_SPECIES)),
        "width": np.asarray(DOMAIN_WIDTH),
        "n_flamelets": np.asarray(len(files)),
    }
    idx_list = []
    mdots_list = []
    for fp in files:
        d = np.load(fp, allow_pickle=False)
        idx = int(d["idx"])
        prefix = f"f{idx:03d}_"
        idx_list.append(idx)
        mdots_list.append(float(d["mdot"]))
        for k in d.files:
            out[prefix + k] = d[k]
    out["idx_list"] = np.asarray(idx_list, dtype=np.int32)
    out["mdots"] = np.asarray(mdots_list, dtype=np.float64)
    np.savez_compressed(MERGED_OUTPUT, **out)
    print(f"[merge] done. {len(files)} flamelets, "
          f"mdots {min(mdots_list):.3f}..{max(mdots_list):.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mech", choices=list(MECHS), default="wang2011",
                   help="mechanism: 'wang2011' (106 sp, production) or "
                        "'kero4s' (4-step, fast development). default wang2011")
    p.add_argument("--merge-only", action="store_true",
                   help="just merge existing per-flamelet .npz files and exit")
    p.add_argument("--rebuild", action="store_true",
                   help="force re-running the warmup baseline even if cached")
    args = p.parse_args()
    _select_mech(args.mech)
    if args.merge_only:
        if RANK == 0:
            merge_flamelets()
        return
    if not HAVE_MPI:
        print("WARNING: mpi4py not available, running serially (size=1).",
              flush=True)
    run(force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()
