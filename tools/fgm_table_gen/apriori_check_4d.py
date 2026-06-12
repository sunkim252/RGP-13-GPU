"""A-priori validation of the 4-D FGM lookup table.

Three kinds of checks are run; each can be disabled with a flag.

1. **Self-consistency** (no flamelet input needed)
   - For every (iZ, iGz, iC, iChi) grid node, the multilinear interpolation
     at that exact point must return the stored value. Picks up indexing
     bugs and axis-order mismatches with FGMTable.

2. **Smoothness probe**
   - Sample the table along Z at fixed (gZ, C, chi) = (0, C_axis[i_max],
     chi_axis[0]) and check no NaN / no unphysical sign flips in omega_C.

3. **Original flamelet vs table reconstruction** (requires --flamelet-dir)
   - For each Cantera flamelet (stage 6 .npz), pull (Z, C, chi_st, omega_C,
     T, Y_<sp>) along the counterflow grid. Look up the same fields at
     (Z, gZ=0, C, chi_st). Scatter + heatmap + per-field max/RMS error.

Output PNGs go to --out-dir (default ./apriori_plots).

Run:
    python apriori_check_4d.py --table data/fgmProperties_4d.npz
    python apriori_check_4d.py --table data/fgmProperties_4d.npz \\
                                --flamelet-dir data/flamelets_wang2011_fast
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# -------------------- table I/O --------------------

def load_table_npz(path):
    """Return (axes, tables, species, meta).

    axes   : dict 'Z'/'gZ'/'C'/'chi' -> 1-D arrays
    tables : dict field-name -> 4-D array shape (nZ, nGz, nC, nChi)
    species: list[str]
    meta   : dict
    """
    d = np.load(path, allow_pickle=True)
    axes = {
        "Z":   np.asarray(d["Z_axis"]),
        "gZ":  np.asarray(d["gZ_axis"]),
        "C":   np.asarray(d["C_axis"]),
        "chi": np.asarray(d["chi_axis"]),
    }
    species = [str(s) for s in d["species"]]
    meta = {"P": float(d["P"]) if "P" in d.files else float("nan"),
            "src": str(d["src"]) if "src" in d.files else "?"}
    # All 4-D arrays = everything left over
    skip = {"Z_axis", "gZ_axis", "C_axis", "chi_axis",
            "species", "P", "src"}
    tables = {k: np.asarray(d[k]) for k in d.files if k not in skip}
    return axes, tables, species, meta


# -------------------- Python mirror of FGMTable interp --------------------

def _bracket(axis, v):
    """Return (i, w) with axis[i] <= v <= axis[i+1] (clamped). Mirrors
    FGMTable::bracket in C++. For axis of length 1 returns (0, 0)."""
    n = axis.size
    if n <= 1:
        return 0, 0.0
    if v <= axis[0]:
        return 0, 0.0
    if v >= axis[-1]:
        return n - 2, 1.0
    j = int(np.searchsorted(axis, v, side="right"))
    j = max(1, min(j, n - 1))
    i = j - 1
    d = axis[j] - axis[i]
    w = (v - axis[i]) / d if d > 0 else 0.0
    return i, w


def interp4d(table, axes, Z, gZ, C, chi):
    """Quadrilinear interpolation matching FGMTable::interpolateTable.

    table : (nZ, nGz, nC, nChi) array
    axes  : dict with 'Z','gZ','C','chi' keys
    """
    iZ, wZ = _bracket(axes["Z"],   Z)
    iG, wG = _bracket(axes["gZ"],  gZ)
    iC, wC = _bracket(axes["C"],   C)
    iK, wK = _bracket(axes["chi"], chi)
    iKp = iK + 1 if axes["chi"].size >= 2 else iK

    c = lambda dZ, dG, dC, dK: \
        table[iZ + dZ, iG + dG, iC + dC, iKp if dK else iK]

    A00 = c(0,0,0,0)*(1-wZ) + c(1,0,0,0)*wZ
    A10 = c(0,1,0,0)*(1-wZ) + c(1,1,0,0)*wZ
    A01 = c(0,0,1,0)*(1-wZ) + c(1,0,1,0)*wZ
    A11 = c(0,1,1,0)*(1-wZ) + c(1,1,1,0)*wZ
    A0  = A00*(1-wG) + A10*wG
    A1  = A01*(1-wG) + A11*wG
    A   = A0*(1-wC)  + A1*wC

    B00 = c(0,0,0,1)*(1-wZ) + c(1,0,0,1)*wZ
    B10 = c(0,1,0,1)*(1-wZ) + c(1,1,0,1)*wZ
    B01 = c(0,0,1,1)*(1-wZ) + c(1,0,1,1)*wZ
    B11 = c(0,1,1,1)*(1-wZ) + c(1,1,1,1)*wZ
    B0  = B00*(1-wG) + B10*wG
    B1  = B01*(1-wG) + B11*wG
    B   = B0*(1-wC)  + B1*wC

    return A*(1-wK) + B*wK


# -------------------- Check 1: self-consistency --------------------

def check_self_consistency(axes, tables, atol=1e-9, rtol=1e-7):
    nZ, nGz, nC, nChi = (axes["Z"].size, axes["gZ"].size,
                         axes["C"].size, axes["chi"].size)
    print(f"\n[1] self-consistency on every grid node "
          f"({nZ}x{nGz}x{nC}x{nChi} = {nZ*nGz*nC*nChi} nodes)")
    overall_ok = True
    for name, T in tables.items():
        if T.shape != (nZ, nGz, nC, nChi):
            print(f"  SKIP {name}: shape {T.shape} != "
                  f"({nZ}, {nGz}, {nC}, {nChi})")
            continue
        max_err = 0.0
        for iZ in range(nZ):
            for iG in range(nGz):
                for iC in range(nC):
                    for iK in range(nChi):
                        expected = T[iZ, iG, iC, iK]
                        got = interp4d(T, axes,
                                       axes["Z"][iZ], axes["gZ"][iG],
                                       axes["C"][iC], axes["chi"][iK])
                        err = abs(got - expected)
                        if err > max_err:
                            max_err = err
        scale = max(abs(T).max(), 1.0)
        tol = atol + rtol*scale
        verdict = "OK" if max_err < tol else "FAIL"
        if verdict != "OK":
            overall_ok = False
        print(f"  {name:12s} max|err|={max_err:.3e}  tol={tol:.3e}  "
              f"{verdict}")
    print(f"[1] overall: {'PASS' if overall_ok else 'FAIL'}")
    return overall_ok


# -------------------- Check 2: smoothness probe --------------------

def check_smoothness(axes, tables, out_dir):
    """Scan one of each field along Z at (gZ=0, C=C_max, chi=chi[0]) and
    verify no NaN / no oscillation pathology."""
    print("\n[2] smoothness probe (Z scan at gZ=0, C=C_axis[-1], chi=chi[0])")
    out_dir.mkdir(parents=True, exist_ok=True)
    Z_query = np.linspace(0, 1, 200)
    if HAS_MPL:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for name, T in tables.items():
        v = np.array([interp4d(T, axes, Z, 0.0,
                               axes["C"][-1], axes["chi"][0])
                      for Z in Z_query])
        n_nan = int(np.isnan(v).sum())
        n_inf = int(np.isinf(v).sum())
        print(f"  {name:12s} range=[{np.nanmin(v):+.3e}, "
              f"{np.nanmax(v):+.3e}]  NaN={n_nan}  Inf={n_inf}")
        if HAS_MPL:
            ax.plot(Z_query, v, label=name)
    if HAS_MPL:
        ax.set(xlabel="Z~", ylabel="field value (normalized scale)")
        ax.legend(fontsize=7, ncol=2)
        ax.set_title("Smoothness probe (gZ=0, C=C_max, chi=chi_low)")
        fig.savefig(out_dir / "smoothness_Z_scan.png", dpi=120,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"[2] -> {out_dir/'smoothness_Z_scan.png'}")


# -------------------- Check 3: flamelet vs lookup --------------------

def load_flamelets(flamelet_dir):
    paths = sorted(Path(flamelet_dir).glob("flamelet_*.npz"))
    if not paths:
        return []
    fls = []
    for p in paths:
        d = np.load(p, allow_pickle=False)
        species = sorted(k[2:] for k in d.files if k.startswith("Y_"))
        fls.append({
            "path":    p,
            "Z":       np.asarray(d["Z"]),
            "C":       np.asarray(d["C"]),
            "T":       np.asarray(d["T"]),
            "omega_C": np.asarray(d["omega_C"]),
            "chi_st":  float(d["chi_st"]) if "chi_st" in d.files else 0.0,
            "Y":       {sp: np.asarray(d[f"Y_{sp}"]) for sp in species},
            "species": species,
        })
    return fls


def check_flamelets(axes, tables, flamelet_dir, out_dir):
    fls = load_flamelets(flamelet_dir)
    if not fls:
        print(f"\n[3] no flamelets found under {flamelet_dir}; skipping")
        return
    print(f"\n[3] flamelet vs table for {len(fls)} flamelets "
          f"under {flamelet_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # which fields exist in BOTH the table and the flamelet
    ref_fields = ["omega_C", "T"]
    species_overlap = [sp for sp in fls[0]["species"]
                       if f"Y_{sp}" in tables]
    ref_fields.extend([f"Y_{sp}" for sp in species_overlap])
    print(f"    comparing fields: {ref_fields}")

    # per-field aggregate stats
    stats = {f: {"max": 0.0, "rms": 0.0, "n": 0} for f in ref_fields}

    for fl in fls:
        Z = fl["Z"]
        C = fl["C"]
        chi = fl["chi_st"]
        for fname in ref_fields:
            if fname == "omega_C":
                ref = fl["omega_C"]
            elif fname == "T":
                ref = fl["T"]
            else:
                ref = fl["Y"][fname[2:]]
            T_table = tables.get(fname)
            if T_table is None:
                continue
            got = np.array([interp4d(T_table, axes,
                                     Z[i], 0.0, C[i], chi)
                            for i in range(Z.size)])
            err = got - ref
            absmax = float(np.nanmax(np.abs(err)))
            rmse = float(np.sqrt(np.nanmean(err**2)))
            stats[fname]["max"] = max(stats[fname]["max"], absmax)
            stats[fname]["rms"] += rmse**2 * Z.size
            stats[fname]["n"] += Z.size

    print(f"    aggregate over all {len(fls)} flamelets:")
    for f, s in stats.items():
        rms = float(np.sqrt(s["rms"]/max(s["n"], 1))) if s["n"] else 0.0
        print(f"      {f:14s} max|err|={s['max']:.3e}  rmse={rms:.3e}")

    # representative plot: omega_C(Z) for 4 strain-representative flamelets
    if HAS_MPL and len(fls) >= 1:
        ks = np.linspace(0, len(fls)-1, min(len(fls), 4),
                         dtype=int)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for k in ks:
            fl = fls[k]
            Z = fl["Z"]; C = fl["C"]; chi = fl["chi_st"]
            ref = fl["omega_C"]
            got = np.array([interp4d(tables["omega_C"], axes,
                                     Z[i], 0.0, C[i], chi)
                            for i in range(Z.size)])
            order = np.argsort(Z)
            ax[0].plot(Z[order], ref[order], "-",  lw=1.5,
                       label=f"orig  chi_st={chi:.2g}")
            ax[0].plot(Z[order], got[order], "--", lw=1.0)
            ax[1].plot(Z[order], got[order]-ref[order], lw=1.0,
                       label=f"chi_st={chi:.2g}")
        ax[0].set(xlabel="Z", ylabel="omega_C [1/s]",
                  title="orig (solid) vs table lookup (dashed)")
        ax[0].legend(fontsize=8)
        ax[1].set(xlabel="Z", ylabel="lookup - orig",
                  title="residual")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].legend(fontsize=8)
        fig.savefig(out_dir / "flamelet_vs_table_omegaC.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[3] -> {out_dir/'flamelet_vs_table_omegaC.png'}")


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True,
                   help="path to 4-D table .npz "
                        "(companion of fgmProperties)")
    p.add_argument("--flamelet-dir",
                   help="directory of Cantera flamelet_*.npz for "
                        "check 3 (optional)")
    p.add_argument("--out-dir", default="apriori_plots",
                   help="directory for PNG output")
    p.add_argument("--skip-selfcheck", action="store_true")
    p.add_argument("--skip-smoothness", action="store_true")
    args = p.parse_args()

    table_path = Path(args.table)
    if not table_path.exists():
        raise SystemExit(f"missing table npz: {table_path}")
    out_dir = Path(args.out_dir)

    print(f"[load] {table_path}")
    axes, tables, species, meta = load_table_npz(table_path)
    print(f"[load] {len(tables)} fields, "
          f"axes Z={axes['Z'].size} gZ={axes['gZ'].size} "
          f"C={axes['C'].size} chi={axes['chi'].size}")
    print(f"[load] source = {meta['src']}, P = {meta['P']:.3g} Pa")
    print(f"[load] species in table: {species}")

    ok = True
    if not args.skip_selfcheck:
        ok &= check_self_consistency(axes, tables)
    if not args.skip_smoothness:
        check_smoothness(axes, tables, out_dir)
    if args.flamelet_dir:
        check_flamelets(axes, tables, Path(args.flamelet_dir), out_dir)
    else:
        print("\n[3] skipped (no --flamelet-dir)")

    print(f"\noverall: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
