"""Append Tier-4 differential-diffusion Lewis-number tables (Le_Z, Le_C) to an
EXISTING fgmProperties dictionary + .npz companion, without regenerating the
flamelets: the Lewis numbers are pure functions of each node's tabulated (T, Y)
at the table pressure, evaluated with the real-fluid (SRK + high-pressure-
Chung/Takahashi) Cantera transport via compute_lewis_tables() from the 4-D
builder (bit-identical to what a fresh --diff-diff-yaml build would produce).

Works for both the 3-D (Z,gZ,c) and 4-D non-adiabatic (Z,gZ,c,dh) layouts --
the node loop is layout-agnostic (flat C-order matches FGMTable).

The dictionary gets:
  * flat Le_Z / Le_C blocks (length nTot) appended before the trailing banner,
  * the constant Le { Z ...; C ...; } fallback values replaced by the manifold
    medians (older solvers keep working; the Tier-4 solver uses the fields),
  * a one-shot .preLe backup of dict and npz (never overwritten).

Usage:
  python3 add_lewis_tables.py --dict data/fgm_nadiab_P525_cold \
      --yaml data/wang2011_srk_v32.yaml [--npz auto] \
      [--tmin-skip 85] [--tmin-eval 85]

For cold (non-adiabatic, cryogenic-inlet) tables pass low --tmin-* so the cold
dense LOX nodes are evaluated at their REAL temperature (Pr rises to ~2-4 in
the compressed liquid) instead of being median-filled.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# import compute_lewis_tables from the digit-prefixed builder module
_spec = importlib.util.spec_from_file_location(
    "fgm4d", HERE / "04_build_fgm_table_4d.py")
_fgm4d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fgm4d)
compute_lewis_tables = _fgm4d.compute_lewis_tables


def _fmt(arr):
    return "\n".join(f"    {v:.8e}" for v in arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", dest="dictfile", required=True,
                    help="fgmProperties dictionary to extend in place")
    ap.add_argument("--npz", default="auto",
                    help="companion .npz (default: <dict>.npz)")
    ap.add_argument("--yaml", dest="srk_yaml", required=True,
                    help="SRK Cantera mechanism (e.g. wang2011_srk_v32.yaml)")
    ap.add_argument("--transport", default="high-pressure-Chung")
    ap.add_argument("--tmin-skip", type=float, default=85.0,
                    help="nodes below this T are median-filled (default 85)")
    ap.add_argument("--tmin-eval", type=float, default=85.0,
                    help="evaluation T floor passed to Cantera (default 85)")
    args = ap.parse_args()

    dpath = Path(args.dictfile)
    npath = Path(args.npz) if args.npz != "auto" \
        else dpath.with_name(dpath.name + ".npz")
    if not dpath.is_file():
        raise SystemExit(f"missing dict {dpath}")
    if not npath.is_file():
        raise SystemExit(f"missing npz companion {npath} "
                         "(needed for the node (T,Y) arrays)")

    d = np.load(npath, allow_pickle=True)
    species = [str(s) for s in d["species"]]
    P = float(d["P"])
    tables = {"T": np.asarray(d["T"], float)}
    for sp in species:
        tables[f"Y_{sp}"] = np.asarray(d[f"Y_{sp}"], float)
    shape = tables["T"].shape
    nTot = tables["T"].size
    print(f"[add-le] {dpath.name}: shape {shape} = {nTot} nodes, "
          f"P = {P/1e5:.2f} bar, {len(species)} species, "
          f"T = [{tables['T'].min():.0f}, {tables['T'].max():.0f}] K")

    if "Le_Z" in d.files:
        raise SystemExit("[add-le] npz already carries Le_Z -- refusing to "
                         "double-apply (restore the .preLe backup first)")

    t0 = time.time()
    LeZ, LeC, medZ, medC = compute_lewis_tables(
        tables, species, P, args.srk_yaml,
        transport_model=args.transport,
        Tmin_skip=args.tmin_skip, Tmin_eval=args.tmin_eval,
    )
    print(f"[add-le] Lewis evaluation done in {time.time()-t0:.0f}s")

    # ---- one-shot backups ----
    for src in (dpath, npath):
        bak = src.with_name(src.name + ".preLe")
        if not bak.exists():
            shutil.copy2(src, bak)
            print(f"[add-le] backup {bak.name}")

    # ---- extend the dictionary text ----
    txt = dpath.read_text()
    if re.search(r"^Le_Z$", txt, flags=re.M):
        raise SystemExit("[add-le] dict already has an Le_Z block")

    # constant-Le fallback -> manifold medians (Z and C only; h untouched)
    def _sub_le(txt, var, val):
        pat = re.compile(r"(^Le\s*\{[^}]*?\b" + var + r"\s+)[-0-9.eE]+(\s*;)",
                         flags=re.M | re.S)
        new, n = pat.subn(lambda m: f"{m.group(1)}{val:.6g}{m.group(2)}", txt)
        if n != 1:
            print(f"[add-le] WARNING: constant Le.{var} not updated "
                  f"(matched {n} times) -- fields still take precedence")
            return txt
        return new

    txt = _sub_le(txt, "Z", medZ)
    txt = _sub_le(txt, "C", medC)

    block = (
        "\n// Tier-4 differential diffusion (add_lewis_tables.py): per-node\n"
        f"// real-fluid ({args.transport}) Lewis numbers on the manifold.\n"
        "// Le_Z = Pr = nu/alpha (Z diffuses like heat);  Le_C = nu/D_C with\n"
        "// D_C = PV-mass-weighted mixture-averaged diffusivity. The solver\n"
        "// uses these per-cell; the constant Le sub-dict is the median\n"
        "// fallback for pre-Tier-4 solvers.\n"
        f"\nLe_Z\n(\n{_fmt(LeZ.reshape(-1))}\n);\n"
        f"\nLe_C\n(\n{_fmt(LeC.reshape(-1))}\n);\n"
    )

    # insert before the trailing banner comment if present, else append
    m = re.search(r"\n// \*+ //\s*$", txt)
    if m:
        txt = txt[:m.start()] + block + txt[m.start():]
    else:
        txt = txt + block
    dpath.write_text(txt)
    print(f"[add-le] dict updated: +Le_Z/+Le_C ({nTot} entries each), "
          f"constant fallback Le(Z={medZ:.3f}, C={medC:.3f})")

    # ---- extend the npz companion ----
    arrs = {k: d[k] for k in d.files}
    arrs["Le_Z"] = np.ascontiguousarray(LeZ)
    arrs["Le_C"] = np.ascontiguousarray(LeC)
    np.savez_compressed(npath, **arrs)
    print(f"[add-le] npz updated: {npath.name}")

    # ---- light self-check: block lengths in the rewritten dict ----
    txt2 = dpath.read_text()
    for key in ("Le_Z", "Le_C"):
        m = re.search(rf"^{key}$\n\(\n(.*?)\n\);", txt2, flags=re.M | re.S)
        ncnt = len(m.group(1).splitlines()) if m else -1
        tag = "OK" if ncnt == nTot else "MISMATCH"
        print(f"[check] {key}: {ncnt} entries (expect {nTot}) {tag}")
        if ncnt != nTot:
            raise SystemExit("[add-le] self-check FAILED -- restore .preLe")

    print("[add-le] done.")


if __name__ == "__main__":
    main()
