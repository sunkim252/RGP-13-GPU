"""Build the FGM (FPV + presumed beta-PDF) lookup table for OpenFOAM.

Input : a merged flamelet library .npz (from 03_flamelet_sweep_mpi.py or
        make_synthetic_flamelets.py). Each flamelet provides Z, C and one or
        more state fields: omega_C (PV source), optionally T, and species
        mass fractions Y_<sp>.

Output: constant/fgmProperties (OpenFOAM dictionary) with 3-D tables of every
        tabulated quantity as a function of (Z~, gZ, C~), where
        gZ = Zvar/(Z~(1-Z~)) is the mixture-fraction segregation factor.

        Keys written:
            nZ, nGz, nC ; Z, gZ, C axes ;
            sourcePV   (Favre-filtered omega_C)
            T          (Favre-filtered temperature, if present)
            species (...)            list of tabulated species
            Y_<sp>     (Favre-filtered mass fraction of each species)
        All 3-D arrays are flat, C-order  idx = (iZ*nGz + iGz)*nC + iC.

R2 coupling: the OpenFOAM-side FGM model reconstructs the composition Y_<sp>
from this table and feeds the real-fluid (SRK + Chung-Takahashi / Ely-Hanley)
thermo, which then computes rho, T, mu, lambda.

Run:
    python 04_build_fgm_table.py --in data/flamelets_synth.npz
    python 04_build_fgm_table.py --in data/flamelets_wang2011.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import griddata
    from scipy.special import gammaln
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

# ---- Table axes resolution ----
N_Z = 51
N_G = 11
N_C = 41
N_ZQ = 200
G_MAX = 0.9
EPS = 1.0e-6


# ----------------------------- IO -----------------------------

def load_flamelets(path):
    """Return (Zs, Cs, fields, species, meta).

    fields: dict name -> list of per-flamelet arrays. Always contains
            'omega_C'; may contain 'T' and 'Y_<sp>'.
    species: list of species names that have Y_<sp> data.
    """
    d = np.load(path, allow_pickle=False)
    n = int(d["n_flamelets"])
    idx_list = d["idx_list"].tolist() if "idx_list" in d.files else list(range(n))

    # discover available per-flamelet field suffixes from flamelet 0
    pre0 = f"f{int(idx_list[0]):03d}_"
    suffixes = [k[len(pre0):] for k in d.files if k.startswith(pre0)]
    species = sorted(s[2:] for s in suffixes if s.startswith("Y_"))
    tab_fields = ["omega_C"] + (["T"] if "T" in suffixes else []) \
        + [f"Y_{s}" for s in species]

    Zs, Cs = [], []
    fields = {f: [] for f in tab_fields}
    for i in idx_list:
        pre = f"f{int(i):03d}_"
        if pre + "Z" not in d.files:
            continue
        Zs.append(np.asarray(d[pre + "Z"], float))
        Cs.append(np.asarray(d[pre + "C"], float))
        for f in tab_fields:
            fields[f].append(np.asarray(d[pre + f], float))

    meta = {
        "P": float(d["P"]) if "P" in d.files else float("nan"),
        "pv_species": str(d["pv_species"]) if "pv_species" in d.files else "",
        "n_flamelets": len(Zs),
    }
    return Zs, Cs, fields, species, meta


# ----------------------- laminar manifold -----------------------

def _laminar(Zs, Cs, Ws, Zq, Cgrid, nonneg=True):
    """Interpolate scattered (Z, C, W) onto (Zq x Cgrid). Shape (nZq, nC).

    Flamelet data only populates the curved region C in [0, C_eq(Z)].
    Inside the convex hull we use linear interpolation; OUTSIDE the hull
    (e.g. C > C_eq(Z)) we clamp to the nearest valid sample (the local
    equilibrium / mixing state) rather than zero, so composition tables keep
    sum(Y)=1 and omega_C stays ~0 in the burnt-out region.
    """
    Zp = np.concatenate(Zs)
    Cp = np.concatenate(Cs)
    Wp = np.concatenate(Ws)
    ZZ, CC = np.meshgrid(Zq, Cgrid, indexing="ij")
    if HAVE_SCIPY:
        out = griddata((Zp, Cp), Wp, (ZZ, CC), method="linear")
        nan = np.isnan(out)
        if nan.any():
            out[nan] = griddata(
                (Zp, Cp), Wp, (ZZ[nan], CC[nan]), method="nearest"
            )
    else:
        out = np.zeros_like(ZZ)
        for i in range(ZZ.shape[0]):
            for j in range(ZZ.shape[1]):
                k = np.argmin((Zp - ZZ[i, j])**2 + (Cp - CC[i, j])**2)
                out[i, j] = Wp[k]
    return np.maximum(out, 0.0) if nonneg else out


# --------------------------- beta-PDF ---------------------------

def beta_pdf_weights(Zmean, g, Zq):
    w = np.zeros_like(Zq)
    if g < 1.0e-4 or Zmean <= EPS or Zmean >= 1.0 - EPS:
        w[np.argmin(np.abs(Zq - Zmean))] = 1.0
        return w
    inv = 1.0 / g - 1.0
    a = Zmean * inv
    b = (1.0 - Zmean) * inv
    Zc = np.clip(Zq, EPS, 1.0 - EPS)
    if HAVE_SCIPY:
        logw = (a - 1.0)*np.log(Zc) + (b - 1.0)*np.log(1.0 - Zc) \
            - (gammaln(a) + gammaln(b) - gammaln(a + b))
        w = np.exp(logw - logw.max())
    else:
        w = Zc**(a - 1.0) * (1.0 - Zc)**(b - 1.0)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        w = np.zeros_like(Zq)
        w[np.argmin(np.abs(Zq - Zmean))] = 1.0
        return w
    return w / s


# --------------------------- assemble ---------------------------

def build_tables(Zs, Cs, fields):
    Z_axis = np.linspace(0.0, 1.0, N_Z)
    g_axis = np.linspace(0.0, G_MAX, N_G)
    C_global_max = max(np.max(c) for c in Cs)
    C_axis = np.linspace(0.0, C_global_max, N_C)
    Zq = np.linspace(0.0, 1.0, N_ZQ)

    print(f"[build] axes nZ={N_Z} nGz={N_G} nC={N_C}  C_max={C_global_max:.4f}")

    # Laminar manifold per field
    t0 = time.time()
    lam = {f: _laminar(Zs, Cs, fields[f], Zq, C_axis) for f in fields}
    print(f"[build] {len(lam)} laminar manifolds in {time.time()-t0:.1f}s "
          f"({list(lam)})")

    # Precompute beta-PDF weights once: W[iZ, iG, :]
    t0 = time.time()
    W = np.zeros((N_Z, N_G, N_ZQ))
    for iZ, Zm in enumerate(Z_axis):
        for iG, g in enumerate(g_axis):
            W[iZ, iG, :] = beta_pdf_weights(Zm, g, Zq)
    print(f"[build] beta weights in {time.time()-t0:.1f}s")

    # Convolve: table[f][iZ,iG,iC] = sum_q W[iZ,iG,q] * lam[f][q, iC]
    tables = {}
    for f, L in lam.items():
        tables[f] = np.einsum("zgq,qc->zgc", W, L)
    return Z_axis, g_axis, C_axis, tables


# --------------------------- writer ---------------------------

def _fmt(arr):
    return "\n".join(f"    {v:.8e}" for v in arr)


def write_fgm_dict(path, Z_axis, g_axis, C_axis, tables, species, meta):
    nZ, nGz, nC = len(Z_axis), len(g_axis), len(C_axis)

    blocks = []
    # PV source
    blocks.append(f"sourcePV\n(\n{_fmt(tables['omega_C'].reshape(-1))}\n);")
    # Temperature (optional)
    if "T" in tables:
        blocks.append(f"T\n(\n{_fmt(tables['T'].reshape(-1))}\n);")
    # Species
    if species:
        blocks.append("species ( " + " ".join(species) + " );")
        for sp in species:
            blocks.append(
                f"Y_{sp}\n(\n{_fmt(tables[f'Y_{sp}'].reshape(-1))}\n);"
            )
    body = "\n\n".join(blocks)

    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| FGM (FPV + beta-PDF) lookup table  --  04_build_fgm_table.py                |
| source flamelets: {meta.get('src','?'):<55}|
| P = {meta.get('P', float('nan')):.6g} Pa                                                       |
| axes: Z~ (nZ={nZ}), gZ=Zvar/(Z~(1-Z~)) (nGz={nGz}), C~ (nC={nC})                |
| flat C-order: idx = (iZ*nGz + iGz)*nC + iC                                  |
| tabulated: sourcePV (omega_C), T, Y_<species> (for SRK+Chung/Ely thermo)   |
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

Z
(
{_fmt(Z_axis)}
);

gZ
(
{_fmt(g_axis)}
);

C
(
{_fmt(C_axis)}
);

{body}

// ************************************************************************* //
"""
    path.write_text(header)
    print(f"[write] {path}")
    print(f"[write]   fields: sourcePV"
          + (", T" if "T" in tables else "")
          + (", Y_{" + ",".join(species) + "}" if species else ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile",
                   default=str(DATA_DIR / "flamelets_synth.npz"))
    p.add_argument("--out", dest="outfile",
                   default=str(DATA_DIR / "fgmProperties"))
    args = p.parse_args()

    infile = Path(args.infile)
    if not infile.exists():
        raise SystemExit(f"missing {infile}")
    print(f"[load] {infile}")
    Zs, Cs, fields, species, meta = load_flamelets(infile)
    meta["src"] = infile.name
    print(f"[load] {meta['n_flamelets']} flamelets; "
          f"fields={list(fields)}; species={species}")

    Z_axis, g_axis, C_axis, tables = build_tables(Zs, Cs, fields)
    for f, T in tables.items():
        print(f"[build]   {f}: range [{T.min():.3g}, {T.max():.3g}]")

    write_fgm_dict(Path(args.outfile), Z_axis, g_axis, C_axis,
                   tables, species, meta)


if __name__ == "__main__":
    main()
