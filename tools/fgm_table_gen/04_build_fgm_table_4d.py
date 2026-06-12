"""Build the 4-axis FGM (FPV + presumed beta-PDF + chi_st) lookup table.

This is the 4-D extension of 04_build_fgm_table.py. The fourth axis is the
scalar dissipation rate evaluated at Z = Z_st (chi_st in 1/s), which lets the
LES side distinguish flamelets sampled at different strain rates (Ma-Hickey-
Ihme 2018 style steady-chi FPV; full Ihme-Pitsch 2008 unsteady FPV is a
follow-up).

Input  : directory of per-flamelet npz files produced by stage 6 of
         05_flamelet_sweep_fast.py (each carries z, Z, T, rho, lam, mu, cp,
         alpha, chi, chi_st, C, omega_C, Y_<sp> and a scalar 'chi_st' from
         _chi_at_Zst).

Output : a single 4-D fgmProperties dictionary readable by FGMTable, with
         axes (Z, gZ, C, chi). The flat C-order index for any field is

             idx = ((iZ*nGz + iGz)*nC + iC)*nChi + iChi

Run    :
    python 04_build_fgm_table_4d.py \\
           --flamelet-dir data/flamelets_wang2011_fast \\
           --out          data/fgmProperties_4d
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
N_CHI = 9          # number of chi_st grid points (log-spaced)
N_ZQ = 200          # internal beta-PDF integration mesh
G_MAX = 0.9
CHI_MIN_DEFAULT = 1.0e-2   # fallback if every flamelet has chi_st = 0
EPS = 1.0e-6

# ---- Equilibrium burnt-end closure / normalized progress variable ----
# The C axis of the table is the NORMALIZED progress variable
#     c = Yc / Yc_eq(Z)  in [0, 1],
# where Yc_eq(Z) is the ADIABATIC CHEMICAL EQUILIBRIUM value of the raw
# progress variable (sum of PV_SPECIES mass fractions) at each mixture
# fraction -- NOT complete combustion / Burke-Schumann (which would be a
# non-physical ~6000 K state) and NOT the lowest-strain flamelet (whose
# omega_C stays positive, which is exactly what let the transported C run
# away in the solver). At c = 1 the source vanishes by construction
# (equilibrium <=> net production = 0), so the transported c is bounded.
# Refs: Pierce thesis (equilibrium = chi->0 limit of the steady-flamelet
# family); van Oijen & de Goey 2000 (table must match the equilibrium
# composition exactly at the burnt boundary); Sharma et al. 2023 PoF Eq.(22)
# (c = Yc/Yc_eq, function of local Z only -- supercritical LOx/CH4 FGM).
# The source is transformed as omega_c = omega_C / Yc_eq(Z); cross-terms of
# the exact normalized-c transport equation are neglected (standard
# practice). Where Yc_eq < CEQ_MIN (pure streams Z->0,1) omega_c := 0.
PV_SPECIES = ("CO2", "CO", "H2O", "H2")   # must match 05_* flamelet C
X_FUEL_SURROGATE = "NC10H22:0.74, PHC3H7:0.15, CYC9H18:0.11"
X_OX = "O2:1.0"
CEQ_MIN = 1.0e-4

DEFAULT_YAML = DATA_DIR / "wang2011_ideal_v32.yaml"


def equilibrium_closure(yaml_file, Zq, P, T_fuel, T_ox, species):
    """Adiabatic-equilibrium and frozen-mixing states on the Zq grid.

    For each mixture fraction Z: the unburnt mixture is the mass-weighted
    blend of the fuel and oxidizer streams, its enthalpy the blend of the
    stream enthalpies (adiabatic mixing at constant P). The frozen state
    (c=0 boundary) is that mixture at its mixing temperature; the burnt
    state (c=1 boundary) is its constant-(H,P) chemical equilibrium, where
    by definition omega_C = 0. Returns C_eq, T_eq, Y_eq, T_u, Y_u."""
    import cantera as ct
    g = ct.Solution(str(yaml_file))
    g.TPX = T_fuel, P, X_FUEL_SURROGATE
    Yf, hf = g.Y.copy(), g.enthalpy_mass
    g.TPX = T_ox, P, X_OX
    Yo, ho = g.Y.copy(), g.enthalpy_mass
    pv_idx = [g.species_index(s) for s in PV_SPECIES
              if s in g.species_names]
    sp_idx = {sp: g.species_index(sp) for sp in species}
    n = len(Zq)
    C_eq = np.zeros(n); T_eq = np.zeros(n); T_u = np.zeros(n)
    Y_eq = {sp: np.zeros(n) for sp in species}
    Y_u = {sp: np.zeros(n) for sp in species}
    for i, Z in enumerate(Zq):
        Ymix = Z * Yf + (1.0 - Z) * Yo
        h = Z * hf + (1.0 - Z) * ho
        g.HPY = h, P, Ymix                     # frozen mixing state
        T_u[i] = g.T
        for sp in species:
            Y_u[sp][i] = Ymix[sp_idx[sp]]
        g.equilibrate("HP")                    # adiabatic equilibrium
        T_eq[i] = g.T
        C_eq[i] = float(sum(g.Y[k] for k in pv_idx))
        for sp in species:
            Y_eq[sp][i] = g.Y[sp_idx[sp]]
    print(f"[eq] equilibrium closure on {n} Z points: "
          f"C_eq max={C_eq.max():.4f} @Z={Zq[np.argmax(C_eq)]:.3f}, "
          f"T_eq max={T_eq.max():.0f} K  (P={P/1e5:.1f} bar, "
          f"T_f={T_fuel:.0f}/T_ox={T_ox:.0f} K)")
    return {"C_eq": C_eq, "T_eq": T_eq, "Y_eq": Y_eq,
            "T_u": T_u, "Y_u": Y_u}


# ----------------------------- IO -----------------------------

def load_flamelet_dir(dir_path):
    """Return a list of per-flamelet dicts, each with the fields we need.

    The per-flamelet npz produced by 05_flamelet_sweep_fast._save_flamelet
    contains scalar metadata (chi_st, P, ...) and 1-D arrays along the
    counterflow grid. We pull just what the 4-D builder needs."""
    paths = sorted(Path(dir_path).glob("flamelet_*.npz"))
    if not paths:
        raise SystemExit(f"no flamelet_*.npz under {dir_path}")
    fls = []
    species = None
    for p in paths:
        d = np.load(p, allow_pickle=False)
        names = set(d.files)
        sp = sorted(k[2:] for k in names if k.startswith("Y_"))
        if species is None:
            species = sp
        elif species != sp:
            # different species lists between files would corrupt the table
            raise SystemExit(f"species mismatch in {p.name}: "
                             f"{sp[:5]}... vs first {species[:5]}...")
        Y = {sp: np.asarray(d[f"Y_{sp}"], float) for sp in species}
        # Composition hygiene: time-stepped solutions accepted without a
        # declared steady state carry ISOLATED unconverged spikes (observed:
        # 1-11 of 152 points with sum(Y) up to ~5). Mask those points out of
        # the cloud and renormalize the rest pointwise; the manifold loses
        # nothing (median |sumY-1| = 0 elsewhere).
        sumY = np.zeros_like(np.asarray(d["T"], float))
        for sp in species:
            sumY += Y[sp]
        keep = np.abs(sumY - 1.0) <= 0.05
        nbad = int((~keep).sum())
        if nbad:
            print(f"[load]   {p.name}: masked {nbad} unconverged point(s) "
                  f"(max|sumY-1|={np.abs(sumY-1).max():.2f})")
        s = np.maximum(sumY[keep], 1e-12)
        fls.append({
            "path":   p,
            "Z":      np.asarray(d["Z"], float)[keep],
            "C":      np.asarray(d["C"], float)[keep] / s,
            "T":      np.asarray(d["T"], float)[keep],
            "omega_C": np.asarray(d["omega_C"], float)[keep],
            "chi_st": float(d["chi_st"]),
            "mdot":   float(d["mdot"]),
            "P":      float(d["P"]),
            "Tmax":   float(np.asarray(d["T"], float)[keep].max()),
            "Y":      {sp: Y[sp][keep] / s for sp in species},
        })
    return fls, species


# ----------------- chi_st grid + flamelet ranking -----------------

def build_chi_axis(fls, n_chi=N_CHI):
    """Pick a log-spaced chi axis that spans the actual range of chi_st
    across the loaded flamelets, with a small safety pad so the LES side
    can clamp outside-range queries to the endpoints."""
    chi_obs = np.array([fl["chi_st"] for fl in fls])
    chi_obs = chi_obs[chi_obs > 0]
    if chi_obs.size == 0:
        # All chi_st = 0 (shouldn't happen, but be defensive).
        return np.geomspace(CHI_MIN_DEFAULT, 100.0, n_chi)
    chi_lo = max(chi_obs.min() * 0.5, CHI_MIN_DEFAULT)
    chi_hi = chi_obs.max() * 2.0
    return np.geomspace(chi_lo, chi_hi, n_chi)


def flamelet_at_chi(fls, chi_target):
    """Return the (Z, C, fields) of the flamelet with the closest chi_st.

    Simpler than linear interpolation between two neighbours and avoids
    artefacts near extinction (where the burning branch ends abruptly).
    We can swap in a 2-flamelet linear interp later if the table looks
    blocky in the chi direction."""
    chis = np.array([fl["chi_st"] for fl in fls])
    k = int(np.argmin(np.abs(np.log(np.maximum(chis, EPS))
                              - np.log(max(chi_target, EPS)))))
    return fls[k]


# ----------------------- laminar manifold -----------------------

def _laminar_one(Z, C, W, Zq, Cgrid, nonneg=True):
    """Interpolate ONE flamelet's scattered (Z, C, W) onto (Zq x Cgrid).

    Same algorithm as the 3-D builder's _laminar but takes just one
    flamelet's arrays at a time (since each chi-axis point gets its own
    flamelet picked by flamelet_at_chi)."""
    ZZ, CC = np.meshgrid(Zq, Cgrid, indexing="ij")
    if HAVE_SCIPY:
        out = griddata((Z, C), W, (ZZ, CC), method="linear")
        nan = np.isnan(out)
        if nan.any():
            out[nan] = griddata(
                (Z, C), W, (ZZ[nan], CC[nan]), method="nearest"
            )
    else:
        out = np.zeros_like(ZZ)
        for i in range(ZZ.shape[0]):
            for j in range(ZZ.shape[1]):
                k = np.argmin((Z - ZZ[i, j])**2 + (C - CC[i, j])**2)
                out[i, j] = W[k]
    return np.maximum(out, 0.0) if nonneg else out


# --------------------------- beta-PDF ---------------------------

def beta_pdf_weights(Zmean, g, Zq):
    """Reproduces 04_build_fgm_table.beta_pdf_weights exactly."""
    w = np.zeros_like(Zq)
    if g < 1.0e-4 or Zmean <= EPS or Zmean >= 1.0 - EPS:
        w[np.argmin(np.abs(Zq - Zmean))] = 1.0
        return w
    inv = 1.0 / g - 1.0
    a = Zmean * inv
    b = (1.0 - Zmean) * inv
    Zc = np.clip(Zq, EPS, 1.0 - EPS)
    if HAVE_SCIPY:
        logw = ((a - 1.0)*np.log(Zc) + (b - 1.0)*np.log(1.0 - Zc)
                - (gammaln(a) + gammaln(b) - gammaln(a + b)))
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

def _norm_envelope(fls, eq, Zq, species):
    """Normalization denominator C_norm(Z) and the c=1 boundary state.

    C_norm(Z) = max( C_eq(Z), max over the steady-flamelet FAMILY of C(Z) )
    scaled by (1+1e-3) so every flamelet point lands strictly below c = 1.

    Why not C_eq alone: in a diffusion flame, cross-Z diffusion imports
    high-C fluid from richer mixture fractions (C_eq peaks near Z~0.45, not
    at Z_st), so the steady flamelet C(Z) exceeds the LOCAL equilibrium by a
    few % around stoichiometry even with unity-Le. Normalizing by C_eq alone
    put the ENTIRE reaction zone at c > 1 and dropping those points hollowed
    the source (-> extinction). The manifold envelope is Pierce (2004)'s
    library truncation expressed as a normalization: every realizable steady
    state maps into c in [0,1), and the c=1 row carries the closure states
    (the envelope owner's state) with omega := 0, which keeps the
    transported c bounded (the boundedness requirement) while the interior
    keeps the full flamelet source."""
    C_eq = eq["C_eq"]
    nf = len(fls)
    C_f = np.zeros((nf, N_ZQ))
    T_f = np.zeros((nf, N_ZQ))
    Y_f = {sp: np.zeros((nf, N_ZQ)) for sp in species}
    for i, fl in enumerate(fls):
        o = np.argsort(fl["Z"])
        Zs = fl["Z"][o]
        C_f[i] = np.interp(Zq, Zs, fl["C"][o])
        T_f[i] = np.interp(Zq, Zs, fl["T"][o])
        for sp in species:
            Y_f[sp][i] = np.interp(Zq, Zs, fl["Y"][sp][o])
    C_fam = C_f.max(axis=0)
    owner = C_f.argmax(axis=0)
    eq_owns = C_eq >= C_fam
    C_norm = np.maximum(np.maximum(C_eq, C_fam), CEQ_MIN) * (1.0 + 1.0e-3)
    # c=1 boundary state: equilibrium where it owns the envelope, else the
    # owning flamelet's local state; omega = 0 on the whole row regardless.
    T1 = np.where(eq_owns, eq["T_eq"], T_f[owner, np.arange(N_ZQ)])
    Y1 = {sp: np.where(eq_owns, eq["Y_eq"][sp],
                       Y_f[sp][owner, np.arange(N_ZQ)])
          for sp in species}
    print(f"[build] C_norm envelope: eq owns {100*eq_owns.mean():.0f}% of Z, "
          f"family/eq max ratio={np.max(C_fam/np.maximum(C_eq,CEQ_MIN)):.3f}")
    return C_norm, T1, Y1


def build_tables(fls, species, n_chi=N_CHI, eq=None, chi_mode=False):
    """Assemble the laminar manifold and beta-PDF-convolve it.

    chi_mode=False (DEFAULT, steady 3-D FPV): the (Z, c) plane is filled by
    the ENTIRE steady-flamelet family at once -- along the burning branch the
    progress variable and chi_st index the same one-parameter family, so a
    chi axis adds no information for steady manifolds and a per-chi
    single-flamelet fill leaves the c direction degenerate (the source
    hollowed out, observed). Returns chi_axis=None -> 3-D table; the solver's
    FGMTable falls back to its 3-D path automatically.

    chi_mode=True (future UFPV): per-chi-slice fill, kept for when UNSTEADY
    flamelets (extinction/reignition transients, Ihme & Pitsch 2008) populate
    the (c, chi) plane off the steady S-curve and the 4th axis becomes
    meaningful. The flamelet npz format (chi_st per file) already supports
    this."""
    Z_axis = np.linspace(0.0, 1.0, N_Z)
    g_axis = np.linspace(0.0, G_MAX, N_G)
    Zq = np.linspace(0.0, 1.0, N_ZQ)

    if eq is None:
        raise SystemExit("build_tables requires the equilibrium closure "
                         "(eq=...); an un-closed manifold lets the "
                         "transported progress variable run away.")

    # C axis = NORMALIZED progress variable c = C/C_norm(Z) in [0, 1].
    C_axis = np.linspace(0.0, 1.0, N_C)
    C_norm, T1, Y1 = _norm_envelope(fls, eq, Zq, species)

    field_names = ["omega_C", "T"] + [f"Y_{sp}" for sp in species]

    # Boundary rows: c=1 -> envelope/equilibrium closure (omega=0),
    # c=0 -> frozen mixing (omega=0).
    bZ = np.concatenate([Zq, Zq])
    bc = np.concatenate([np.ones(N_ZQ), np.zeros(N_ZQ)])
    bval = {"omega_C": np.zeros(2 * N_ZQ),
            "T": np.concatenate([T1, eq["T_u"]])}
    for sp in species:
        bval[f"Y_{sp}"] = np.concatenate([Y1[sp], eq["Y_u"][sp]])

    def _cloud(flamelets):
        """Scattered (Z, c, {field: W}) cloud of a set of flamelets, with
        the normalized source and the closure boundary rows appended."""
        Zs_l, cs_l = [bZ], [bc]
        W_l = {f: [bval[f]] for f in field_names}
        for fl in flamelets:
            Z = fl["Z"]
            Cn_at = np.interp(Z, Zq, C_norm)
            c = np.clip(fl["C"] / Cn_at, 0.0, 1.0)
            Zs_l.append(Z)
            cs_l.append(c)
            W_l["omega_C"].append(np.where(Cn_at > CEQ_MIN,
                                           fl["omega_C"] / Cn_at, 0.0))
            W_l["T"].append(fl["T"])
            for sp in species:
                W_l[f"Y_{sp}"].append(fl["Y"][sp])
        return (np.concatenate(Zs_l), np.concatenate(cs_l),
                {f: np.concatenate(W_l[f]) for f in field_names})

    t0 = time.time()
    if not chi_mode:
        chi_axis = None
        print(f"[build] steady 3-D FPV: nZ={N_Z} nGz={N_G} nC={N_C} "
              f"(c in [0,1], envelope-normalized), full family of "
              f"{len(fls)} flamelets fills (Z,c)")
        Zs, cs, Ws = _cloud(fls)
        lam = {f: _laminar_one(Zs, cs, Ws[f], Zq, C_axis)[None, ...]
               for f in field_names}
        n_slices = 1
    else:
        chi_axis = build_chi_axis(fls, n_chi)
        print(f"[build] UFPV chi-mode: nZ={N_Z} nGz={N_G} nC={N_C} "
              f"nChi={n_chi} chi=[{chi_axis[0]:.3g},{chi_axis[-1]:.3g}] 1/s")
        lam = {f: np.zeros((n_chi, N_ZQ, N_C)) for f in field_names}
        for iChi, chi_target in enumerate(chi_axis):
            Zs, cs, Ws = _cloud([flamelet_at_chi(fls, chi_target)])
            for f in field_names:
                lam[f][iChi] = _laminar_one(Zs, cs, Ws[f], Zq, C_axis)
        n_slices = n_chi
    print(f"[build] {len(field_names)} fields × {n_slices} slice(s) "
          f"in {time.time()-t0:.1f}s")

    # Precompute beta-PDF weights once: W[iZ, iG, q]
    t0 = time.time()
    W = np.zeros((N_Z, N_G, N_ZQ))
    for iZ, Zm in enumerate(Z_axis):
        for iG, g in enumerate(g_axis):
            W[iZ, iG, :] = beta_pdf_weights(Zm, g, Zq)
    print(f"[build] beta weights in {time.time()-t0:.1f}s")

    # Convolve over Z only: table[f][iZ,iG,iC(,iChi)]
    #   = sum_q W[iZ,iG,q] * lam[f][iChi, q, iC]
    tables = {}
    for f, L_chi in lam.items():
        # L_chi shape: (n_slices, N_ZQ, N_C); reorder to (N_ZQ, N_C, n)
        L = np.moveaxis(L_chi, 0, -1)
        out = np.einsum("zgq,qck->zgck", W, L)
        tables[f] = out[..., 0] if chi_axis is None else out
    return Z_axis, g_axis, C_axis, chi_axis, tables


# --------------------------- writer ---------------------------

def _fmt(arr):
    return "\n".join(f"    {v:.8e}" for v in arr)


def write_fgm_npz(path, Z_axis, g_axis, C_axis, chi_axis, tables,
                  species, meta, eq=None):
    """Companion of write_fgm_dict: same data in a single .npz so Python
    consumers (e.g. apriori_check_4d.py) can reload the 4-D arrays without
    re-parsing the OpenFOAM dictionary."""
    arrs = {
        "Z_axis":   np.asarray(Z_axis),
        "gZ_axis":  np.asarray(g_axis),
        "C_axis":   np.asarray(C_axis),
        "chi_axis": (np.zeros(0) if chi_axis is None
                     else np.asarray(chi_axis)),
        "species":  np.array(species, dtype=object),
        "P":        np.asarray(meta.get("P", float("nan"))),
        "src":      np.array(meta.get("src", "?"), dtype=object),
        "closure":  np.array(meta.get("closure", "raw-C"), dtype=object),
    }
    if eq is not None:
        # Diagnostics for a-priori checks: C is normalized in the table, so
        # consumers need Yc_eq(Z) (and the boundary states) to de-normalize.
        arrs["Zq_eq"] = np.linspace(0.0, 1.0, len(eq["C_eq"]))
        arrs["C_eq"] = np.asarray(eq["C_eq"])
        arrs["T_eq"] = np.asarray(eq["T_eq"])
        arrs["T_u"] = np.asarray(eq["T_u"])
    for name, T in tables.items():
        # store as native (nZ, nGz, nC, nChi) ordering
        arrs[name] = np.ascontiguousarray(T)
    np.savez_compressed(path, **arrs)
    shape = f"{Z_axis.size}x{g_axis.size}x{C_axis.size}" + \
            ("" if chi_axis is None else f"x{chi_axis.size}")
    print(f"[write] {path}  ({len(tables)} fields, {shape})")


def write_fgm_dict(path, Z_axis, g_axis, C_axis, chi_axis, tables,
                   species, meta):
    """chi_axis=None writes a 3-D (Z, gZ, c) table -- no nChi/chi entries,
    so the solver's FGMTable takes its 3-D legacy path automatically. A
    chi_axis writes the 4-D layout (kept for the future UFPV manifold)."""
    nZ, nGz, nC = len(Z_axis), len(g_axis), len(C_axis)
    nChi = 1 if chi_axis is None else len(chi_axis)

    blocks = []
    blocks.append(f"sourcePV\n(\n{_fmt(tables['omega_C'].reshape(-1))}\n);")
    if "T" in tables:
        blocks.append(f"T\n(\n{_fmt(tables['T'].reshape(-1))}\n);")
    if species:
        blocks.append("species ( " + " ".join(species) + " );")
        for sp in species:
            blocks.append(
                f"Y_{sp}\n(\n{_fmt(tables[f'Y_{sp}'].reshape(-1))}\n);"
            )
    body = "\n\n".join(blocks)

    if chi_axis is None:
        kind = "3-D (steady FPV: Z~, gZ, c)"
        idx = "idx = (iZ*nGz + iGz)*nC + iC"
        axes_blk = ""
        nchi_blk = ""
    else:
        kind = "4-D (UFPV: Z~, gZ, c, chi_st)"
        idx = "idx = ((iZ*nGz + iGz)*nC + iC)*nChi + iChi"
        axes_blk = f"\nchi\n(\n{_fmt(chi_axis)}\n);\n"
        nchi_blk = f"nChi    {nChi};\n"

    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| FGM {kind} lookup table -- 04_build_fgm_table_4d.py
| source: {meta.get('src','?')}
| P_ref = {meta.get('P', float('nan')):.6g} Pa | closure: {meta.get('closure','?')}
| c = C/C_norm(Z), C_norm = max(equilibrium, family envelope); omega(c=1)=0
| flat C-order:  {idx}
| tabulated: sourcePV (omega_c), T, Y_<species>
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
{nchi_blk}
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
{axes_blk}
{body}

// ************************************************************************* //
"""
    path.write_text(header)
    print(f"[write] {path}  [{kind}]")
    print(f"[write]   fields: sourcePV, T, "
          f"Y_{{{','.join(species)}}}")
    print(f"[write]   shape per field: ({nZ}, {nGz}, {nC}"
          + (f", {nChi})" if chi_axis is not None else ")")
          + f" = {nZ*nGz*nC*nChi} entries")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flamelet-dir", required=True,
                   help="directory of flamelet_*.npz files (stage 6 output)")
    p.add_argument("--out", dest="outfile",
                   default=str(DATA_DIR / "fgmProperties_4d"))
    p.add_argument("--n-chi", type=int, default=N_CHI,
                   help="number of chi_st axis grid points (log-spaced)")
    p.add_argument("--species", default=None,
                   help="space-separated subset of species to tabulate "
                        "(default: all). Used to build compact tables matching "
                        "the OpenFOAM case thermo; un-tabulated species are "
                        "absorbed by the solver's defaultSpecie.")
    p.add_argument("--yaml", dest="yaml_file", default=str(DEFAULT_YAML),
                   help="Cantera mechanism for the equilibrium burnt-end "
                        "closure (ideal-gas variant, consistent with the "
                        "flamelet structure solve)")
    p.add_argument("--chi-axis", action="store_true",
                   help="build the 4-D (Z,gZ,c,chi_st) UFPV layout instead "
                        "of the default steady 3-D FPV (Z,gZ,c). Only "
                        "meaningful once UNSTEADY flamelets populate the "
                        "(c,chi) plane; for steady families c and chi_st "
                        "are redundant.")
    args = p.parse_args()

    fl_dir = Path(args.flamelet_dir)
    if not fl_dir.is_dir():
        raise SystemExit(f"missing dir {fl_dir}")
    print(f"[load] {fl_dir}")

    fls, species = load_flamelet_dir(fl_dir)
    print(f"[load] {len(fls)} flamelets; species count = {len(species)}")
    if args.species:
        want = set(args.species.split())
        missing = [s for s in want if s not in species]
        if missing:
            raise SystemExit(f"requested species not in flamelets: {missing}")
        species = [s for s in species if s in want]   # keep sorted order
        print(f"[load] restricted to {len(species)} species: "
              f"{' '.join(species)}")
    P_ref = float(np.median([fl["P"] for fl in fls]))
    d0 = np.load(fls[0]["path"])
    T_fuel = float(d0["T_fuel"]); T_ox = float(d0["T_ox"])
    meta = {"src": fl_dir.name, "P": P_ref,
            "closure": "envelope-normalized-c (eq + family max, omega(1)=0)"}

    # Equilibrium burnt-end closure on the internal Zq grid (same grid the
    # beta-PDF convolution integrates over).
    Zq = np.linspace(0.0, 1.0, N_ZQ)
    eq = equilibrium_closure(args.yaml_file, Zq, P_ref, T_fuel, T_ox, species)

    Z_axis, g_axis, C_axis, chi_axis, tables = build_tables(
        fls, species, n_chi=args.n_chi, eq=eq, chi_mode=args.chi_axis
    )
    for f, T in tables.items():
        print(f"[build]   {f}: range [{T.min():.3g}, {T.max():.3g}]")

    out_dict = Path(args.outfile)
    write_fgm_dict(out_dict, Z_axis, g_axis, C_axis, chi_axis,
                   tables, species, meta)
    # Companion npz next to the dict file. apriori_check_4d.py reads this.
    out_npz = out_dict.with_suffix(out_dict.suffix + ".npz") \
              if out_dict.suffix else out_dict.with_name(out_dict.name + ".npz")
    write_fgm_npz(out_npz, Z_axis, g_axis, C_axis, chi_axis,
                  tables, species, meta, eq=eq)


if __name__ == "__main__":
    main()
