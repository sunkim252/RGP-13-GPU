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


def compute_hrr(fls, yaml_file, species, P):
    """Attach a per-point volumetric heat-release rate [W/m^3] to each flamelet,
    re-evaluated from the saved (T, Y) structure with the ideal-gas Cantera
    mechanism (Cantera heat_release_rate = -sum_k h_k w_dot_k). Enables the
    consumer to integrate the global q_dot for the Wang-2015 Fig.19c check."""
    import cantera as ct
    g = ct.Solution(str(yaml_file))
    names = g.species_names
    idx = {sp: names.index(sp) for sp in species if sp in names}
    for fl in fls:
        T = fl["T"]; n = len(T)
        hrr = np.zeros(n)
        Ymat = np.zeros((n, g.n_species))
        for sp, j in idx.items():
            Ymat[:, j] = fl["Y"][sp]
        for i in range(n):
            s = Ymat[i].sum()
            if s <= 0:
                continue
            g.TPY = max(float(T[i]), 250.0), P, Ymat[i] / s
            hrr[i] = g.heat_release_rate
        fl["hrr"] = hrr
    print(f"[hrr] volumetric heat-release rate for {len(fls)} flamelets "
          f"(max {max(float(fl['hrr'].max()) for fl in fls):.3g} W/m^3)")


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


def _laminar_branch(Z, C, W, Zq, Cgrid):
    """Burning-branch (upper-envelope) closure for the MULTI-VALUED source.

    The flamelet family is an S-curve: at one (Z, c) several flamelets of
    different strain coexist, with DIFFERENT omega_C (upper = burning branch,
    lower = extinguishing). griddata(method='linear') triangulates the whole
    concatenated cloud and LINEARLY BLENDS those branches, hollowing the source
    wherever the cloud is dense and multi-valued. Verified culprit of the 10 atm
    weak source: griddata gave peak omega_c(Zst)=1619 vs the true burning-branch
    4571 (-2.8x); 1 atm / 50 atm clouds barely overlap in c so they were
    unaffected (max-bin == griddata there). FPV (Pierce 2004) tabulates the
    STABLE BURNING branch, so we take the per-cell MAX (the upper envelope) and
    griddata-fill the empty cells. Used for omega_C / hrr only; T, Y_k are
    effectively single-valued in (Z, c) and keep the smooth linear interpolant."""
    nz, ncg = len(Zq), len(Cgrid)
    # Smooth upper envelope: per grid node take the MAX over cloud points in a
    # (dz, dc) neighborhood (not a single cell -- that spikes and the gap-fill
    # then undershoots between spikes). The window picks the burning branch and
    # spreads it across the c range the burning family actually occupies.
    dz = 2.0 * (Zq[1] - Zq[0])
    dc = 0.06
    out = np.full((nz, ncg), np.nan)
    for i, zq in enumerate(Zq):
        zm = np.abs(Z - zq) <= dz
        if not zm.any():
            continue
        Czi, Wzi = C[zm], W[zm]
        for j, cq in enumerate(Cgrid):
            m = np.abs(Czi - cq) <= dc
            if m.any():
                out[i, j] = Wzi[m].max()
    filled = ~np.isnan(out)
    if HAVE_SCIPY and not filled.all() and filled.any():
        ZZ, CC = np.meshgrid(Zq, Cgrid, indexing="ij")
        pts = np.column_stack([ZZ[filled], CC[filled]])
        vals = out[filled]
        out[~filled] = griddata(pts, vals,
                                np.column_stack([ZZ[~filled], CC[~filled]]),
                                method="linear")
        nan2 = np.isnan(out)
        if nan2.any():
            out[nan2] = griddata(pts, vals,
                                 np.column_stack([ZZ[nan2], CC[nan2]]),
                                 method="nearest")
    out[np.isnan(out)] = 0.0
    # NEGATIVE source is kept (2026-07-03): near/above local equilibrium the
    # net PV production is genuinely negative (recombination back toward
    # equilibrium; with differential diffusion the cloud reaches c up to
    # ~1.06). Clipping it to 0 punched exact-zero holes with sharp edges into
    # the MA table wherever the envelope window held only near-equilibrium
    # points (Z~0.3-0.5, c>0.9), stalling c progression there. A (small)
    # negative source is the physically correct relaxation sink and is
    # STABILISING in the CFD: it pulls c back toward equilibrium instead of
    # letting it pile up at the c=1 clamp. omega(c=1)~0 still holds to within
    # the recombination scale.
    return out


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
        # a flamelet contributes to the envelope only within its own Z range
        # (edge extrapolation is harmless for full-span counterflow families
        # but corrupts C_norm for premixed flamelets, which live at a single
        # Z: their stoichiometric C would otherwise be claimed at every Z)
        span = (Zq >= Zs[0] - 1e-9) & (Zq <= Zs[-1] + 1e-9)
        if not span.all():
            pad = 0.5 * (Zq[1] - Zq[0])
            span = (Zq >= Zs[0] - pad) & (Zq <= Zs[-1] + pad)
            C_f[i][~span] = 0.0
    C_fam = C_f.max(axis=0)
    owner = C_f.argmax(axis=0)
    eq_owns = C_eq >= C_fam
    C_norm = np.maximum(np.maximum(C_eq, C_fam), CEQ_MIN) * (1.0 + 1.0e-3)
    # c=1 boundary STATE = chemical equilibrium ALWAYS (Pierce & Moin 2004; van
    # Oijen & de Goey 2000: "the manifold must match the equilibrium composition
    # exactly" at the burnt end). The earlier code used the highest-C flamelet
    # (envelope owner) wherever eq_owns is False, which is wrong on two counts:
    #   (1) a finite-domain steady flamelet never fully reaches equilibrium, so
    #       even the lowest-strain member sits 30-100 K BELOW T_eq at the burnt end;
    #   (2) if the flamelet family carries a NOISY (under-converged) member, the
    #       max-C owner can be a cool, CO-rich, half-extinguished state (observed:
    #       chi=87 flamelet, T(Z_st)=2789 K vs T_eq=3794 K) -- this hollowed the
    #       c=1 column and corrupted the whole high-c manifold.
    # C_norm = max(C_eq, family envelope) still bounds the transported c in [0,1]
    # (cross-Z diffusion pushes the flamelet product mass fraction above the LOCAL
    # equilibrium on the rich side, so C_fam > C_eq there); only the c=1 STATE is
    # pinned to equilibrium. omega = 0 on the whole c=1 row regardless.
    T1 = np.asarray(eq["T_eq"], dtype=float).copy()
    Y1 = {sp: np.asarray(eq["Y_eq"][sp], dtype=float).copy() for sp in species}
    print(f"[build] C_norm envelope: eq owns {100*eq_owns.mean():.0f}% of Z "
          f"(c=1 STATE forced to chemical equilibrium everywhere); "
          f"family/eq max ratio={np.max(C_fam/np.maximum(C_eq,CEQ_MIN)):.3f}")
    return C_norm, T1, Y1


def make_z_axis(n, z_cluster=None):
    """Z axis: uniform by default; with z_cluster=(Z_st, beta) an
    Eriksson/Roberts interior-point clustering concentrates nodes around
    Z_st (needed for H2/O2, Z_st~0.11, where a uniform 51-pt grid puts only
    ~6 nodes across the whole flame). The solver's FGMTable::bracket() does
    a general linear scan, so non-uniform axes are supported as-is."""
    if not z_cluster:
        return np.linspace(0.0, 1.0, n)
    z_st, beta = z_cluster
    eta = np.linspace(0.0, 1.0, n)
    eta0 = (1.0 / (2.0 * beta)) * np.log(
        (1.0 + (np.exp(beta) - 1.0) * z_st)
        / (1.0 + (np.exp(-beta) - 1.0) * z_st))
    Z = z_st * (1.0 + np.sinh(beta * (eta - eta0)) / np.sinh(beta * eta0))
    Z[0], Z[-1] = 0.0, 1.0
    print(f"[grid] Z axis clustered at Z_st={z_st} (beta={beta}): "
          f"{int((Z <= 2.5 * z_st).sum())}/{n} nodes below 2.5*Z_st")
    return Z


def _smooth_manifold(tables, smooth_c, C_axis):
    """Mild Gaussian smoothing of the assembled manifold in the c (and, at
    half strength, Z) directions. Rationale: the T/source contours are faceted
    because the interpolant traces a NOISY multi-valued flamelet cloud (denser
    coverage does not help -- 28-pt and 20-pt families are equally rough; a
    monotone-cubic PCHIP interpolant makes it WORSE by tracing every scatter
    point). A SMOOTHER (regression), not a higher-order interpolant, is the
    fix: T c-roughness 27 -> 7 K, source 2.6x lower, physics preserved.

    Only T and the source (omega_C, hrr) are smoothed -- Y_k are left intact so
    solver-consistent RG/Le tabulation keeps physical compositions. Boundaries
    are restored after smoothing: c=0 (mixing) and c=1 (equilibrium anchor) for
    T; omega(c=0)=omega(c=1)=0 for the source. Source is smoothed in log space
    to preserve its dynamic range and peak.
    """
    from scipy.ndimage import gaussian_filter
    sc, sz = float(smooth_c), 0.5 * float(smooth_c)
    src = {"omega_C", "hrr"}
    for f, A in tables.items():
        if f.startswith("Y_"):
            continue                       # keep compositions untouched
        cax = A.ndim - 2 if A.ndim == 4 else A.ndim - 1   # c is 2nd-last (chi) or last
        # per-slice sigma: Z(axis0)=sz, c(cax)=sc, others 0
        sig = [0.0] * A.ndim
        sig[0] = sz
        sig[cax] = sc
        c0 = np.take(A, 0, axis=cax).copy()          # mixing boundary
        c1 = np.take(A, A.shape[cax] - 1, axis=cax).copy()   # anchor
        peak0 = np.abs(A).max()
        A[...] = gaussian_filter(A, sigma=sig, mode="nearest")
        if f in src:
            # renormalize to the pre-smoothing peak (mild Gaussian shaves the
            # sharp mid-c source peak a few %); keeps sign / small negative
            # (recombination) sink intact. Restore omega(c=0)=omega(c=1)=0.
            pk = np.abs(A).max()
            if pk > 0:
                A *= peak0 / pk
            _put(A, 0, cax, np.zeros_like(c0))
            _put(A, A.shape[cax] - 1, cax, np.zeros_like(c1))
        else:
            _put(A, 0, cax, c0)            # restore exact mixing / equilibrium
            _put(A, A.shape[cax] - 1, cax, c1)
    print(f"[smooth] Gaussian c-smoothing sigma_c={sc:.2f} sigma_Z={sz:.2f} "
          f"cells (T + source; Y_k intact; boundaries restored)")


def _put(A, idx, axis, val):
    """A.take(idx,axis) = val, in place (np.put_along_axis wrapper)."""
    sl = [slice(None)] * A.ndim
    sl[axis] = idx
    A[tuple(sl)] = val


def build_tables(fls, species, n_chi=N_CHI, eq=None, chi_mode=False,
                 z_cluster=None, smooth_c=0.0):
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
    Z_axis = make_z_axis(N_Z, z_cluster)
    g_axis = np.linspace(0.0, G_MAX, N_G)
    Zq = np.linspace(0.0, 1.0, N_ZQ)

    if eq is None:
        raise SystemExit("build_tables requires the equilibrium closure "
                         "(eq=...); an un-closed manifold lets the "
                         "transported progress variable run away.")

    # C axis = NORMALIZED progress variable c = C/C_norm(Z) in [0, 1].
    C_axis = np.linspace(0.0, 1.0, N_C)
    C_norm, T1, Y1 = _norm_envelope(fls, eq, Zq, species)
    # expose the actual normalization to npz consumers (a-priori checks
    # de-normalize with THIS, not C_eq -- they differ wherever the family
    # envelope owns the bound, strongly so for H2/O2 differential diffusion)
    eq["C_norm_Zq"] = C_norm.copy()

    # "hrr" (volumetric heat-release rate [W/m^3]) is tabulated when the
    # flamelets carry it (added by compute_hrr); it lets a consumer integrate
    # q_dot over the flame zone for the Wang-2015 Fig.19c comparison.
    has_hrr = all("hrr" in fl for fl in fls)
    field_names = ["omega_C", "T"] + (["hrr"] if has_hrr else []) \
        + [f"Y_{sp}" for sp in species]

    # Boundary rows: c=1 -> envelope/equilibrium closure (omega=0),
    # c=0 -> frozen mixing (omega=0). Both have zero net reaction => hrr=0.
    bZ = np.concatenate([Zq, Zq])
    bc = np.concatenate([np.ones(N_ZQ), np.zeros(N_ZQ)])
    bval = {"omega_C": np.zeros(2 * N_ZQ),
            "T": np.concatenate([T1, eq["T_u"]])}
    if has_hrr:
        bval["hrr"] = np.zeros(2 * N_ZQ)
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
            if has_hrr:
                W_l["hrr"].append(fl["hrr"])
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
        # Source fields (omega_C, hrr) are multi-valued on the S-curve -> use the
        # burning-branch (upper-envelope) closure; griddata-linear blends the
        # branches and hollowed the dense 10 atm source. T, Y_k are single-valued
        # in (Z, c) and keep the smooth linear interpolant.
        SRC_FIELDS = {"omega_C", "hrr"}
        lam = {f: (_laminar_branch(Zs, cs, Ws[f], Zq, C_axis)
                   if f in SRC_FIELDS
                   else _laminar_one(Zs, cs, Ws[f], Zq, C_axis))[None, ...]
               for f in field_names}
        n_slices = 1
    else:
        chi_axis = build_chi_axis(fls, n_chi)
        print(f"[build] UFPV chi-mode: nZ={N_Z} nGz={N_G} nC={N_C} "
              f"nChi={n_chi} chi=[{chi_axis[0]:.3g},{chi_axis[-1]:.3g}] 1/s")
        lam = {f: np.zeros((n_chi, N_ZQ, N_C)) for f in field_names}
        # chi-WINDOW coverage fix (2026-06-21): a single chi flamelet is a 1-D
        # curve in (Z,c), so _laminar_one nearest-extrapolates off it -> source
        # coverage gaps -> the CFD extinguishes. Instead fill each chi slice from
        # a CLOUD of flamelets within a log-chi window (overlapping adjacent
        # slices) so the (Z,c) plane is covered while keeping the slice strain-
        # local (window narrower than the global family -> no upper-envelope
        # over-sourcing).
        cw = (chi_axis[-1] / chi_axis[0]) ** (0.65 / max(n_chi - 1, 1))
        for iChi, chi_target in enumerate(chi_axis):
            lo, hi = chi_target / cw, chi_target * cw
            window = [fl for fl in fls if lo <= fl["chi_st"] <= hi]
            if len(window) < 3:
                window = sorted(fls, key=lambda fl: abs(
                    np.log(max(fl["chi_st"], EPS)) - np.log(max(chi_target, EPS))))[:3]
            Zs, cs, Ws = _cloud(window)
            for f in field_names:
                lam[f][iChi] = _laminar_one(Zs, cs, Ws[f], Zq, C_axis)
        print(f"[build] chi-window cw={cw:.2f}x, ~{len(window)} flamelets/slice")
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
    if smooth_c and smooth_c > 0:
        _smooth_manifold(tables, smooth_c, C_axis)
    return Z_axis, g_axis, C_axis, chi_axis, tables


# --------------------------- writer ---------------------------

def _fmt(arr):
    return "\n".join(f"    {v:.8e}" for v in arr)


def write_fgm_npz(path, Z_axis, g_axis, C_axis, chi_axis, tables,
                  species, meta, eq=None, fourth_kind=None):
    """Companion of write_fgm_dict: same data in a single .npz so Python
    consumers (e.g. apriori_check_4d.py) can reload the 4-D arrays without
    re-parsing the OpenFOAM dictionary. fourth_kind='dilution' also stores
    the 4th axis under 'W_axis' (its physical name) for analysis clarity."""
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
        "fourth_kind": np.array(fourth_kind or "chi", dtype=object),
    }
    if fourth_kind == "dilution" and chi_axis is not None:
        arrs["W_axis"] = np.asarray(chi_axis)
    if eq is not None:
        # Diagnostics for a-priori checks: C is normalized in the table, so
        # consumers need Yc_eq(Z) (and the boundary states) to de-normalize.
        arrs["Zq_eq"] = np.linspace(0.0, 1.0, len(eq["C_eq"]))
        arrs["C_eq"] = np.asarray(eq["C_eq"])
        if "C_norm_Zq" in eq:
            arrs["C_norm"] = np.asarray(eq["C_norm_Zq"])
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
                   species, meta, fourth_kind=None):
    """chi_axis=None writes a 3-D (Z, gZ, c) table -- no nChi/chi entries,
    so the solver's FGMTable takes its 3-D legacy path automatically. A
    chi_axis writes the 4-D layout.

    fourth_kind selects what the 4th axis MEANS (the array/flat-order code is
    identical -- 4th axis innermost -- only the emitted keywords + header text
    change so the solver's generic 4th-axis machinery reads the right one):
      None       -> chi_st (UFPV)         : nChi + chi(...)
      'dilution' -> steam-in-oxidiser W   : fourthAxis dilution; nW + W(...)
    """
    nZ, nGz, nC = len(Z_axis), len(g_axis), len(C_axis)
    nChi = 1 if chi_axis is None else len(chi_axis)

    # Tier-4: tabulated differential-diffusion Lewis numbers (flat Le_<var>
    # fields). When present the solver applies a per-cell Le(Z,gZ,c[,chi]); the
    # constant 'Le' sub-dict below is kept only as a fallback for older solvers.
    has_le = ("Le_Z" in tables) and ("Le_C" in tables)

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
    if has_le:
        blocks.append(f"Le_Z\n(\n{_fmt(tables['Le_Z'].reshape(-1))}\n);")
        blocks.append(f"Le_C\n(\n{_fmt(tables['Le_C'].reshape(-1))}\n);")
    body = "\n\n".join(blocks)

    if chi_axis is None:
        kind = "3-D (steady FPV: Z~, gZ, c)"
        idx = "idx = (iZ*nGz + iGz)*nC + iC"
        axes_blk = ""
        nchi_blk = ""
    elif fourth_kind == "dilution":
        # 4th axis = steam-in-oxidiser mole fraction W. Reuses the generic
        # 4th-axis slot (nChi/chi machinery) but tags it 'dilution' so the
        # solver forms the local W from a transported steam-dilution scalar
        # (analogous to the 'enthalpy' 4th-axis defect path). Innermost axis.
        kind = "4-D (steam-diluted FPV: Z~, gZ, c, W)"
        idx = "idx = ((iZ*nGz + iGz)*nC + iC)*nW + iW"
        axes_blk = (f"\nfourthAxis  dilution;\nnW      {nChi};\n"
                    f"W\n(\n{_fmt(chi_axis)}\n);\n")
        nchi_blk = ""
    else:
        kind = "4-D (UFPV: Z~, gZ, c, chi_st)"
        idx = "idx = ((iZ*nGz + iGz)*nC + iC)*nChi + iChi"
        axes_blk = f"\nchi\n(\n{_fmt(chi_axis)}\n);\n"
        nchi_blk = f"nChi    {nChi};\n"

    # Constant 'Le' sub-dict: the manifold-median Lewis numbers when Tier-4
    # tabulated fields are present (kept only as a fallback for older solvers
    # that ignore Le_Z/Le_C), otherwise the historical option-B constants.
    le_meta = meta.get("le", None)
    if has_le and le_meta is not None:
        le_z_c, le_c_c = float(le_meta[0]), float(le_meta[1])
        le_comment = (
            "// Tier-4 DIFFERENTIAL DIFFUSION: per-cell Le_Z(Z,gZ,c)/Le_C tabulated\n"
            "// below (real-fluid SRK+Chung/Takahashi). The solver uses those fields;\n"
            "// this constant sub-dict is the manifold MEDIAN, a fallback for solvers\n"
            "// that predate the tabulated-Le hook. Solver: rho*D_lam = mu/Le.")
    else:
        le_z_c, le_c_c = 0.63, 0.60
        le_comment = (
            "// Differential-diffusion Lewis numbers (option B, 2026-06-20). Solver uses\n"
            "// rho*D_lam = mu/Le, i.e. these are effectively Schmidt numbers Sc = nu/D.\n"
            "//   Sc_C ~= 0.60  (HRR-weighted, consistent across 1-150 atm; C diffuses ~alpha)\n"
            "//   Sc_Z ~= 0.63 (= Pr; Z treated as conserved scalar at unity-Lewis-thermal,\n"
            "//                 since the directly-computed Sc_Z is erratic, H/H2-dominated)\n"
            "// Corrects the unity base (Le=1 -> D=nu=Pr*alpha, under-diffusing) to D~alpha.")

    header = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| FGM {kind} lookup table -- 04_build_fgm_table_4d.py
| source: {meta.get('src','?')}
| P_ref = {meta.get('P', float('nan')):.6g} Pa | closure: {meta.get('closure','?')}
| c = C/C_norm(Z), C_norm = max(equilibrium, family envelope); omega(c=1)=0
| flat C-order:  {idx}
| tabulated: sourcePV (omega_c), T, Y_<species>{', Le_Z, Le_C (Tier-4 diff-diff)' if has_le else ''}
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
{le_comment}
Le
{{
    Z   {le_z_c:.6g};
    C   {le_c_c:.6g};
}}

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


def compute_lewis_tables(tables, species, P, srk_yaml,
                         transport_model="high-pressure-Chung",
                         Tmin_skip=200.0, Tmin_eval=250.0,
                         soret=False, Z_axis=None, C_axis=None,
                         C_norm=None, x_fuel=None, x_ox=None):
    """Tier-4: tabulate the differential-diffusion Lewis numbers Le_Z and Le_C
    over the manifold from the REAL-FLUID (SRK + high-pressure-Chung/Takahashi)
    transport evaluated at each node's tabulated (T, Y).

    The solver applies rho*D_lam = mu/Le, so Le here acts as a Schmidt number
    Sc = nu/D. The two control variables get a physically distinct, spatially
    varying closure that generalises the old constant Sc_Z~0.63 / Sc_C~0.60:

        Le_Z = Pr = mu*cp/lam = nu/alpha
            Mixture fraction is a conserved scalar carried at unity THERMAL
            Lewis number, so it diffuses like heat (D_Z = alpha). Le_Z is then
            the local real-fluid Prandtl number.

        Le_C = nu / D_C,
            D_C = sum_{k in PV} Y_k D_k,mix / sum_{k in PV} Y_k
            Progress variable diffuses with the PV-mass-weighted mixture-
            averaged species diffusivity (Takahashi-corrected at high p).

    soret=True folds the thermal-diffusion (Soret) flux of each control
    variable into an EFFECTIVE Lewis number, so the solver captures Soret with
    NO extra transport term (the Le fields already feed D=nu/Le). On the
    manifold T=T(Z,gZ,c[,W]) so grad(T)=(dT/dZ)grad Z+(dT/dc)grad c: the Soret
    mass flux j^S = -(D^T/T)grad T projects onto each variable's own gradient
    (diagonal approximation; the Z<->c cross term is level-3 future work):
        D_C,eff = D_C   + (D^T_C /(rho T C_norm)) (dT/dc)
        D_Z,eff = alpha + (D^T_Z /(rho T))        (dT/dZ)
    D^T_C = sum_{PV} D^T_k ; D^T_Z = (sum_k b_k D^T_k)/dbeta, b_k the Bilger
    coefficient (2 nH_k - nO_k)/M_k, dbeta = beta_fuel - beta_ox. Requires the
    grid axes + C_norm(Z) + stream compositions. Falls back to the diagonal
    (Soret-off) Le wherever the correction is unavailable/unphysical.

    Returns (Le_Z, Le_C) arrays shaped like tables['T'], plus (medZ, medC) the
    manifold-median values used as the constant-Le fallback / degenerate fill.
    """
    import cantera as ct

    gas = ct.Solution(srk_yaml)
    gas.transport_model = transport_model
    names = gas.species_names
    sp_in_mech = [s for s in species if s in names]
    missing = [s for s in species if s not in names]
    if missing:
        print(f"[lewis] WARNING: {len(missing)} tabulated species absent from "
              f"{Path(srk_yaml).name}, treated as 0 in transport: {missing[:8]}"
              + (" ..." if len(missing) > 8 else ""))
    idx = {s: names.index(s) for s in sp_in_mech}
    pv = [s for s in PV_SPECIES if s in idx]
    if not pv:
        raise SystemExit(f"[lewis] none of the PV species {PV_SPECIES} are in "
                         f"{srk_yaml}; cannot define Le_C")
    pv_cols = [idx[s] for s in pv]

    T = np.asarray(tables["T"])
    shape = T.shape
    flatT = T.reshape(-1)
    nNode = flatT.size

    Ymat = np.zeros((nNode, gas.n_species))
    for s in sp_in_mech:
        Ymat[:, idx[s]] = np.asarray(tables[f"Y_{s}"]).reshape(-1)

    # ---- Soret setup: manifold gradients + Bilger coefficients ----
    do_soret = bool(soret) and Z_axis is not None and C_axis is not None
    if do_soret:
        # dT/dc (axis 2 = C) and dT/dZ (axis 0 = Z), flattened to node order
        dTdc_f = np.gradient(T, np.asarray(C_axis), axis=2).reshape(-1)
        dTdZ_f = np.gradient(T, np.asarray(Z_axis), axis=0).reshape(-1)
        # C_norm(Z) broadcast to the table shape (default 1 -> no rescale)
        if C_norm is not None:
            cn = np.asarray(C_norm)
            bshape = [1] * T.ndim; bshape[0] = shape[0]
            Cn_f = np.broadcast_to(cn.reshape(bshape), shape).reshape(-1)
        else:
            Cn_f = np.ones(nNode)
        Mk = gas.molecular_weights
        nH = np.array([gas.n_atoms(k, "H") for k in range(gas.n_species)])
        nO = np.array([gas.n_atoms(k, "O") for k in range(gas.n_species)])
        b_k = (2.0 * nH - nO) / Mk                         # Bilger coeff
        def _beta(xstr):
            gas.TPX = 300.0, P, xstr
            return float(np.dot(b_k, gas.Y))
        try:
            dbeta = _beta(x_fuel) - _beta(x_ox)
        except Exception:
            dbeta = None
        if not dbeta or abs(dbeta) < 1e-9:
            do_soret = "C-only"           # Z-Soret needs a valid dbeta
        soret_dc = np.zeros(nNode); soret_dz = np.zeros(nNode)

    LeZ = np.full(nNode, np.nan)
    LeC = np.full(nNode, np.nan)
    nfail = 0
    t0 = time.time()
    for n in range(nNode):
        if n and n % 25000 == 0:
            print(f"[lewis]   {n}/{nNode} nodes "
                  f"({time.time()-t0:.0f}s, {nfail} fails)", flush=True)
        Tn = float(flatT[n])
        y = Ymat[n]
        ssum = y.sum()
        if not np.isfinite(Tn) or Tn < Tmin_skip or ssum <= 1e-8:
            continue
        try:
            gas.TPY = max(Tn, Tmin_eval), P, y / ssum
            mu = gas.viscosity
            lam = gas.thermal_conductivity
            cp = gas.cp_mass
            rho = gas.density
            nu = mu / max(rho, 1e-30)
            alpha = lam / max(rho * cp, 1e-30)
            Dk = np.clip(gas.mix_diff_coeffs, 1e-12, None)
            ypv = y[pv_cols]
            wsum = float(ypv.sum())
            DC = (float((ypv * Dk[pv_cols]).sum() / wsum)
                  if wsum > 1e-12 else float(alpha))
            DZ = alpha                                      # D_Z = alpha (Pr)
            if do_soret:
                DT = gas.thermal_diff_coeffs               # [kg/m/s]
                # Diagonal projection of the Soret flux onto each variable's own
                # gradient. LIMITED to [-50%, +100%] of the base diffusivity:
                # the projection is a leading-order correction, and the large H2
                # Soret can otherwise drive the effective D singular (D<=0) where
                # dT/dZ opposes -- that pathology belongs to the explicit
                # cross-term (level 3), not to an effective-Le fold.
                def _lim(dD, base):
                    return base * (1.0 + min(1.0, max(-0.5, dD / max(base, 1e-30))))
                DTc = float(DT[pv_cols].sum())
                dDC = DTc * dTdc_f[n] / (max(rho, 1e-30) * max(Tn, 1e-30)
                                         * max(Cn_f[n], 1e-6))
                DC = _lim(dDC, DC)
                if do_soret is True:       # Z-Soret only with a valid dbeta
                    DTz = float(np.dot(b_k, DT)) / dbeta
                    dDZ = DTz * dTdZ_f[n] / (max(rho, 1e-30) * max(Tn, 1e-30))
                    DZ = _lim(dDZ, DZ)
                DC = max(DC, 1e-30); DZ = max(DZ, 1e-30)
            LeZ[n] = nu / max(DZ, 1e-30)
            LeC[n] = nu / max(DC, 1e-30)
        except Exception:
            nfail += 1

    # Robust fill: unconverged / degenerate corners (high-gZ, off-manifold c)
    # take the manifold median; clamp to a sane Lewis window so the table never
    # carries NaN/inf or a runaway diffusivity.
    def _fill(a, lo, hi, fallback):
        good = np.isfinite(a)
        med = float(np.median(a[good])) if good.any() else fallback
        nbad = int((~good).sum())
        a[~good] = med
        return np.clip(a, lo, hi), med, nbad

    LeZ, medZ, nbadZ = _fill(LeZ, 0.1, 10.0, 0.63)
    LeC, medC, nbadC = _fill(LeC, 0.1, 10.0, 0.60)
    stag = ("OFF" if not do_soret else
            ("PV-only (no dbeta)" if do_soret == "C-only" else "Z+C"))
    print(f"[lewis] real-fluid differential diffusion ({transport_model}, "
          f"Soret={stag}) on "
          f"{nNode} nodes: Le_Z med={medZ:.3f} [{LeZ.min():.2f},{LeZ.max():.2f}] "
          f"Le_C med={medC:.3f} [{LeC.min():.2f},{LeC.max():.2f}]")
    print(f"[lewis]   filled {nbadZ} degenerate Le_Z / {nbadC} Le_C nodes with "
          f"median; {nfail} transport eval failures")
    return LeZ.reshape(shape), LeC.reshape(shape), medZ, medC


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
    p.add_argument("--exclude-idx", default="",
                   help="comma-separated flamelet indices (NNN in "
                        "flamelet_NNN.npz) to drop before building -- used to "
                        "remove half-transitioned outliers that break the "
                        "monotonic T(c) envelope (Wang Fig18/19 low-P clean).")
    p.add_argument("--diff-diff-yaml", default=None,
                   help="Tier-4 differential diffusion: SRK Cantera mechanism "
                        "(e.g. data/wang2011_srk_v32.yaml). When given, the "
                        "per-cell Lewis numbers Le_Z(Z,gZ,c[,chi]) and Le_C are "
                        "tabulated from the real-fluid transport at each node's "
                        "(T,Y) and written as Le_Z/Le_C fields. Omit to keep the "
                        "legacy constant Le {Z;C} block.")
    p.add_argument("--diff-diff-transport", default="high-pressure-Chung",
                   help="Cantera transport model for the Tier-4 Le tabulation "
                        "(default: high-pressure-Chung, matching the solver).")
    p.add_argument("--soret-le", action="store_true",
                   help="fold the thermal-diffusion (Soret) flux of Z and c "
                        "into the tabulated effective Le_Z/Le_C (manifold-"
                        "projected). Captures Soret with NO extra solver term. "
                        "Default OFF (diagonal mixture-averaged Le only).")
    p.add_argument("--x-fuel", default=None,
                   help="fuel-stream mole-fraction string for the equilibrium "
                        "closure (default: kerosene surrogate). E.g. 'H2:1.0' "
                        "for the H2/O2 campaign.")
    p.add_argument("--x-ox", default=None,
                   help="oxidizer-stream mole-fraction string (default "
                        "'O2:1.0'). E.g. 'O2:0.7, H2O:0.3' for steam-diluted "
                        "oxy-hydrogen tables.")
    p.add_argument("--pv-species", default=None,
                   help="space/comma-separated progress-variable species, "
                        "overriding the kerosene default CO2 CO H2O H2. Must "
                        "match the C saved by the flamelet sweep (e.g. 'H2O' "
                        "for H2/O2).")
    p.add_argument("--z-st", type=float, default=None,
                   help="cluster the Z axis around this stoichiometric "
                        "mixture fraction (Roberts interior stretching). "
                        "Default: uniform axis (legacy).")
    p.add_argument("--smooth-c", type=float, default=0.0,
                   help="Gaussian smoothing of T + source in c (sigma in "
                        "cells; Z at half strength). 0=off (default). "
                        "Recommended ~1.0 for deployment: removes interpolant "
                        "faceting (T roughness 27->9 K, T_max within 0.8%) "
                        "while preserving peak/equilibrium/source-magnitude; "
                        "Y_k left intact.")
    p.add_argument("--z-beta", type=float, default=5.0,
                   help="Z-axis clustering strength (default 5.0)")
    args = p.parse_args()

    # stream/PV overrides (H2/O2 campaign etc.) -- module globals feed
    # equilibrium_closure() and the Tier-4 lewis block.
    global X_FUEL_SURROGATE, X_OX, PV_SPECIES
    if args.x_fuel:
        X_FUEL_SURROGATE = args.x_fuel
    if args.x_ox:
        X_OX = args.x_ox
    if args.pv_species:
        PV_SPECIES = tuple(args.pv_species.replace(",", " ").split())
    if args.x_fuel or args.x_ox or args.pv_species:
        print(f"[cfg] streams override: fuel=[{X_FUEL_SURROGATE}] "
              f"ox=[{X_OX}] PV={PV_SPECIES}")

    fl_dir = Path(args.flamelet_dir)
    if not fl_dir.is_dir():
        raise SystemExit(f"missing dir {fl_dir}")
    print(f"[load] {fl_dir}")

    fls, species = load_flamelet_dir(fl_dir)
    print(f"[load] {len(fls)} flamelets; species count = {len(species)}")
    if args.exclude_idx.strip():
        drop = {int(x) for x in args.exclude_idx.replace(",", " ").split()}
        def _fidx(fl):
            import re as _re
            m = _re.search(r"flamelet_(\d+)\.npz", Path(fl["path"]).name)
            return int(m.group(1)) if m else -1
        before = len(fls)
        fls = [fl for fl in fls if _fidx(fl) not in drop]
        print(f"[load] excluded idx {sorted(drop)}: {before} -> {len(fls)} flamelets")
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

    # Volumetric heat-release rate per flamelet point (Wang-2015 Fig.19c).
    compute_hrr(fls, args.yaml_file, species, P_ref)

    # Equilibrium burnt-end closure on the internal Zq grid (same grid the
    # beta-PDF convolution integrates over).
    Zq = np.linspace(0.0, 1.0, N_ZQ)
    eq = equilibrium_closure(args.yaml_file, Zq, P_ref, T_fuel, T_ox, species)

    z_cluster = None
    if args.z_st is not None:
        z_cluster = (args.z_st, args.z_beta)
    Z_axis, g_axis, C_axis, chi_axis, tables = build_tables(
        fls, species, n_chi=args.n_chi, eq=eq, chi_mode=args.chi_axis,
        z_cluster=z_cluster, smooth_c=args.smooth_c
    )

    # Tier-4: optional real-fluid differential-diffusion Lewis-number tables.
    if args.diff_diff_yaml:
        if "T" not in tables:
            raise SystemExit("[lewis] differential diffusion needs the T table")
        C_norm_onZ = None
        if args.soret_le and "C_norm_Zq" in eq:
            C_norm_onZ = np.interp(Z_axis, np.linspace(0.0, 1.0, N_ZQ),
                                   eq["C_norm_Zq"])
        LeZ, LeC, medZ, medC = compute_lewis_tables(
            tables, species, P_ref, args.diff_diff_yaml,
            transport_model=args.diff_diff_transport,
            soret=args.soret_le, Z_axis=Z_axis, C_axis=C_axis,
            C_norm=C_norm_onZ, x_fuel=X_FUEL_SURROGATE, x_ox=X_OX,
        )
        tables["Le_Z"] = LeZ
        tables["Le_C"] = LeC
        meta["le"] = (medZ, medC)
        meta["closure"] += " + tabulated-Le(Z,c) [Tier-4 diff-diff]"

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
