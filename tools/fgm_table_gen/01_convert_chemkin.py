"""Convert Wang 2011 CHEMKIN kerosene mechanism to Cantera YAML with SRK EOS.

Pipeline:
  1) ck2yaml: 3-file CHEMKIN (reactions/thermo/transport) -> ideal-gas YAML
  2) Patch phase block: ideal-gas -> Soave-Redlich-Kwong
  3) Inject explicit critical-parameters for fuel surrogate species
     (NC10H22, PHC3H7, CYC9H18) using values from DagautSRKReduced.yaml.
     Other species are auto-resolved by Cantera from its built-in
     critical-properties.yaml database. Species missing from both will
     fail at Solution() time -- those need manual addition.

Run from Windows side (ct-env2):
    (ct-env2)$ python 01_convert_chemkin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml  # PyYAML, ships with cantera env

try:
    from cantera import ck2yaml
except ImportError as e:
    sys.exit(
        "Cantera not importable. Activate ct-env2 first:\n"
        "    conda activate ct-env2\n"
        f"({e})"
    )


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]  # .../RGP-13
MECH_DIR = (
    REPO_ROOT
    / "references"
    / "FGM"
    / "ScienceDirect_files_20May2026_05-36-08.515"
)

INPUT_FILE = MECH_DIR / "1-s2.0-S0010218011001659-mmc1.txt"
THERMO_FILE_RAW = MECH_DIR / "1-s2.0-S0010218011001659-mmc2.txt"
TRANSPORT_FILE = MECH_DIR / "1-s2.0-S0010218011001659-mmc3.txt"

OUT_DIR = HERE / "data"
OUT_DIR.mkdir(exist_ok=True)
THERMO_FILE = OUT_DIR / "wang2011_thermo_clean.txt"  # cleaned copy used by ck2yaml
IDEAL_YAML = OUT_DIR / "wang2011_ideal.yaml"
SRK_YAML = OUT_DIR / "wang2011_srk.yaml"

# Thermo entries to drop entirely. These appear at the head of mmc2.txt but
# (a) have malformed element-composition columns ("0   00   00..." — no
# element symbol present) and (b) are not in the mechanism's SPECIES list,
# so dropping them is safe.
SKIP_THERMO_SPECIES = {"HE", "NE"}


# Explicit critical parameters: kerosene surrogate fuels (3) + species
# whose values in cantera's built-in critical-properties.yaml may be
# missing or inconsistent for SRK use (radicals, minor combustion
# intermediates). Values copied verbatim from DagautSRKReduced.yaml
# shipped with ct-env2 to keep our SRK manifold consistent with the
# reduced Dagaut SRK convention.
EXPLICIT_CRITICAL = {
    # ---- Fuel surrogate ----
    "NC10H22":  dict(Tc=617.8, Pc=2.11e6, omega=0.488),   # n-decane
    "PHC3H7":   dict(Tc=638.4, Pc=3.20e6, omega=0.344),   # n-propylbenzene
    "CYC9H18":  dict(Tc=629.0, Pc=2.85e6, omega=0.354),   # n-propylcyclohexane
    # ---- Common combustion species (baseline from DagautSRKReduced) ----
    "CH4":  dict(Tc=190.56,  Pc=4.599e6, omega=0.011),
    "H2":   dict(Tc=33.18,   Pc=1.30e6,  omega=-0.220),
    "O2":   dict(Tc=154.58,  Pc=5.043e6, omega=0.022),
    "CO":   dict(Tc=134.45,  Pc=3.499e6, omega=0.066),
    "CO2": dict(Tc=304.18,   Pc=7.38e6,  omega=0.228),
    "H2O":  dict(Tc=647.1,   Pc=2.206e7, omega=0.354),
    "H2O2": dict(Tc=722.0,   Pc=2.06e7,  omega=0.370),
    "N2":   dict(Tc=126.19,  Pc=3.398e6, omega=0.039),
    # ---- Radicals (nominal: low mole-frac, EOS contribution small) ----
    "H":    dict(Tc=181.13, Pc=2.77e7, omega=0.0085),
    "O":    dict(Tc=99.94,  Pc=6.33e6, omega=0.0085),
    "OH":   dict(Tc=99.94,  Pc=6.33e6, omega=0.0085),
    "HCO":  dict(Tc=643.05, Pc=1.80e7, omega=0.0468),
    "HO2":  dict(Tc=134.16, Pc=4.28e6, omega=0.0085),
}


# Canonical critical parameters by structural class. Used as a fallback for
# radicals and intermediates whose species name does not appear in
# EXPLICIT_CRITICAL above. EOS contribution scales with mole fraction; these
# species are minor inside the flame so using parent-molecule values is a
# standard approximation in the FPV literature.
CANONICAL = {
    # --- n-alkanes (linear CnH2n+2) by carbon count ---
    "alkane": {
        1:  dict(Tc=190.56, Pc=4.60e6, omega=0.011),   # methane
        2:  dict(Tc=305.32, Pc=4.87e6, omega=0.099),   # ethane
        3:  dict(Tc=369.83, Pc=4.25e6, omega=0.152),   # propane
        4:  dict(Tc=425.12, Pc=3.80e6, omega=0.200),   # n-butane
        5:  dict(Tc=469.70, Pc=3.37e6, omega=0.251),   # n-pentane
        6:  dict(Tc=507.60, Pc=3.03e6, omega=0.299),   # n-hexane
        7:  dict(Tc=540.10, Pc=2.74e6, omega=0.349),   # n-heptane
        8:  dict(Tc=568.70, Pc=2.49e6, omega=0.398),   # n-octane
        9:  dict(Tc=594.60, Pc=2.29e6, omega=0.445),   # n-nonane
        10: dict(Tc=617.80, Pc=2.11e6, omega=0.488),   # n-decane
    },
    # --- 1-alkenes / alkene radicals ---
    "alkene": {
        2:  dict(Tc=282.34, Pc=5.04e6, omega=0.087),   # ethylene
        3:  dict(Tc=365.57, Pc=4.67e6, omega=0.142),   # propylene
        4:  dict(Tc=419.60, Pc=4.02e6, omega=0.194),   # 1-butene
        5:  dict(Tc=464.78, Pc=3.56e6, omega=0.233),   # 1-pentene
        6:  dict(Tc=504.00, Pc=3.21e6, omega=0.281),   # 1-hexene
        7:  dict(Tc=537.40, Pc=2.83e6, omega=0.358),   # 1-heptene
    },
    # --- alkynes / acetylenics ---
    "alkyne": {
        2:  dict(Tc=308.30, Pc=6.14e6, omega=0.190),   # acetylene
        3:  dict(Tc=402.40, Pc=5.63e6, omega=0.218),   # propyne
        4:  dict(Tc=454.00, Pc=5.06e6, omega=0.215),   # vinylacetylene
    },
    # --- aromatics ---
    "benzene":        dict(Tc=562.05, Pc=4.895e6, omega=0.212),
    "toluene":        dict(Tc=591.80, Pc=4.106e6, omega=0.264),
    "ethylbenzene":   dict(Tc=617.20, Pc=3.61e6,  omega=0.304),
    "propylbenzene":  dict(Tc=638.40, Pc=3.20e6,  omega=0.344),
    "styrene":        dict(Tc=648.00, Pc=3.99e6,  omega=0.297),
    "phenol":         dict(Tc=694.25, Pc=6.13e6,  omega=0.444),
    "benzaldehyde":   dict(Tc=695.00, Pc=4.65e6,  omega=0.346),
    # --- cyclics ---
    "cyclopentadiene":    dict(Tc=509.00, Pc=4.85e6, omega=0.087),
    "cyclohexane":        dict(Tc=553.40, Pc=4.08e6, omega=0.212),
    "propylcyclohexane":  dict(Tc=629.00, Pc=2.85e6, omega=0.354),
    # --- oxygenates ---
    "formaldehyde":  dict(Tc=408.00, Pc=6.59e6, omega=0.218),
    "acetaldehyde":  dict(Tc=466.00, Pc=5.57e6, omega=0.291),
    "methanol":      dict(Tc=512.64, Pc=8.10e6, omega=0.557),  # only for CH3OH-like
    "ketene":        dict(Tc=380.00, Pc=6.40e6, omega=0.220),
    "acrolein":      dict(Tc=506.00, Pc=5.04e6, omega=0.330),
}


def _composition_counts(comp):
    """Return (nC, nH, nO, nN) given the YAML composition mapping."""
    return (
        int(comp.get("C", 0)),
        int(comp.get("H", 0)),
        int(comp.get("O", 0)),
        int(comp.get("N", 0)),
    )


def infer_critical(name, composition):
    """Return critical-parameters dict for a species using a class-of-
    species fallback. Returns None if no reasonable class can be inferred.
    """
    nC, nH, nO, nN = _composition_counts(composition)
    upper = name.upper()

    # ---- Named aromatics / known small molecules ----
    if upper == "TOLUEN":
        return CANONICAL["toluene"]
    if upper == "STYREN":
        return CANONICAL["styrene"]
    if upper == "C6H6":
        return CANONICAL["benzene"]
    if upper == "C6H5":           # phenyl radical
        return CANONICAL["benzene"]
    if upper == "C6H5OH":
        return CANONICAL["phenol"]
    if upper == "C6H5O":          # phenoxy radical
        return CANONICAL["phenol"]
    if upper == "CPD":            # 1,3-cyclopentadiene
        return CANONICAL["cyclopentadiene"]
    if upper == "ACROL":          # acrolein
        return CANONICAL["acrolein"]
    if upper in ("CH2O",):
        return CANONICAL["formaldehyde"]
    if upper in ("CH3HCO",):
        return CANONICAL["acetaldehyde"]
    if upper in ("CH2HCO", "CH3CO", "C2H3CO"):  # acyl/oxo radicals
        return CANONICAL["acetaldehyde"]
    if upper in ("CH2CO", "HCCO"):  # ketene/ketenyl
        return CANONICAL["ketene"]
    if upper in ("CH3O", "CH3O2", "CH2OH"):  # methoxy / methyl peroxy / hydroxymethyl
        return CANONICAL["formaldehyde"]
    if upper in ("C3H5O",):
        return CANONICAL["acrolein"]
    if upper in ("MEALL",):       # methylallyl
        return CANONICAL["alkene"][4]   # ~1-butene
    if upper in ("C2H4O2H", "C2H5O2"):
        return CANONICAL["acetaldehyde"]

    # ---- Phenyl / propylbenzene side-chain family (Wang naming "PH*", "*PHC*", "*PHPROPY") ----
    if "PHPROPY" in upper:
        return CANONICAL["propylbenzene"]
    if "PHC3H" in upper or "APHC3H" in upper or "BPHC3H" in upper or "CPHC3H" in upper:
        return CANONICAL["propylbenzene"]
    if "APHC2H" in upper:
        return CANONICAL["ethylbenzene"]
    if upper.startswith("PHCH") or upper.startswith("PHHCO") or upper.startswith("PHCO"):
        return CANONICAL["toluene"]
    if upper.startswith("PH"):
        return CANONICAL["toluene"]
    if "PHC" in upper:
        return CANONICAL["propylbenzene"]

    # ---- Cyclic family ----
    if "CYC9" in upper or "C9H17C" in upper or "C9H18C" in upper or upper == "AC9H18":
        return CANONICAL["propylcyclohexane"]
    if "CYC6" in upper:
        return CANONICAL["cyclohexane"]
    if "C5H4O" in upper or "C5H4OH" in upper or "C5H5O" in upper:
        return CANONICAL["cyclopentadiene"]
    if upper.startswith("C5H7") or "C5H813" in upper or "C5H913" in upper or upper == "C5H5":
        return CANONICAL["cyclopentadiene"]

    # ---- Hex-ring or hex-chain (C6H11xx series of Wang are cyclohexenyl/hexenyl) ----
    if upper.startswith("C6H11"):
        return CANONICAL["cyclohexane"]

    # ---- Pure hydrocarbon by composition ----
    if nC > 0 and nO == 0 and nN == 0:
        # alkane: H >= 2C  -> use alkane class indexed by C
        if nH >= 2 * nC:
            base = CANONICAL["alkane"]
            return base.get(nC) or base[10]
        # alkene / alkyne by H/C ratio
        if nH >= 2 * nC - 2:
            base = CANONICAL["alkene"]
            return base.get(nC) or base[max(base)]
        if nH >= 2 * nC - 4:
            base = CANONICAL["alkyne"]
            return base.get(nC) or base[max(base)]
        # very H-poor / aromatic-like -> use benzene as proxy
        return CANONICAL["benzene"]

    # ---- Fallback: cannot classify ----
    return None


def assign_fallback_critical(species_list, name_idx):
    """Walk all species without a critical-parameters block and assign
    fallback values inferred from name + composition. Returns the list of
    species that the fallback could not classify (still need manual input).
    """
    inferred = 0
    unresolved = []
    for sp in species_list:
        if "critical-parameters" in sp:
            continue
        name = sp.get("name", "?")
        comp = sp.get("composition", {})
        params = infer_critical(name, comp)
        if params is None:
            unresolved.append(name)
            continue
        sp["critical-parameters"] = {
            "critical-temperature": params["Tc"],
            "critical-pressure": params["Pc"],
            "acentric-factor": params["omega"],
        }
        inferred += 1
    print(f"[patch] fallback critical-parameters inferred for {inferred} species")
    if unresolved:
        print(f"[patch] WARN: {len(unresolved)} species unresolved: {unresolved}")
    return unresolved


def clean_thermo_file():
    """Strip orphan/malformed thermo entries and post-END junk lines.

    Writes a sanitized copy of mmc2.txt to OUT_DIR for ck2yaml to consume.
    """
    print(f"[clean] reading {THERMO_FILE_RAW}")
    raw = THERMO_FILE_RAW.read_text(errors="replace").splitlines()

    out_lines = []
    i = 0
    in_thermo = False
    skipped_species = []
    while i < len(raw):
        line = raw[i]
        stripped = line.strip()
        upper = stripped.upper()

        if not in_thermo:
            out_lines.append(line)
            if upper.startswith("THERMO"):
                in_thermo = True
            i += 1
            continue

        # Within THERMO block. Each species record is exactly 4 lines whose
        # 80th column carries the index 1/2/3/4. Line 1 holds the species
        # name in cols 1-18.
        if upper == "END":
            out_lines.append(line)
            # Anything past the END line is post-block junk (citations,
            # comments). Drop the remainder.
            break

        # Header line of a record: skip the optional temperature-range line
        # that appears once at the top of THERMO (no "1" in col 80).
        is_record_header = len(line) >= 80 and line[79] == "1"
        if not is_record_header:
            out_lines.append(line)
            i += 1
            continue

        sp_name = line[:18].strip().split()[0] if line.strip() else ""
        block = raw[i : i + 4]
        if sp_name in SKIP_THERMO_SPECIES:
            skipped_species.append(sp_name)
        else:
            out_lines.extend(block)
        i += 4

    THERMO_FILE.write_text("\n".join(out_lines) + "\n")
    print(f"[clean] wrote {THERMO_FILE}")
    if skipped_species:
        print(f"[clean] dropped malformed entries: {skipped_species}")


def run_ck2yaml():
    """Run cantera's ck2yaml converter (3-file mode)."""
    print(f"[ck2yaml] input    : {INPUT_FILE}")
    print(f"[ck2yaml] thermo   : {THERMO_FILE}")
    print(f"[ck2yaml] transport: {TRANSPORT_FILE}")
    print(f"[ck2yaml] output   : {IDEAL_YAML}")

    if not INPUT_FILE.exists():
        sys.exit(f"missing: {INPUT_FILE}")

    # ck2yaml.main wants a list of CLI-style args.
    args = [
        f"--input={INPUT_FILE}",
        f"--thermo={THERMO_FILE}",
        f"--transport={TRANSPORT_FILE}",
        f"--output={IDEAL_YAML}",
        "--permissive",  # Wang 2011 has duplicate decls -> tolerate
    ]
    rc = ck2yaml.main(args)
    if rc not in (None, 0):
        sys.exit(f"ck2yaml failed (rc={rc})")
    print(f"[ck2yaml] OK -> {IDEAL_YAML}")


def patch_to_srk():
    """Load ideal-gas YAML, switch phase to SRK, inject critical params."""
    print(f"[patch] reading {IDEAL_YAML}")
    with open(IDEAL_YAML, "r") as fh:
        doc = yaml.safe_load(fh)

    # ---- Phase block: ideal-gas -> Soave-Redlich-Kwong ----
    phases = doc.get("phases", [])
    if not phases:
        sys.exit("no phases in YAML; ck2yaml output malformed?")
    for phase in phases:
        if phase.get("thermo") == "ideal-gas":
            phase["thermo"] = "Soave-Redlich-Kwong"
            print(f"[patch] phase {phase.get('name','?')}: ideal-gas -> SRK")

    # ---- Inject critical-parameters for surrogate species ----
    species_list = doc.get("species", [])
    name_idx = {s.get("name"): i for i, s in enumerate(species_list)}

    injected = 0
    for sp_name, params in EXPLICIT_CRITICAL.items():
        idx = name_idx.get(sp_name)
        if idx is None:
            # Not in this mechanism's species list — fine, just skip.
            continue
        sp = species_list[idx]
        sp["critical-parameters"] = {
            "critical-temperature": params["Tc"],
            "critical-pressure": params["Pc"],
            "acentric-factor": params["omega"],
        }
        injected += 1
    print(f"[patch] explicit critical-parameters injected for {injected} species")
    missing = [n for n in EXPLICIT_CRITICAL if n not in name_idx]
    if missing:
        print(f"[patch] (not in mech, skipped: {missing})")

    # ---- Fallback: infer critical-parameters for the rest ----
    unresolved = assign_fallback_critical(species_list, name_idx)

    # ---- Write ----
    with open(SRK_YAML, "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=None)
    print(f"[patch] OK -> {SRK_YAML}")
    if unresolved:
        print(
            "[patch] HINT: smoke test will fail on the unresolved species "
            "listed above. Add them to EXPLICIT_CRITICAL or extend infer_critical()."
        )


def smoke_test():
    """Try to instantiate the SRK mechanism."""
    try:
        import cantera as ct
    except ImportError:
        return
    print(f"[test] loading {SRK_YAML.name} into ct.Solution...")
    try:
        gas = ct.Solution(str(SRK_YAML))
    except Exception as e:
        # Most likely failure: critical-parameters missing for some species
        print(f"[test] FAILED: {e}")
        print(
            "[test] Resolution: add critical-parameters for the species "
            "named in the error to EXPLICIT_CRITICAL above, then re-run."
        )
        return
    print(f"[test] OK: {gas.n_species} species, {gas.n_reactions} reactions")
    print(f"[test] thermo model: {gas.thermo_model}")


if __name__ == "__main__":
    clean_thermo_file()
    run_ck2yaml()
    patch_to_srk()
    smoke_test()
