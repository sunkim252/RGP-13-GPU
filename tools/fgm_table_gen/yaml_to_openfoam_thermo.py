"""Convert a Cantera SRK (Redlich-Kwong) YAML mechanism into an OpenFOAM
real-fluid thermo dictionary (`thermo.compressibleGas`-style) usable by the
RGP-13 SRKchungTaka / SRKelyHanley mixtures.

Per species the OpenFOAM entry needs:
  specie         { molWeight }
  rfProperties   { penelouxShift; Tc; Pc; Vc[cm3/mol]; omega; kappai; miui; sigmvi }
  thermodynamics { Tlow Thigh Tcommon highCpCoeffs(7) lowCpCoeffs(7) }   # JANAF
  transport      { As Ts }                                               # Sutherland
  elements       { ... }

Sources:
  - molWeight, composition, NASA-7 coeffs, Lennard-Jones (well-depth, diameter),
    dipole : from the Cantera Solution.
  - Tc, Pc, omega : from each species' YAML `critical-parameters`.
  - Vc      : estimated, Vc = Zc R Tc / Pc with Zc = 0.2918 - 0.0928 omega
              (Pitzer); validated against O2 (74.2 vs tabulated 73.5 cm3/mol).
  - sigmvi  : Fuller diffusion volume from atomic contributions.
  - As, Ts  : Sutherland fit to the Chapman-Enskog dilute viscosity from the
              Lennard-Jones parameters (2-point fit at 300/1500 K).
  - miui (dipole) -> Debye; kappai (association) -> 0 (water gets a small value).

Run:
  python yaml_to_openfoam_thermo.py --yaml data/wang2011_srk_v32.yaml \
         --out ../../testCases/.../constant/thermo.wang2011
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import cantera as ct

R = 8.314462618          # J/mol/K
NA = 6.02214076e23
KB = 1.380649e-23

# Fuller (1966) atomic diffusion-volume increments [cm^3/mol]; special diatomics
FULLER_ATOM = {"C": 15.9, "H": 2.31, "O": 6.11, "N": 4.54, "S": 22.9}
FULLER_MOL = {"O2": 16.3, "N2": 18.5, "H2": 6.12, "CO": 18.0, "CO2": 26.9,
              "H2O": 13.1, "H2": 6.12}


def fuller_sigmv(name, comp):
    if name in FULLER_MOL:
        return FULLER_MOL[name]
    return sum(FULLER_ATOM.get(el, 15.9) * n for el, n in comp.items())


def vc_estimate(Tc, Pc, omega):
    """Critical volume [cm3/mol] from Zc = 0.2918 - 0.0928 omega (Pitzer)."""
    Zc = 0.2918 - 0.0928 * omega
    return Zc * R * Tc / Pc * 1e6   # m3/mol -> cm3/mol


def ce_viscosity(MW, eps_k, sigma_A, T):
    """Chapman-Enskog dilute viscosity [Pa.s]. eps_k in K, sigma in Angstrom."""
    Tstar = T / eps_k
    # Neufeld collision integral Omega(2,2)*
    A, B, C, D, E, F = 1.16145, 0.14874, 0.52487, 0.77320, 2.16178, 2.43787
    omega22 = (A / Tstar**B + C / math.exp(D * Tstar)
               + E / math.exp(F * Tstar))
    return 2.6693e-6 * math.sqrt(MW * T) / (sigma_A**2 * omega22)


def sutherland_fit(MW, eps_k, sigma_A):
    """Fit mu = As*sqrt(T)/(1+Ts/T) to CE viscosity at 300 and 1500 K."""
    T1, T2 = 300.0, 1500.0
    m1, m2 = (ce_viscosity(MW, eps_k, sigma_A, T1),
              ce_viscosity(MW, eps_k, sigma_A, T2))
    # mu/sqrt(T) = As/(1+Ts/T) -> 1/(mu/sqrt(T)) = (1+Ts/T)/As
    y1, y2 = math.sqrt(T1) / m1, math.sqrt(T2) / m2     # = (1+Ts/T)/As
    # y = (1/As) + (Ts/As)/T  -> linear in 1/T
    slope = (y2 - y1) / (1.0 / T2 - 1.0 / T1)
    inv_As = y1 - slope * (1.0 / T1)
    As = 1.0 / inv_As
    Ts = slope * As
    return As, Ts


def nasa7(sp):
    """Return (Tlow, Thigh, Tcommon, high[7], low[7]) from a NasaPoly2.

    Thigh is floored at 5000 K: OpenFOAM's janaf mixture takes the MIN of
    Thigh over ALL species in the multicomponent mixture and its limit()
    clamps the temperature solve there -- 15 fuel-side intermediates fitted
    only to 3000 K pinned the WHOLE flame at exactly 3000.0 K (equilibrium
    is ~3790 K at 50 atm). Extrapolating their high-T branch is benign:
    those intermediates carry ~zero mass fraction above ~1500 K."""
    c = sp.thermo.coeffs           # [Tmid, a1..a7(high), a1..a7(low)]
    Tmid = c[0]
    high = list(c[1:8])
    low = list(c[8:15])
    return sp.thermo.min_temp, max(sp.thermo.max_temp, 5000.0), Tmid, high, low


def fmt_coeffs(v):
    return "( " + " ".join(f"{x:.8g}" for x in v) + " )"


def species_block(g, name, assoc, crit):
    i = g.species_index(name)
    sp = g.species(name)
    MW = g.molecular_weights[i]
    comp = {el: int(round(n)) for el, n in sp.composition.items()}
    # Critical parameters come from the RAW YAML: Cantera's RK phase consumes
    # the acentric-factor internally and does NOT round-trip it through
    # Species.input_data (it would come back as 0), so we read the file directly.
    cp = crit[name]
    Tc = float(cp["critical-temperature"])
    Pc = float(cp["critical-pressure"])
    omega = float(cp.get("acentric-factor", 0.0))
    Vc = vc_estimate(Tc, Pc, omega)
    sigmv = fuller_sigmv(name, comp)
    tp = sp.transport
    eps_k = tp.well_depth / KB                 # K
    sigma_A = tp.diameter * 1e10               # Angstrom
    dipole_D = (tp.dipole / 3.33564e-30) if tp.dipole else 0.0  # C.m -> Debye
    As, Ts = sutherland_fit(MW, eps_k, sigma_A)
    Tlow, Thigh, Tcommon, high, low = nasa7(sp)
    el = "\n".join(f"        {e:<3} {n};" for e, n in comp.items())
    return f"""{name}
{{
    specie
    {{
        molWeight       {MW:.5f};
    }}
    rfProperties
    {{
        penelouxShift   true;
        Tc              {Tc:.4g};
        Pc              {Pc:.5g};
        Vc              {Vc:.4f};
        omega           {omega:.4g};
        kappai          {assoc.get(name, 0.0):.4g};
        miui            {dipole_D:.4g};
        sigmvi          {sigmv:.4g};
    }}
    thermodynamics
    {{
        Tlow            {Tlow:.0f};
        Thigh           {Thigh:.0f};
        Tcommon         {Tcommon:.0f};
        highCpCoeffs    {fmt_coeffs(high)};
        lowCpCoeffs     {fmt_coeffs(low)};
    }}
    transport
    {{
        As              {As:.6g};
        Ts              {Ts:.6g};
    }}
    elements
    {{
{el}
    }}
}}
"""


HEADER = """/*--------------------------------*- C++ -*----------------------------------*\\
| Generated by yaml_to_openfoam_thermo.py from a Cantera SRK YAML mechanism.   |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermo.compressibleGas;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = ct.Solution(a.yaml)
    import yaml as _yaml
    raw = _yaml.safe_load(open(a.yaml))
    crit = {s["name"]: s["critical-parameters"] for s in raw["species"]}
    # small association factor for strongly polar / H-bonding species
    assoc = {"H2O": 0.076, "OH": 0.0}
    blocks = [species_block(g, nm, assoc, crit) for nm in g.species_names]
    body = (HEADER
            + f"species {g.n_species} ( {' '.join(g.species_names)} );\n\n"
            + "\n".join(blocks))
    Path(a.out).write_text(body)
    print(f"[ok] wrote {a.out}: {g.n_species} species")


if __name__ == "__main__":
    main()
