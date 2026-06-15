"""Extract Wang-2015 Fig19 metrics (T_max, flame thickness delta, peak heat-
release q_dot) from a RECONSTRUCTED a=1000 fgmFluid solution.

Usage: python3 extract_fig19.py <caseDir> <time> [nx] [ny]
Centerline (j = ny//2) T(x) profile -> T_max; delta = FWHM of (T - T_inlet) in
mm; q_dot = peak tabulated hrr on the centerline [W/m^3]. blockMesh cell order
is i-fastest (idx = i + j*nx), so the centerline row is [j*nx : (j+1)*nx]."""
import re, sys, numpy as np
from pathlib import Path

case = Path(sys.argv[1]); t = sys.argv[2]
nx = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
ny = int(sys.argv[4]) if len(sys.argv) > 4 else 20
Lx = 0.02

def rd(fld):
    f = case / t / fld
    if not f.exists():
        return None
    m = re.search(r'List<scalar>\s*\n(\d+)\s*\n\(\n(.*?)\n\)\s*;', open(f).read(), re.S)
    if not m:
        mm = re.search(r'internalField\s+uniform\s+([-0-9.eE+]+)', open(f).read())
        return None if not mm else np.full(nx*ny, float(mm.group(1)))
    return np.array([float(x) for x in m.group(2).split()])

T = rd('T'); hrr = rd('hrr'); C = rd('C')
if T is None:
    sys.exit(f"no reconstructed T at {case}/{t} (run reconstructPar first)")
j = ny // 2
xc = (np.arange(nx) + 0.5) * (Lx / nx)
Tline = T[j*nx:(j+1)*nx]
Tmax = Tline.max(); Tinlet = Tline[[0, -1]].min()
half = Tinlet + 0.5*(Tmax - Tinlet)
above = np.where(Tline >= half)[0]
delta_mm = (xc[above[-1]] - xc[above[0]]) * 1e3 if len(above) > 1 else 0.0
qdot = hrr[j*nx:(j+1)*nx].max() if hrr is not None else float('nan')
cmax = C[j*nx:(j+1)*nx].max() if C is not None else float('nan')
print(f"{case.name} t={t}: T_max={Tmax:.0f} K  delta={delta_mm:.4f} mm  "
      f"q_dot_peak={qdot:.3e} W/m3  C_max={cmax:.3f}  "
      f"x_flame={xc[Tline.argmax()]*1e3:.2f} mm")
