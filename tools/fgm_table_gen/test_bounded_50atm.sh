#!/usr/bin/env bash
# End-to-end C-boundedness test at 50 atm with the equilibrium-closed,
# normalized-c table built from unity-Lewis flamelets.
#   1. build the normalized table into the wang2015_P50atm case
#   2. reset + decompose + run the parallel counterflow case
#   3. report max(C) (must stay ~<=1) and max(T) (~equilibrium 3790 K)
set -eu
cd "$(dirname "$0")"
source /home/sunkim/miniforge3/etc/profile.d/conda.sh
conda activate ct-env3

CASE_H=../../testCases/wang2015_P50atm                      # host path
CASE_C=/home/openfoam/RGP-13/RGP-13-realFluid/testCases/wang2015_P50atm

echo "=== [1/3] build normalized table (unity-Le flamelets) ==="
python3 -u 04_build_fgm_table_4d.py \
    --flamelet-dir data/flamelets_dualgas_P50atm \
    --out "$CASE_H/constant/fgmProperties"

echo "=== [2/3] reset + parallel run ==="
( cd "$CASE_H" && rm -rf processor* 0.[0-9]* postProcessing )
bash /home/sunkim/openfoam/RGP-13/test/of13.sh \
  "cd $CASE_C && decomposePar -force > log.decomposePar 2>&1 \
   && mpirun -np 8 foamRun -parallel > log.foamRun.par 2>&1 \
   && reconstructPar -latestTime > log.reconstructPar 2>&1"

echo "=== [3/3] boundedness check ==="
python3 - <<'PY'
import re, numpy as np, glob, os
os.chdir("../../testCases/wang2015_P50atm")
ts = [d[:-1] for d in glob.glob("[0-9]*/")
      if re.match(r"^[\d.]+$", d[:-1]) and float(d[:-1]) > 0]
L = sorted(ts, key=float)[-1] + "/"
def internal(fn):
    t = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<scalar>\s*\n\s*\d+\s*\n'
                  r'\(\s*(.*?)\)\s*;', t, re.S)
    if m:
        return np.array([float(x) for x in m.group(1).split()])
    m = re.search(r'internalField\s+uniform\s+([-\d.eE+]+)', t)
    return np.array([float(m.group(1))])
T = internal(L + "T"); C = internal(L + "C"); Z = internal(L + "Z")
print(f"@t={L[:-1]}  T_max={T.max():.0f} K  C_max={C.max():.4f}  "
      f"C_mean={C.mean():.4f}  Z in [{Z.min():.3g},{Z.max():.3g}]")
ok_c = C.max() < 1.05
ok_t = 3300.0 < T.max() < 3900.0
print("C bounded:", "PASS" if ok_c else "FAIL",
      "| T physical:", "PASS" if ok_t else "FAIL",
      "(equilibrium Tmax ref ~3790 K @50 atm)")
PY
