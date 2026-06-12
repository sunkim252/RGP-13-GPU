#!/usr/bin/env bash
# Wang-Huo-Yang 2015 pressure-sweep flamelet generation.
# Generates dual-gas (ideal structure + SRK+Chung props) counterflow flamelets
# at each pressure into data/flamelets_dualgas_P<atm>atm/, then builds a 4-D
# table per pressure. Sequential so the conda single-thread BLAS isn't oversub.
set -u
cd "$(dirname "$0")"
source /home/sunkim/miniforge3/etc/profile.d/conda.sh
conda activate ct-env3

PRESSURES="1 10 25 50 100 150 200"
LOG=data/wang2015_sweep.log
echo "=== Wang2015 sweep start: $PRESSURES atm ===" > "$LOG"

for P in $PRESSURES; do
  echo "############ PRESSURE ${P} atm ############" | tee -a "$LOG"
  python3 -u 05_flamelet_sweep_dualgas.py --pressure-atm "$P" --restart --log 0 \
      >> "$LOG" 2>&1
  rc=$?
  fl="data/flamelets_dualgas_P${P}atm"
  n=$(ls "$fl"/*.npz 2>/dev/null | wc -l)
  echo "==> ${P} atm: rc=$rc, $n flamelets in $fl" | tee -a "$LOG"
  # Table build is deferred: tables are large (106-species) and only needed once
  # the OpenFOAM counterflow extraction is validated, at which point we build
  # reduced-species tables per pressure. Flamelets alone are ~33 MB total.
done
echo "=== Wang2015 sweep DONE ===" | tee -a "$LOG"
