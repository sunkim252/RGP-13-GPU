#!/usr/bin/env bash
# Regenerate ALL dual-gas flamelets with unity-Lewis structure transport
# (TRANSPORT_IDEAL = unity-Lewis-number in 05_flamelet_sweep_dualgas.py),
# replacing the mixture-averaged sets so c = C/C_eq stays <= 1.
# Order: 50 atm first (OpenFOAM C-boundedness test), then 1 atm (diagnosis
# pair), then the rest of the Wang-2015 sweep, then the 52.5 bar production
# point (default pressure when --pressure-atm is omitted).
set -u
cd "$(dirname "$0")"
source /home/sunkim/miniforge3/etc/profile.d/conda.sh
conda activate ct-env3

LOG=data/unityLe_regen.log
echo "=== unity-Lewis regeneration start ===" > "$LOG"

for P in 50 1 10 25 100 150 200; do
  echo "############ PRESSURE ${P} atm (unity-Le) ############" | tee -a "$LOG"
  rm -rf "data/flamelets_dualgas_P${P}atm" "data/ckpt_dualgas_P${P}atm"
  python3 -u 05_flamelet_sweep_dualgas.py --pressure-atm "$P" --restart --log 0 \
      >> "$LOG" 2>&1
  n=$(ls "data/flamelets_dualgas_P${P}atm"/*.npz 2>/dev/null | wc -l)
  echo "==> ${P} atm: rc=$?, $n flamelets" | tee -a "$LOG"
done

echo "############ 52.5 bar production (unity-Le) ############" | tee -a "$LOG"
rm -rf data/flamelets_dualgas data/ckpt_dualgas
python3 -u 05_flamelet_sweep_dualgas.py --restart --log 0 >> "$LOG" 2>&1
n=$(ls data/flamelets_dualgas/*.npz 2>/dev/null | wc -l)
echo "==> 52.5 bar: rc=$?, $n flamelets" | tee -a "$LOG"
echo "=== unity-Lewis regeneration DONE ===" | tee -a "$LOG"
