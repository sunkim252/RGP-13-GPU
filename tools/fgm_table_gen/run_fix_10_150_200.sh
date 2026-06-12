#!/usr/bin/env bash
# Fill the remaining Wang-2015 pressure points:
#   10 atm  : redo with GENTLER strain steps (x1.10 instead of x1.25) --
#             the x1.25 sweep walked the flame through partially-extinguishing
#             transients (chi_st collapsed 5.7 -> 0.09) before extinction at
#             step 7; finer steps keep each accepted state on the burning
#             branch up to the genuine extinction point.
#   150 atm : pressure continuation from the converged 100 atm stage-A
#             checkpoint (cold start cannot find the razor-thin flame).
#   200 atm : continuation from the NEW 150 atm stage-A checkpoint.
set -u
cd "$(dirname "$0")"
source /home/sunkim/miniforge3/etc/profile.d/conda.sh
conda activate ct-env3

LOG=data/fix_10_150_200.log
echo "=== fix sweep start ===" > "$LOG"

echo "############ 10 atm redo (mdot x1.10) ############" | tee -a "$LOG"
rm -rf data/flamelets_dualgas_P10atm data/ckpt_dualgas_P10atm
python3 -u 05_flamelet_sweep_dualgas.py --pressure-atm 10 --restart \
    --mdot-factor 1.10 --log 0 >> "$LOG" 2>&1
echo "==> 10 atm: $(ls data/flamelets_dualgas_P10atm/*.npz 2>/dev/null | wc -l) flamelets" | tee -a "$LOG"

echo "############ 150 atm (warm from 100 atm) ############" | tee -a "$LOG"
rm -rf data/flamelets_dualgas_P150atm data/ckpt_dualgas_P150atm
python3 -u 05_flamelet_sweep_dualgas.py --pressure-atm 150 --restart --log 0 \
    --warm-from data/ckpt_dualgas_P100atm/stageA_ideal.yaml >> "$LOG" 2>&1
n150=$(ls data/flamelets_dualgas_P150atm/*.npz 2>/dev/null | wc -l)
echo "==> 150 atm: $n150 flamelets" | tee -a "$LOG"

if [ "$n150" -gt 0 ]; then
  echo "############ 200 atm (warm from 150 atm) ############" | tee -a "$LOG"
  rm -rf data/flamelets_dualgas_P200atm data/ckpt_dualgas_P200atm
  python3 -u 05_flamelet_sweep_dualgas.py --pressure-atm 200 --restart --log 0 \
      --warm-from data/ckpt_dualgas_P150atm/stageA_ideal.yaml >> "$LOG" 2>&1
  echo "==> 200 atm: $(ls data/flamelets_dualgas_P200atm/*.npz 2>/dev/null | wc -l) flamelets" | tee -a "$LOG"
else
  echo "==> 150 atm failed; skipping 200 atm" | tee -a "$LOG"
fi
echo "=== fix sweep DONE ===" | tee -a "$LOG"
