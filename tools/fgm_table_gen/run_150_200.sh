#!/usr/bin/env bash
# 150/200 atm pressure-continuation runs ONLY (10 atm already done: 64
# flamelets -- do NOT touch it). Patched warm-from path: REFINE_COARSE
# re-imposed after restore (grid stays ~150 pts) + multi-round settle at the
# target pressure (Tmax must sit BELOW the adiabatic-equilibrium ceiling).
set -u
cd "$(dirname "$0")"
source /home/sunkim/miniforge3/etc/profile.d/conda.sh
conda activate ct-env3

LOG=data/fix_150_200.log
echo "=== 150/200 atm continuation start ===" > "$LOG"

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
echo "=== 150/200 DONE ===" | tee -a "$LOG"
