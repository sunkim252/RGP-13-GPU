#!/usr/bin/env bash
# Rebuild all Wang-2015 pressure tables with the burning-branch source closure
# (_laminar_branch in 04_build_fgm_table_4d.py). Only 10 atm needs the cool-
# flamelet exclusion; 1/25/50/100/150/200 atm are clean 28-flamelet families.
cd "$(dirname "$0")"
source /home/sunkim/miniforge3/etc/profile.d/conda.sh && conda activate ct-env3
declare -A EXCL=( [10]="5,6,7,8,13,38" )
for P in 1 10 25 50 100 150 200; do
  dir=data/flamelets_dualgas_P${P}atm
  if [ ! -d "$dir" ]; then echo "SKIP ${P}atm (no $dir)"; continue; fi
  ex=""
  [ -n "${EXCL[$P]:-}" ] && ex="--exclude-idx ${EXCL[$P]}"
  echo "===== building ${P}atm (excl='${EXCL[$P]:-none}') ====="
  python3 -u 04_build_fgm_table_4d.py --flamelet-dir "$dir" $ex \
    --out data/fgmProperties_${P}atm 2>&1 \
    | grep -aE 'load\]|excluded|omega_C: range|T: range|Error|Traceback'
done
echo "BUILDALLDONE"
