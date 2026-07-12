#!/bin/bash
#SBATCH -J rgp-smoke2
#SBATCH -p <beta-partition>
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 8
#SBATCH -t 00:40:00
#SBATCH -o smoke2.%j.out

# 스모크 #2: native(자동) vs 강제-copy 비트-정합 — GH200 코히런트 경로의
# 실기 확정 게이트. 최종 시간 디렉터리 전 필드 cmp로 비트-동일 판정.
set -eu
SIF=${SIF:-$PWD/openfoam13-rgp-gpu-arm64.sif}
CASE=${CASE:-$PWD/RGP-13-realFluid/testCases/h2CounterFlow_gpuChem}

run_one() {  # $1=디렉터리 $2=RGP_GPU_UNIFIED값("auto"면 미설정)
    local d=$1 mode=$2
    rm -rf "$d" && cp -a "$CASE" "$d"
    apptainer exec --nv "$SIF" bash -c "
      set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null; set -e
      [ '$mode' != auto ] && export RGP_GPU_UNIFIED=$mode
      cd '$d'
      foamDictionary -entry endTime -set 2e-5 system/controlDict
      blockMesh > log.blockMesh 2>&1 || true
      [ -f system/setFieldsDict ] && setFields > log.setFields 2>&1 || true
      foamRun > log.run 2>&1
    "
}

run_one "$SLURM_SUBMIT_DIR/smoke2-native" auto
run_one "$SLURM_SUBMIT_DIR/smoke2-copy"   0

grep -m1 "coherent HW" "$SLURM_SUBMIT_DIR/smoke2-native/log.run" || true

# 최종 시간 디렉터리 비교
tN=$(ls -d "$SLURM_SUBMIT_DIR"/smoke2-native/[0-9]* | sort -g | tail -1)
tC=$(ls -d "$SLURM_SUBMIT_DIR"/smoke2-copy/[0-9]*   | sort -g | tail -1)
echo "compare: $(basename "$tN") vs $(basename "$tC")"
fail=0
for f in "$tN"/*; do
    b=$(basename "$f"); [ -f "$f" ] || continue
    cmp -s "$f" "$tC/$b" || { echo "DIFF $b"; fail=1; }
done
[ $fail -eq 0 ] && echo "SMOKE2 PASS (native == copy, bit-identical)" \
                || echo "SMOKE2 FAIL — RGP_GPU_UNIFIED=0으로 운용하고 이슈 보고"
