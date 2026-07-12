#!/bin/bash
#SBATCH -J rgp-smoke1
#SBATCH -p <beta-partition>          # KISTI 6호기 베타 파티션명으로 교체
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 8
#SBATCH -t 00:20:00
#SBATCH -o smoke1.%j.out

# 스모크 #1: 기동 + coherent 자동감지 확인 (GH200 1장)
set -eu
SIF=${SIF:-$PWD/openfoam13-rgp-gpu-arm64.sif}
CASE=${CASE:-$PWD/RGP-13-realFluid/testCases/h2CounterFlow_gpuChem}
RUN=$SLURM_SUBMIT_DIR/smoke1-case
rm -rf "$RUN" && cp -a "$CASE" "$RUN"

apptainer exec --nv "$SIF" bash -c "
  set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null; set -e
  cd '$RUN'
  foamDictionary -entry endTime -set 2e-5 system/controlDict
  blockMesh > log.blockMesh 2>&1 || true
  [ -f system/setFieldsDict ] && setFields > log.setFields 2>&1 || true
  foamRun > log.smoke1 2>&1
"

echo '=== coherent 감지 배너 ==='
grep -m2 -E "ARMED|coherent HW|unified" "$RUN/log.smoke1" || echo "배너 없음 — GPU 미기동?"
echo '=== 종료 상태 ==='
tail -3 "$RUN/log.smoke1"
grep -q "^End$" "$RUN/log.smoke1" && echo "SMOKE1 PASS" || echo "SMOKE1 FAIL"
