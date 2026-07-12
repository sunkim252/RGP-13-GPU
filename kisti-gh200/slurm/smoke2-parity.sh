#!/bin/bash
#SBATCH -J rgp-smoke2
#SBATCH -p <beta-partition>
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 8
#SBATCH -t 00:40:00
#SBATCH -o smoke2.%j.out

# 스모크 #2: native(자동) vs 강제-copy 정합 — GH200 코히런트 경로 실기 확정.
#
# 판정 설계 (x86에서 확립): GPU atomics 리덕션 필드(U/phi 등)는 같은
# 모드끼리도 run-to-run 비트-비동일(노이즈 클래스)이다. 따라서
#   ① copy 모드 2회(A,B) → A==B인 필드 = "결정론 필드 집합"
#   ② native 1회 → 결정론 필드 전부가 copyA와 비트-동일이면 PASS
# dt는 고정(adjustTimeStep off)해 노이즈가 dt 시퀀스로 전파되는 것을 차단.
set -eu
SIF=${SIF:-$PWD/openfoam13-rgp-gpu-arm64.sif}
CASE=${CASE:-$PWD/RGP-13-realFluid/testCases/h2CounterFlow_gpuChem}
SUB=${SLURM_SUBMIT_DIR:-$PWD}

run_one() {  # $1=디렉터리 $2=RGP_GPU_UNIFIED("auto"=미설정) $3=chemonly?
    local d=$1 mode=$2 chemonly=${3:-no}
    rm -rf "$d" && cp -a "$CASE" "$d"
    apptainer exec --nv "$SIF" bash -c "
      set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null; set -e
      [ '$mode' != auto ] && export RGP_GPU_UNIFIED=$mode
      cd '$d'
      [ -d 0 ] || cp -a 0.orig 0
      foamDictionary -entry endTime -set 2e-5 system/controlDict
      foamDictionary -entry adjustTimeStep -set no system/controlDict
      foamDictionary -entry writeControl -set runTime system/controlDict
      foamDictionary -entry writeInterval -set 2e-5 system/controlDict
      if [ '$chemonly' = yes ]; then
          foamDictionary -entry gpuUEqn -set off constant/gpuProperties
          foamDictionary -entry gpuPEqn -set off constant/gpuProperties
      fi
      blockMesh > log.blockMesh 2>&1 || true
      [ -f system/setFieldsDict ] && setFields > log.setFields 2>&1 || true
      foamRun > log.run 2>&1
    "
}

# ── Phase A: 화학-only GPU — 완전 결정론이므로 전 필드 비트 판정 (강) ──
run_one "$SUB/smoke2A-copy"   0    yes
run_one "$SUB/smoke2A-native" auto yes
ttA=$(ls -d "$SUB"/smoke2A-copy/[0-9]* | grep -v "/0$" | sort -g | tail -1)
ttA=$(basename "${ttA:-}")
[ -z "$ttA" ] && { echo "SMOKE2A FAIL: no time dirs"; exit 1; }
fa=0
for f in "$SUB/smoke2A-copy/$ttA"/*; do
    b=$(basename "$f"); [ -f "$f" ] || continue
    cmp -s "$f" "$SUB/smoke2A-native/$ttA/$b"         || { echo "SMOKE2A REAL DIFF: $b"; fa=1; }
done
[ $fa -eq 0 ] && echo "SMOKE2A PASS (chem-only: native == copy, ALL fields bit)"               || { echo "SMOKE2A FAIL"; exit 1; }

# ── Phase B: 풀-GPU — 결정론 필드 부분집합 판정 (보조) ──
run_one "$SUB/smoke2-copyA"  0
run_one "$SUB/smoke2-copyB"  0
run_one "$SUB/smoke2-native" auto

grep -m1 "coherent HW" "$SUB/smoke2-native/log.run" || true

tt=$(ls -d "$SUB"/smoke2-copyA/[0-9]* | grep -v "/0$" | sort -g | tail -1)
tt=$(basename "${tt:-}")
[ -z "$tt" ] && { echo "SMOKE2 FAIL: no time dirs written"; exit 1; }
echo "comparing at t=$tt"

fail=0; ndet=0
for f in "$SUB/smoke2-copyA/$tt"/*; do
    b=$(basename "$f"); [ -f "$f" ] || continue
    # ① 결정론 필드만 판정 대상 (copyA==copyB)
    cmp -s "$f" "$SUB/smoke2-copyB/$tt/$b" || { echo "  (noise-class: $b)"; continue; }
    ndet=$((ndet+1))
    # ② 결정론 필드는 native와 비트-동일해야 함
    cmp -s "$f" "$SUB/smoke2-native/$tt/$b" \
        || { echo "REAL DIFF (deterministic field): $b"; fail=1; }
done
echo "deterministic fields compared: $ndet"
[ "$ndet" -eq 0 ] && { echo "SMOKE2 FAIL: no deterministic fields (unexpected)"; exit 1; }
[ $fail -eq 0 ] && echo "SMOKE2 PASS (native == copy on all deterministic fields)" \
                || { echo "SMOKE2 FAIL — RGP_GPU_UNIFIED=0으로 운용하고 이슈 보고"; exit 1; }
