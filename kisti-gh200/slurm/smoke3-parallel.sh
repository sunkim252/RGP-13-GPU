#!/bin/bash
#SBATCH -J rgp-smoke3
#SBATCH -p <beta-partition>
#SBATCH --gres=gpu:4                 # 1장뿐이면 gpu:1로 — 랭크들이 같은 GPU 공유
#SBATCH -N 1 -n 4 -c 4
#SBATCH -t 01:00:00
#SBATCH -o smoke3.%j.out

# 스모크 #3: 4랭크 병렬 — 화학-only GPU(결정론 구성; x86 로컬에서
# dlb on/off·mode0/mode1 4랭크 비트-동일 실증)에서
#   ① DLB on vs off 비트-파리티
#   ② native(auto) vs 강제-copy 비트-파리티  ← GH200 병렬 코히런트 확정
# 풀-GPU(UEqn/pEqn on)는 atomics 노이즈로 비트 비교가 원리적으로 불가 —
# 그 경로의 native 검증은 smoke2 Phase B(결정론 집합)가 담당.
set -eu
SIF=${SIF:-$PWD/openfoam13-rgp-gpu-arm64.sif}
CASE=${CASE:-$PWD/RGP-13-realFluid/testCases/h2CounterFlow_gpuChem}
SUB=${SLURM_SUBMIT_DIR:-$PWD}

prep_run() {  # $1=디렉터리 $2=dlb(on/off) $3=RGP_GPU_UNIFIED("auto"=미설정)
    local d=$1 dlb=$2 mode=$3
    rm -rf "$d" && cp -a "$CASE" "$d"
    apptainer exec --nv "$SIF" bash -c "
      set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null; set -e
      [ '$mode' != auto ] && export RGP_GPU_UNIFIED=$mode
      cd '$d'
      foamDictionary -entry endTime -set 2e-5 system/controlDict
      foamDictionary -entry adjustTimeStep -set no system/controlDict
      foamDictionary -entry writeControl -set runTime system/controlDict
      foamDictionary -entry writeInterval -set 2e-5 system/controlDict
      foamDictionary -entry gpuUEqn -set off constant/gpuProperties
      foamDictionary -entry gpuPEqn -set off constant/gpuProperties
      foamDictionary -entry gpuCoeffs/dlb -set $dlb \
          constant/chemistryProperties
      blockMesh > log.blockMesh 2>&1 || true
      [ -f system/setFieldsDict ] && setFields > log.setFields 2>&1 || true
      decomposePar -force > log.decomp 2>&1
      mpirun -np 4 foamRun -parallel > log.par 2>&1
    "
}

prep_run "$SUB/smoke3-dlb1"   on  0
prep_run "$SUB/smoke3-dlb0"   off 0
prep_run "$SUB/smoke3-native" on  auto

echo '=== 랭크→GPU 매핑/DLB 텔레메트리 ==='
grep -m2 -E "GPU|DLB|coherent" "$SUB/smoke3-dlb1/log.par" || true
grep -m1 "coherent" "$SUB/smoke3-native/log.par" || true

tt=$(ls -d "$SUB"/smoke3-dlb1/processor0/[0-9]* | grep -v "/0$" | sort -g | tail -1)
tt=$(basename "${tt:-}")
[ -z "$tt" ] && { echo "SMOKE3 FAIL: no time dirs"; exit 1; }

fail=0
cmp_pair() { # $1=A $2=B $3=tag
    local A=$1 B=$2 tag=$3 d=0 r b f
    for r in 0 1 2 3; do
        for f in "$A/processor$r/$tt"/*; do
            b=$(basename "$f"); [ -f "$f" ] || continue
            cmp -s "$f" "$B/processor$r/$tt/$b" \
                || { echo "DIFF $tag p$r/$b"; d=1; }
        done
    done
    if [ $d -eq 0 ]; then echo "PASS $tag (bit-identical, 4 ranks)"
    else echo "FAIL $tag"; fail=1; fi
}

cmp_pair "$SUB/smoke3-dlb1" "$SUB/smoke3-dlb0"   "DLB-on-vs-off"
cmp_pair "$SUB/smoke3-dlb1" "$SUB/smoke3-native" "copy-vs-native"
[ $fail -eq 0 ] && echo "SMOKE3 PASS" || { echo "SMOKE3 FAIL"; exit 1; }

# (선택) AmgX 분산 동작 확인: 케이스 dict에 gpuPEqnSolver amgx; 넣고 재실행
