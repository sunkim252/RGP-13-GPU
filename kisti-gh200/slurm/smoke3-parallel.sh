#!/bin/bash
#SBATCH -J rgp-smoke3
#SBATCH -p <beta-partition>
#SBATCH --gres=gpu:4                 # 1장뿐이면 gpu:1로 — 랭크들이 같은 GPU 공유
#SBATCH -N 1 -n 4 -c 4
#SBATCH -t 01:00:00
#SBATCH -o smoke3.%j.out

# 스모크 #3: 4랭크 병렬 (pcg + 화학 DLB on/off 비트-파리티, 선택 AmgX)
set -eu
SIF=${SIF:-$PWD/openfoam13-rgp-gpu-arm64.sif}
CASE=${CASE:-$PWD/RGP-13-realFluid/testCases/h2CounterFlow_gpuChem}

prep() {  # $1=디렉터리 $2=dlb(on/off)
    local d=$1 dlb=$2
    rm -rf "$d" && cp -a "$CASE" "$d"
    apptainer exec "$SIF" bash -c "
      set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null; set -e
      cd '$d'
      foamDictionary -entry endTime -set 2e-5 system/controlDict
  foamDictionary -entry writeControl -set timeStep system/controlDict
      foamDictionary -entry writeInterval -set 20 system/controlDict
      blockMesh > log.blockMesh 2>&1 || true
      [ -f system/setFieldsDict ] && setFields > log.setFields 2>&1 || true
      foamDictionary -entry gpuCoeffs/dlb -set $dlb \
          constant/chemistryProperties
      decomposePar -force > log.decomp 2>&1
    "
}

runp() {
    local d=$1
    apptainer exec --nv "$SIF" bash -c "
      set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null; set -e
      cd '$d'
      mpirun -np 4 foamRun -parallel > log.par 2>&1
    "
}

prep "${SLURM_SUBMIT_DIR:-$PWD}/smoke3-dlb1" on  && runp "${SLURM_SUBMIT_DIR:-$PWD}/smoke3-dlb1"
prep "${SLURM_SUBMIT_DIR:-$PWD}/smoke3-dlb0" off && runp "${SLURM_SUBMIT_DIR:-$PWD}/smoke3-dlb0"

echo '=== 랭크→GPU 매핑/DLB 텔레메트리 ==='
grep -m2 -E "GPU|DLB" "${SLURM_SUBMIT_DIR:-$PWD}/smoke3-dlb1/log.par" || true

# DLB on/off 최종장 비트-파리티 (랭크별)
fail=0
for r in 0 1 2 3; do
    tD=$(ls -d "${SLURM_SUBMIT_DIR:-$PWD}"/smoke3-dlb1/processor$r/[0-9]* | sort -g | tail -1)
    case "$(basename "$tD")" in 0) echo "SMOKE3 FAIL: no time dirs"; fail=1; break;; esac
    tO="${SLURM_SUBMIT_DIR:-$PWD}/smoke3-dlb0/processor$r/$(basename "$tD")"
    for f in "$tD"/*; do
        b=$(basename "$f"); [ -f "$f" ] || continue
        cmp -s "$f" "$tO/$b" || { echo "DIFF p$r/$b"; fail=1; }
    done
done
[ $fail -eq 0 ] && echo "SMOKE3 PASS (DLB on/off bit-identical, 4 ranks)" \
                || echo "SMOKE3 FAIL"

# (선택) AmgX 분산 동작 확인: 케이스 dict에 gpuPEqnSolver amgx; 넣고 재실행
