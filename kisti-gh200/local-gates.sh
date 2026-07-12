#!/bin/bash
# RGP-13-GPU 로컬(x86 discrete) 게이트 배터리 — 코드 변경 후 실행.
# 사용: SCRATCH=<벤치 모음 디렉터리> RGPLIB=<오버레이 libbin> ./local-gates.sh
#
# 구성 (2026-07-13 야간 세션에서 확립):
#  G1 bench_fgmMD   : fgm 풀스택(왜곡 메시, gpuThermo on) — T 전 시점 비트
#  G2 bench_gmcMD   : gmc 모듈(U/P GPU) — T 전 시점 비트
#  G3 bench_chemEDCg: GPU 화학 직렬 — 전 필드 비트
#  G4 chemBig_dlb0/1: 4랭크 화학 DLB on/off — 최종장 전 필드·전 랭크 비트
#  G5 rd0110        : SRK 106종 리얼가스 2.6M — 기동+2스텝 생존 (⚠️ 소형
#                     게이트가 못 잡는 리얼가스 CPU-mixture 경로 커버;
#                     RG D2H 다이어트 sigFpe를 잡았던 케이스)
#  판정 원칙: 결정론 필드(T 등)는 비트-동일 필수. U/phi 등 atomics
#  리덕션 필드는 run-to-run 노이즈 클래스 — same-mode 재실행(runA/runB)
#  으로 3-way 분류해 "runA==runB && !=ref"만 실변경으로 본다.
set -u
S=${SCRATCH:?}; L=${RGPLIB:?}; R=${RGPROOT:-$HOME/openfoam/RGP-13-GPU}
SIF=$R/openfoam13-rgp-gpu.sif
run() { apptainer exec --no-home --nv --bind /usr/lib/wsl --bind "$1":/case \
  --bind "$L":/rgpout "$SIF" bash -c '
  set +e; . /opt/OpenFOAM/OpenFOAM-13/etc/bashrc 2>/dev/null
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/rgpout:$LD_LIBRARY_PATH
  cd /case; '"$2"' > log.gates 2>&1' 2>/dev/null; }

fail=0
gate_T() { # $1=case $2..=times : ref<t>/T 대비 T 비트
  local c=$1; shift
  for t in "$@"; do rm -rf "$S/$c/$t"; done
  run "$S/$c" foamRun
  for t in "$@"; do
    cmp -s "$S/$c/ref$t/T" "$S/$c/$t/T" \
      && echo "PASS $c T@$t" || { echo "FAIL $c T@$t"; fail=1; }
  done
}

gate_T bench_fgmMD 1e-07 2e-07 3e-07
gate_T bench_gmcMD 1e-06 2e-06

# G3: 전 필드 비트
rm -rf "$S/bench_chemEDCg/1e-05"
run "$S/bench_chemEDCg" foamRun
d=0; for f in "$S"/bench_chemEDCg/ref1e-05/*; do b=$(basename "$f")
  [ -f "$f" ] && ! cmp -s "$f" "$S/bench_chemEDCg/1e-05/$b" && d=1; done
[ $d -eq 0 ] && echo "PASS chemEDCg all-fields" || { echo "FAIL chemEDCg"; fail=1; }

# G4: 4랭크 DLB 파리티
for c in chemBig_dlb0 chemBig_dlb1; do
  rm -rf "$S/$c"/processor*/2.1e-05
  run "$S/$c" "mpirun -np 4 --allow-run-as-root foamRun -parallel"
done
d=0; for r in 0 1 2 3; do
  for f in "$S"/chemBig_dlb0/processor$r/2.1e-05/*; do b=$(basename "$f")
    [ -f "$f" ] && ! cmp -s "$f" "$S/chemBig_dlb1/processor$r/2.1e-05/$b" && d=1
  done
done
[ $d -eq 0 ] && echo "PASS DLB-parity 4rank" || { echo "FAIL DLB-parity"; fail=1; }

# G5: rd0110 리얼가스 생존 (비트 아님 — 기동·물성·2스텝 완주)
rm -rf "$S/rd0110/1" "$S/rd0110/2"
run "$S/rd0110" foamRun
grep -q "^End$" "$S/rd0110/log.gates" \
  && echo "PASS rd0110 realgas-survive" \
  || { echo "FAIL rd0110 (리얼가스 경로 — 소형 게이트 밖 회귀!)"; fail=1; }

[ $fail -eq 0 ] && echo "=== ALL GATES PASS ===" || echo "=== GATE FAILURE ==="
exit $fail
