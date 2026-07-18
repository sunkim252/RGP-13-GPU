/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::solvers::fgmFluid

Description
    Solver module for steady or transient turbulent flow of compressible
    multicomponent fluids with optional mesh motion and change.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient and steady simulations.

    Optional fvModels and fvConstraints are provided to enhance the simulation
    in many ways including adding various sources, chemical reactions,
    combustion, Lagrangian particles, radiation, surface film etc. and
    constraining or limiting the solution.

    Reference:
    \verbatim
        Greenshields, C. J., & Weller, H. G. (2022).
        Notes on Computational Fluid Dynamics: General Principles.
        CFD Direct Ltd.: Reading, UK.
    \endverbatim

SourceFiles
    fgmFluid.C

See also
    Foam::solvers::fluidSolver
    Foam::solvers::isothermalFluid

\*---------------------------------------------------------------------------*/

#ifndef fgmFluid_H
#define fgmFluid_H

#include "isothermalFluid.H"
#include "fluidMulticomponentThermo.H"
#include "fluidMulticomponentThermophysicalTransportModel.H"
#include "multivariateScheme.H"
#include "FGMTable.H"

#include <thread>
#include <chrono>
#include <cstdlib>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                          Class fgmFluid Declaration
\*---------------------------------------------------------------------------*/

class fgmFluid
:
    public isothermalFluid
{

protected:

    // Thermophysical properties

        fluidMulticomponentThermo& thermo_;


    // Composition

        PtrList<volScalarField>& Y_;


    // FGM lookup table and control variables

        //- FGM (FPV) lookup table: (Z~, gZ, C~) -> omega_C, T, Y_k
        FGMTable fgmTable_;

        //- Mixture fraction Z~ (transported, no source)
        volScalarField Z_;

        //- Progress variable C~ (transported, source = rho*omega_C)
        volScalarField C_;

        //- Mixture-fraction segregation factor gZ = Zvar/(Z~(1-Z~)) (derived)
        volScalarField gZ_;

        //- Scalar dissipation rate at Z = Z_st [1/s] (derived per cell from
        //  Pitsch-Steiner 2000: chi_st = chi_tilde * f(Z_st)/f(Z~), with
        //  chi_tilde = 2 D_eff |grad Z|^2 and the textbook polynomial
        //  shape f(Z) = Z(1-Z) used in place of exp(-2[erfc^-1(2Z)]^2)).
        volScalarField chi_st_;

        //- Cached volumetric PV source rho*omega_C [kg/m^3/s]
        volScalarField sourcePV_;

        //- Total (absolute) enthalpy h [J/kg], transported ONLY for the
        //  non-adiabatic FPV table (method b). It is a CONSERVED scalar (no
        //  reaction source: total enthalpy is invariant under reaction); the
        //  manifold is then queried at the enthalpy defect
        //  dh = h - ((1-Z) hOx + Z hFuel). Null for adiabatic (chi/3-D) tables.
        //  Crucially T is still READ FROM THE TABLE (T(Z,gZ,c,dh)) -- h is a
        //  coordinate, not inverted to T -- so the he<->T drift instability that
        //  forbids a transported energy equation does NOT apply here.
        autoPtr<volScalarField> hPtr_;

        //- Steam-dilution coordinate W [-], transported ONLY for a
        //  steam-diluted (4-D dilution) table. Conserved scalar (no reaction
        //  source: injected steam is inert as a stream tracer); the manifold's
        //  4th axis is queried directly at W = local steam-in-oxidiser
        //  fraction. Case supplies the field + BCs (steam inlet W=W_ox, O2 and
        //  fuel inlets W=0). Null for chi/enthalpy/3-D tables.
        //  v1 model: W is a single conserved mixing scalar; the mass/mole
        //  basis and the exclusion of combustion-produced H2O are documented
        //  approximations to be refined with a two-tracer reconstruction.
        autoPtr<volScalarField> WPtr_;

        //- LES mixture-fraction variance coefficient (Cv)
        scalar Cv_;

        //- Bilger stoichiometric mixture fraction (mass-based). Default
        //  0.0625 for LOX/Russian rocket kerosene; override in
        //  fgmProperties via 'Z_st'.
        scalar Z_st_;

        //- Per-variable Lewis numbers, keyed by field name ("Z", "C", and
        //  -- in the future differential-diffusion extension -- per species).
        //  Defaults to 1 (unity Lewis) for any name not listed.
        HashTable<scalar> Le_;

        //- Turbulent Schmidt number for the control-variable subgrid flux
        scalar Sct_;

        //- Progress-variable source scale (default 1). Diagnostic/tuning knob
        //  for the source-vs-diffusion balance of the burning branch: at low
        //  pressure the tabulated omega_C is ~p^2 weaker and the FGM burning
        //  branch can become unstable (Wang-2015 10 atm collapse). Set via
        //  'sourcePVscale' in fgmProperties.
        scalar sourcePVscale_;

        //- RUFPV-style limiting of the scalar-dissipation table coordinate
        //  (Gierth et al. 2022; Ihme & Pitsch). For a chi-axis (UFPV) table the
        //  per-cell chi_st = 2 D |grad Z|^2 f(Z_st)/f(Z~) is clamped to
        //  [chiClampMin_, chiClampMax_] BEFORE the source lookup. This breaks the
        //  destabilising instantaneous-|grad Z|^2 feedback (cooling -> grad Z
        //  relaxes -> chi drops -> tabulated source drops -> further cooling ->
        //  spurious cold collapse): the floor keeps the source on the burning
        //  branch; the cap (= chi_st,ign) excludes the extinguished branch.
        //  Defaults (0, GREAT) = no clamp -> legacy behaviour. Set via
        //  'chiClampMin'/'chiClampMax' in fgmProperties.
        scalar chiClampMin_;
        scalar chiClampMax_;

        //- Mixture-fraction variance closure regime, auto-selected from the
        //  momentum-transport "simulationType":
        //    les     -> mixing length L = Delta = V^(1/3)   (subgrid filter)
        //    ras     -> mixing length L = k^(3/2)/epsilon   (integral scale)
        //    laminar -> L = 0 (no unresolved variance; gZ -> 0)
        enum class varModel { laminar, ras, les };
        varModel varModel_;

        //- Indices of thermo species that are tabulated as Y_<sp>
        labelList tabSpecieIDs_;

        //- Tier-2: per-cell tabulated real-gas mixture coefficients (RG_*),
        //  filled from fgmTable_ each updateManifold and looked up by the
        //  SRKchungTaka mixture in place of the live O(n^2) pair mixing.
        //  Empty unless the table carries the RG_* fields, the mixture
        //  supports the hook, and 'tabulatedThermoCoeffs' is not disabled.
        PtrList<volScalarField> RGcoeffFields_;

        //- True when the tabulated-coefficient lookup is active.
        bool tabRealGasCoeffs_;

        //- Reusable per-cell scratch for the interpolated RG coefficients.
        List<scalar> RGcoeffBuf_;

        //- Mesh cell count for which the tabulated real-gas coefficient lookup
        //  was last armed. When it changes (a runtime redistribution or an AMR
        //  refinement move cells between processors / reallocate the coefficient
        //  and control-variable fields), the cached mixture field pointers --
        //  used to recover a cell index by address range -- are stale;
        //  updateManifold() then re-arms them via rearmRealGasCoeffTabulation()
        //  before any cell mixture is evaluated. -1 until the first successful
        //  arming (the guard only fires while tabRealGasCoeffs_ is true).
        label armedNCells_;

        //- Data pointer of Y_[0]'s internal field at the last arming. A parallel
        //  redistribution that KEEPS the local cell count reallocates the fields
        //  at a new address, which a bare nCells guard misses (stale pointers ->
        //  calcMixture reads freed storage). Re-arm whenever this pointer moves,
        //  not only when the count changes. nullptr until the first arming.
        const scalar* armedYData_;

        //- Tier-4: optional per-cell differential-diffusion Lewis-number fields
        //  for the control variables Z and C, filled from fgmTable_ each
        //  updateManifold. Null unless the table carries the corresponding
        //  Le_<var> field; when present, Deff(var) uses the local
        //  Le(Z,gZ,c[,chi]) in place of the constant Le_ scalar. This is the
        //  standard differential-diffusion FGM closure (Le baked into the
        //  manifold from the real-fluid flamelet transport).
        autoPtr<volScalarField> LeZField_;
        autoPtr<volScalarField> LeCField_;

        //- True when at least one tabulated Lewis-number field is active.
        bool tabLewis_;

        //- thermoGPU: offload the per-cell property refresh (the
        //  RhoFluidThermo::calculate() loop -- Cp/Cv/psi/rho/mu/kappa at the
        //  tabulated (p, T, Y)) to the CUDA kernels. Opt-in via
        //  'gpuThermo on;' in fgmProperties; requires the SRKchungTaka
        //  (Chung) mixture and a visible CUDA device.
        Switch gpuThermo_;

        //- True once the GPU tables have been uploaded (lazy, first call).
        bool gpuArmed_;

        //- pEqnGPU: correctPressurePEP의 압력 방정식을 fvMatrix 우회로
        //  GPU에서 조립+Jacobi-PCG 솔브+플럭스 추출. 'gpuPEqn on;'
        //  (fgmProperties). 'Gauss linear orthogonal' laplacian 전용.
        Switch gpuPEqn_;

        //- pEqnGPU 검증 모드: CPU fvMatrix와 계수/플럭스 대조 출력
        Switch gpuPEqnCheck_;

        //- pEqnGPU 솔버 선택: "pcg"(자체 Jacobi-PCG) | "amgx"(디바이스
        //  CSR을 libRGP13amgx의 rgpPEqnAmgxSolve로 — dlsym 연결이라
        //  controlDict libs에 libRGP13amgx.so 필요)
        word gpuPEqnSolver_;

        //- pcg 전처리기: "jacobi"(기본) | "dic"(멀티컬러 DIC —
        //  그리디 컬러링 색-레벨 병렬 스윕, 반복수 절감 vs 스윕 비용)
        word gpuPEqnPrecon_;

        //- pEqnGPU CSR nnz (amgx 경로, 아밍 시 설정)
        mutable label gpuPEqnNnz_;

        //- amgx 분산: 파티션 오프셋 [nProcs+1] (전역 셀 번호 경계)
        mutable List<long long> gpuPEqnPartOff_;

        //- ZCGPU: Z/C 수송 방정식(ddt+multivariate limitedLinear div+Sp+
        //  laplacian)을 GPU 조립+Jacobi-BiCGStab로. 'gpuZC on;'
        //  (fgmProperties). gpuPEqn과 동일한 v1 제약(비커플드/Euler/직교).
        Switch gpuZC_;
        mutable bool gpuZCArmed_;
        mutable List<double> gpuZCBuf_;   // 경계 연접 (bDiag,bSrc,bPsi)

        //- UEqnGPU: 운동량 방정식을 GPU 조립+성분별 솔브, 행렬 디바이스
        //  상주로 pCorr의 rAU/HbyA도 GPU 산출. 'gpuUEqn on;'
        //  (가중치·dev2 명시항 등 물리 필드는 호스트가 OF 스킴으로 계산)
        Switch gpuUEqn_;
        mutable List<double> gpuUBuf_;    // SoA/경계 스테이징

        //- balanced-force 압력구배: 셀 grad(p) 대신 면-재구성
        //  reconstruct(snGrad(p)*magSf)를 모멘텀 소스와 pCorr U-보정에
        //  사용 (VOF 계열 대밀도비 표준 — pEqn 라플라시안 플럭스와
        //  면-일관 이산화). snGrad는 케이스 스킴(limited corrected)
        //  그대로 호스트에서 계산 — GPU는 완성된 셀 필드만 소비
        //  (UEqn: gradP3 훅 / pCorr: rgpPCorrUHostGrad).
        //  'pGradRecon on;' (fgmProperties, 기본 off)
        Switch pGradRecon_;
        mutable List<double> gpuGradReconBuf_;   // [3nc] SoA 재사용 버퍼

        //- pGradRecon 디바이스 경로 상태 (0 미시도 / 1 활성 / -1 강등).
        //  첫 활성 호출에서 호스트 fvc 재계산과 비트 자가검증 —
        //  불일치·실패 시 영구 호스트 recon 강등 (기능 동일)
        mutable label gpuPGradReconState_;
        mutable bool gpuPGradReconGateDone_;

        //- 호스트 버퍼/필드 page-lock 등록 완료 플래그 (스텝 2에서
        //  1회 — 모든 grow-only 버퍼가 최종 크기에 도달한 뒤)
        mutable bool gpuPinned_;

        //- GPU 메시 구조의 아밍 시점 상태 (AMR/재분배 감지 → 재아밍)
        mutable label gpuMeshCells_;
        mutable label gpuMeshFaces_;

        //- topoChanged()/moving() 플래그 기반 재아밍의 스텝-단위 래치
        //  (플래그가 스텝 내내 유지되므로 timeIndex당 1회만 재아밍)
        mutable label gpuMeshStampTime_;

        //- pEqnGPU 메시 업로드 완료 플래그 + 작업 버퍼
        mutable bool gpuPEqnArmed_;

        //- devChain 강등 래치 (VRAM 부족 시 1회 경고 후 CPU-prep 폴백)
        mutable bool gpuDevChainOff_ = false;

        //- W6: devChain을 비직교에서도 활성 (fgmProperties, 기본 off)
        Switch gpuDevChainNO_;

        //- 비직교(corrected) snGrad 스킴 감지 (아밍 시 설정) —
        //  GPU laplacian 면 계수(nonOrthDeltaCoeffs)·명시 보정 소스·
        //  pEqn 비직교 보정자 루프 활성화
        mutable bool gpuNonOrtho_ = false;

        //- 비직교 안내 래치 (아밍마다 반복 출력 방지)
        mutable bool gpuNonOrthoWarned_ = false;

        //- 비직교 보정 소스 디바이스화 (아밍 시 자격 판정: corrected/
        //  limited-corrected snGrad + [cellMDLimited] Gauss linear
        //  grad(p) + 직렬) — 부적격/VRAM 부족이면 호스트 fvc 경로 유지.
        //  kf3/Cf3 SoA는 호스트 보존(패스별 스테이징 업로드)
        mutable bool gpuNOSrcDev_ = false;
        mutable scalar gpuNOLimitCoeff_ = -1;
        mutable List<double> gpuNOKf3_;
        mutable List<double> gpuNOCf3_;
        mutable List<double> gpuPEqnBuf_;   // 경계 연접 (bDiag,bSrc,phiB)
        mutable List<double> gpuPEqnFlux_;  // 내부면 플럭스 out
        mutable List<double> gpuPEqnParB_;  // processor 인터페이스 계수
        mutable List<double> gpuPEqnParPhb_;  // devChain proc-면 phiHbyAv
        mutable List<double> gpuPEqnParRhof_; // devChain proc-면 rhof

        //- fgmGPU: updateManifold의 셀 루프(대수 gZ/chi_st + 전 필드
        //  16-코너 보간)를 CUDA로 오프로드. 'gpuManifold on;' (fgmProperties).
        Switch gpuManifold_;

        //- 병렬에서 같은 GPU를 쓰는 랭크들이 매니폴드 테이블 디바이스
        //  할당을 CUDA IPC로 공유(리더만 업로드, 랭크당 ~250MB 중복 제거).
        //  IPC 미지원(WSL 등)이나 임포트 실패 시 랭크별 업로드로 폴백.
        //  'gpuManifoldShare off;' (fgmProperties, 기본 on)
        Switch gpuManifoldShare_;

        //- M1+M2 전송 다이어트: 산포에서 Y 슬라이스(테이블 종, 필드
        //  [2, 2+nY))의 전체 D2H·산포를 생략하고 경계-인접(소유) 셀만
        //  부분 게더로 갱신. 호스트 내부 Y의 값-소비자는 경계 BC 평가
        //  (소유 셀 값)와 비체인 폴백(syncHostTabY 가드)뿐이라 결과는
        //  비트-동일. copy 스테이징 모드에서만 활성(zero-copy 제외).
        //  'gpuManifoldYDiet off;' (fgmProperties, 기본 on)
        Switch gpuManifoldYDiet_;

        //- VRAM 다이어트: 디바이스 SoA에서 Y 블록을 아예 저장하지 않고
        //  (2.6M셀 106종 = 2.2GB) 체인이 테이블에서 재보간(동일 산술 =
        //  비트-동일). gpuManifoldYDiet과 copy 모드 전제.
        //  'gpuManifoldSoADiet off;' (fgmProperties, 기본 on)
        Switch gpuManifoldSoADiet_;
        mutable bool gpuYDietArmed_;
        mutable bool gpuHostYStale_;
        mutable labelList gpuYDietCells_;
        mutable List<double> gpuYDietBuf_;

        //- 경계 RG 채움(Opt-2)의 GPU 배치 상태: -1 미검증(첫 호출에서
        //  CPU 경로와 비트 자가검증), 1 GPU 확정, 0 CPU 폴백(검증 실패
        //  또는 커널 오류). 버퍼는 [Z|gZ|C|c4|out(13)]×nbf 연접.
        mutable int gpuBndCoeffMode_;
        mutable List<double> gpuBndBuf_;

        //- 비동기 오버랩: UEqn 성분 솔브(rgpUEqnSolveRun)를 워커
        //  스레드로 돌리고 메인은 tp pre/updateManifold(fgm·thermo
        //  버퍼만 사용 — pEqn 스크래치와 분리)를 진행. U 소비 직전
        //  (LAD/pCorr) joinUEqnSolve()로 합류해 산포·BC·K를 마무리.
        //  직렬 전용(병렬은 bicgSolve의 전역 리덕션이 MPI 경유).
        //  'gpuUEqnAsync off;' (fgmProperties, 기본 on)
        Switch gpuUEqnAsync_;
        mutable std::thread gpuUEqnThread_;
        mutable bool gpuUEqnPending_;
        mutable int gpuUEqnRc_;
        mutable int gpuUEqnCmpt_[3];
        mutable double gpuUEqnRes0_[3];
        mutable double gpuUEqnResF_[3];
        mutable int gpuUEqnIters_[3];
        mutable std::chrono::steady_clock::time_point gpuUEqnT0_;

        //- 테이블 업로드 완료 플래그 + 보간 출력 호스트 버퍼
        mutable bool gpuManifoldArmed_;

        //- he 재시드 경로: -1 미확정, 0 CPU 폴백(믹스처가 GPU thermo
        //  테이블을 제공하지 않음 — gpuManifold만 켠 경우), 1 GPU
        mutable int gpuHeMode_;
        mutable label gpuNFields_;
        mutable List<double> gpuFgmOut_;

        //- Print the wall time of the per-step thermo property refresh
        //  (thermo_.correct() or gpuThermoCorrect()). 'thermoTimings on;'
        //  in fgmProperties; benchmarking aid, default off.
        Switch thermoTimings_;

        //- Reusable host-side gather/scatter buffers for the GPU evaluate
        //  (internal cells + all patch faces, grow-only).
        List<double> gpuP_, gpuT_, gpuY_;
        List<double> gpuRho_, gpuMu_, gpuKappa_, gpuCp_, gpuCv_, gpuPsi_;

        //- Multivariate convection field table (he, Z, C)
        multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;


    // Thermophysical transport

        autoPtr<fluidMulticomponentThermophysicalTransportModel>
            thermophysicalTransport;


private:

    // Private Member Functions

        //- Set rDeltaT for LTS
        virtual void setRDeltaT();

        //- Pressure-equilibrium-preserving pressure equation (called by the
        //  overridden pressureCorrector). Replaces the base
        //  isothermalFluid::correctPressure so the transcritical density
        //  contact does not radiate a spurious pressure spike.
        void correctPressurePEP();

        //- Update gZ, look up the manifold, set inactive-species Yi and
        //  the cached PV source from the FGM table.
        //- hostScatter=false: 디바이스 상주 SoA(체인 소비)만 갱신하고
        //  호스트로의 fieldsOut D2H(~2.3GB@2.4M셀)와 산포를 생략 —
        //  압력 보정자 중간 단계 전용(그 사이 호스트 T/Y/srcc/Le/RG
        //  내부값 소비자가 없음; 마지막 outer는 항상 풀 산포)
        void updateManifold(const bool hostScatter = true);

        //- Y-다이어트 부분 산포: 경계-인접 셀의 Y만 디바이스 SoA에서
        //  압축 게더해 갱신 (updateManifold 산포 블록 전용). 성공 시
        //  gpuHostYStale_=true(내부 비경계 셀 stale), 실패 시 전체
        //  복원으로 폴백.
        void scatterTabYSubset(List<scalarField*>& Yref);

        //- 비동기 UEqn 합류: 워커 스레드를 join하고 잔차 출력·U 산포·
        //  BC 보정·K 갱신을 수행. 미발진 상태면 no-op (멱등).
        void joinUEqnSolve();

        //- pGradRecon 디바이스 경로: recon = reconstruct(snGrad(p)*magSf)
        //  를 결정론 CSR 게더로 디바이스 계산해 gU.gradP에 상주.
        //  updateU=true면 pcUUpdate(U=HbyA−rAU·recon)+U3out D2H까지.
        //  경계 ssf는 호스트(proc halo 포함), 첫 호출에 비트 자가검증.
        //  false 반환 = 미아밍/강등 — 호출자는 호스트 recon 경로 사용.
        bool deviceReconGrad(const bool updateU, double* U3out) const;

        //- 게더/산포 복사 루프의 OpenMP 스레드 수 (결과는 스레드 수와
        //  무관 — 반복 독립 복사 전용). 기본: 직렬 4, 병렬 1(클러스터
        //  과다구독 방지). env RGP_OMP_THREADS로 재정의 — 로컬 소수
        //  랭크(코어 여유) 런에서 랭크당 CPU 병렬화에 사용.
        static inline int ompThreads()
        {
            static int n = -1;
            if (n < 0)
            {
                const char* e = getenv("RGP_OMP_THREADS");
                n = e ? atoi(e) : (Pstream::parRun() ? 1 : 4);
                if (n < 1) { n = 1; }
            }
            return n;
        }

        //- 비체인 폴백 가드: 호스트 테이블-종 Y가 stale이면 디바이스
        //  SoA에서 전체 복원 (gpuThermoCorrect/gpuHeReseed의 비체인
        //  경로가 호스트 내부 Y를 게더하기 전 호출)
        void syncHostTabY();

        //- Tier-2 setup: if the table carries RG_* coefficients and the
        //  mixture supports the tabulatedRealGasMixture hook, allocate the 13
        //  per-cell coefficient fields and enable the mixture lookup. Must run
        //  before the first updateManifold()/thermo correct so the fields are
        //  filled before any cell mixture is evaluated through the lookup.
        void setupRealGasCoeffTabulation();

        //- Re-arm the tabulated real-gas coefficient lookup after a mesh-cell-
        //  count change: refresh the cached mixture field pointers (the internal
        //  coefficient fields, the Opt-1 base-blend stencil fields and the Opt-2
        //  patch fields) to the reallocated fields, WITHOUT rebuilding the mesh-
        //  independent manifold node mixtures. Called from updateManifold() when
        //  mesh.nCells() differs from armedNCells_ (runtime redistribution or AMR
        //  refinement). Bit-identical to the freshly armed state.
        void rearmRealGasCoeffTabulation();

        //- Per-cell tabulated Lewis-number field for a control variable, or
        //  nullptr when the table does not tabulate Le for it (constant Le_
        //  fallback). Tier-4 differential-diffusion hook.
        const volScalarField* leField(const word& var) const;

        //- thermoGPU: one-time arming -- export the mixture tables via the
        //  tabulatedRealGasMixture::gpuThermoTables hook and upload them to
        //  the CUDA device. FatalError when 'gpuThermo on;' cannot be
        //  honoured (unsupported mixture / no device), so a requested GPU
        //  run never silently falls back to the CPU.
        void armGpuThermo();

        //- thermoGPU: replacement for thermo_.correct() -- batch-evaluate
        //  Cp/Cv/psi/rho/mu/kappa at the tabulated (p, T, Y) on the GPU for
        //  the internal cells and all patch faces, and scatter into the
        //  thermo fields. T is NOT re-inverted from he (it is the manifold
        //  temperature set in updateManifold; the CPU inversion just returns
        //  it to Newton tolerance).
        void gpuThermoCorrect();

        //- fgmGPU: replacement for the updateManifold he re-seed
        //  thermo_.he() = thermo_.he(p, T) -- batch-evaluate the absolute
        //  enthalpy ha(p, T_table, Y_table) on the GPU (janaf blend + SRK
        //  mixture departure, same tables as armGpuThermo) for internal
        //  cells and all patch faces. Arms the thermo tables if needed.
        void gpuHeReseed();

        //- 대용량 전송에 쓰이는 상주 호스트 버퍼/필드를 page-lock 등록
        //  (pageable 전송 병목 완화). 버퍼 크기 확정 후 1회.
        void pinGpuHostBuffers();

        //- 랭크→CUDA 디바이스 매핑 후 rgpGpuInit (idempotent). 모든 GPU
        //  아밍(thermo 테이블·fgm 테이블)은 디바이스 할당 전에 이걸 거쳐야
        //  멀티-GPU 병렬에서 랭크별 디바이스에 올라간다.
        //  반환: 선택된 디바이스(>=0) | -1 디바이스 없음 | -2 init 실패
        int gpuSelectDevice() const;

        //- GPU 경로가 재현하지 않는 행렬-레벨 fvConstraint가 해당 필드에
        //  걸려 있으면 FatalError. 필드-레벨 전용 타입(bound/limitPressure/
        //  limitTemperature/limitMag)은 CPU 경로와 동일하게 솔브 후
        //  constrain(field)로 적용되므로 허용.
        void checkGpuConstraints
        (
            const word& fieldName,
            const char* gpuSwitch
        ) const;

        //- pEqnGPU: 정적 메시(owner/neigh/기하계수/V/경계 faceCells)
        //  업로드 (1회, idempotent). gpuPEqn/gpuZC 양쪽에서 사용.
        void armGpuPEqnMesh();

        //- 수송용 정적 메시(선형 가중치/Sf/d/경계 Sf) 업로드 (1회,
        //  idempotent). gpuZC/gpuUEqn 양쪽에서 사용.
        void armGpuSTMesh();

        //- Effective (dynamic) diffusivity [kg/m/s] for a transported control
        //  variable, using its Lewis number:
        //    Deff = alpha_laminar/Le + rho*nut/Sct
        //  Le is a per-cell tabulated Le(Z,gZ,c[,chi]) when the table provides
        //  Le_<var> (Tier-4 differential diffusion), else the constant Le_
        //  scalar (default 1). The single extension point for non-unity and
        //  per-species differential diffusion.
        tmp<volScalarField> Deff(const word& var);

        //- W4: Deff 공통부 캐시 — turb=rho·nut/Sct와 mu가 outer당
        //  최대 5회(updateManifold의 Z + predictor의 Z/C/h[/W]) 동일
        //  재계산되던 것을 thermophysicalPredictor 스코프에서 1회로.
        //  피연산자만 재사용하고 합산식(mu/Le + turb)은 기존 그대로라
        //  비트-동일. predictor 밖(다른 경로의 Deff 호출)은 비캐시.
        mutable tmp<volScalarField> DeffTurbC_;
        mutable tmp<volScalarField> DeffMuC_;

        //- Mixture-fraction variance mixing-length squared L^2 [m^2] per cell,
        //  auto-selected from the momentum-transport simulationType (LES filter
        //  width / RAS integral length / laminar zero). Used in the algebraic
        //  variance closure Zvar = Cv * L^2 * |grad Z~|^2.
        tmp<scalarField> varianceLengthSqr();


public:

    // Public Data

        //- Reference to the fluid thermophysical properties
        const fluidMulticomponentThermo& thermo;

        //- Reference to the composition
        const PtrList<volScalarField>& Y;


    //- Runtime type information
    TypeName("fgmFluid");


    // Constructors

        //- Construct from region mesh
        fgmFluid(fvMesh& mesh);

        //- Disallow default bitwise copy construction
        fgmFluid(const fgmFluid&) = delete;


    //- Destructor
    virtual ~fgmFluid();


    // Member Functions

        //- Predict thermophysical transport
        virtual void thermophysicalTransportPredictor();

        //- Construct and solve the energy equation,
        //  convert to temperature
        //  and update thermophysical and transport properties
        virtual void thermophysicalPredictor();

        //- Construct and solve the momentum equation, adding a localized
        //  artificial (shear) viscosity (LADUCoeff) to damp the spurious
        //  velocity overshoot at the transcritical density interface
        virtual void momentumPredictor();

        //- Construct and solve the pressure equation in the PISO loop using a
        //  pressure-equilibrium-preserving (pressure-evolution) formulation
        //  that removes the spurious pressure spike at the transcritical
        //  density contact (see pressureCorrector.C).
        virtual void pressureCorrector();

        //- Correct the thermophysical transport
        virtual void thermophysicalTransportCorrector();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const fgmFluid&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
