/*---------------------------------------------------------------------------*\
Class
    Foam::solvers::gpuMulticomponentFluid

Description
    GPU-accelerated variant of the stock multicomponentFluid solver module
    for general finite-rate combustion (laminar / EDC / PaSR TCI models).

    Reuses the RGP-13 foam-free CUDA transport layer (librgpThermoGPU):
    the multivariate limitedLinear weights (all Y + he, CPU convention),
    every solved YiEqn (ddt + mvConvection + divj == R) and optionally the
    EEqn are assembled and solved on the device (Jacobi-BiCGStab, OF
    residual conventions). Chemistry acceleration is orthogonal: select
    'chemistryType { solver gpu; }' (libRGP13gpuChemistry) as usual.

    Opt-in via constant/gpuProperties:
        gpuYEqn  on;    // Y species batch on GPU
        gpuEEqn  on;    // energy equation on GPU (requires gpuYEqn)
        gpuCheck off;   // coefficient cross-check vs CPU fvMatrix

    Diffusion: unityLewis-type transport uses the fast device path
    (cell alphaEff, face-interpolated on the GPU); any other model
    (Fickian/FickianFourier/FickianEddyDiffusivity -- non-unity Lewis,
    per-specie DEff, Soret) is handled generically by extracting the
    face/boundary coefficients and explicit sources from the model's
    own divj/divq matrices and injecting them into the device assembly.

    Supported physics: he == h or e (pressureWork e-form via device
    flux-div with the shared multivariate weights), MRF (DDt stored
    source + makeRelative/constrainPressure), buoyancy (p_rgh
    formulation 1:1 with correctBuoyantPressure incl. netForce), AMR
    topo changes and load-balancing redistribution (automatic
    re-arming). Serial and MPI-parallel (processor coupling).

    v1 constraints (loud Fatal, no silent divergence): Euler/LTS ddt,
    no fvModels sources / fvConstraints / relaxation on the GPU-solved
    fields, no transonic (fvm::div(phid,p) makes the p-matrix
    asymmetric -- needs upwind assembly + BiCGStab pEqn) and no
    'consistent', static mesh (continuous mesh motion lacks
    meshPhi/V0 physics on the device).

SourceFiles
    gpuMulticomponentFluid.C
    thermophysicalPredictor.C

\*---------------------------------------------------------------------------*/

#ifndef gpuMulticomponentFluid_H
#define gpuMulticomponentFluid_H

#include "multicomponentFluid.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                   Class gpuMulticomponentFluid Declaration
\*---------------------------------------------------------------------------*/

class gpuMulticomponentFluid
:
    public multicomponentFluid
{
    // Private data

        //- GPU 스위치 (constant/gpuProperties, READ_IF_PRESENT)
        Switch gpuYEqn_;
        Switch gpuEEqn_;
        Switch gpuUEqn_;
        Switch gpuPEqn_;
        Switch gpuCheck_;

        //- Fickian류(비-unity Lewis) 확산: 모델이 조립한 divj/divq
        //  행렬에서 면 계수·경계 계수·명시 소스를 추출해 GPU 조립에
        //  주입 (unityLewis*는 셀-gamma 고속 경로)
        bool gpuDiffExtract_;

        //- 메시 아밍 상태 (토폴로지/이동 감지 재아밍 — fgmFluid와 동일)
        mutable bool gpuMeshArmed_;
        mutable label gpuMeshCells_;
        mutable label gpuMeshFaces_;
        mutable label gpuMeshStampTime_;

        //- 경계 연접 작업 버퍼 (bDiag, bSrc, bPsi)
        mutable List<double> gpuBuf_;

        //- UEqn SoA/경계 스테이징 (fgmFluid 레이아웃과 동일)
        mutable List<double> gpuUBuf_;

        //- pEqn 경계 연접 + 내부면 플럭스
        mutable List<double> gpuPBuf_;
        mutable List<double> gpuPFlux_;

        //- processor 인터페이스 계수 (방정식별 솔브 직전 업로드)
        mutable List<double> gpuParB_;

        //- proc-면 multivariate limited 가중치 (weights 단계에서 채움,
        //  solveScalarGpu의 coupled 계수에 사용)
        mutable List<double> gpuProcW_;

        //- 비직교 스킴 (corrected/limited snGrad) — 아밍 시 감지.
        //  GPU 라플라시안: 면 계수 = 스킴 deltaCoeffs(nonOrthDeltaCoeffs)
        //  + 호스트 명시 보정 소스 + pEqn 보정자 루프 (fgmFluid 1:1)
        mutable bool gpuNonOrtho_;
        mutable bool gpuNonOrthoWarned_;


    // Private Member Functions

        //- 정적 메시(pEqn 구조 + 수송 가중치/Sf/d) 업로드 (idempotent)
        void armGpuMesh();

        //- 스칼라 수송 GPU 솔브: ddt(rho,psi) + div(phi,psi;w_limited)
        //  − laplacian(gamma,psi) + sp*psi == src (per-volume sp/src)
        void solveScalarGpu
        (
            volScalarField& psiF,
            const volScalarField& gamma,
            const scalarField& sp,
            const scalarField& src,
            const word& dictBase,
            fvScalarMatrix* diffOp = nullptr
        );

        //- v1 가드: relax/fvModels/fvConstraints가 GPU-솔브 필드에
        //  작용하면 Fatal (침묵 오차 방지)
        void checkGpuGuards(const volScalarField& f) const;

        //- PISO 보정자 1회분의 GPU 압력 보정 (pressureCorrector의
        //  while(pimple.correct()) 루프 본체)
        void correctPressureGpu();


public:

    //- Runtime type information
    TypeName("gpuMulticomponentFluid");


    // Constructors

        //- Construct from region mesh
        gpuMulticomponentFluid(fvMesh& mesh);

        //- Disallow default bitwise copy construction
        gpuMulticomponentFluid(const gpuMulticomponentFluid&) = delete;


    //- Destructor
    virtual ~gpuMulticomponentFluid();


    // Member Functions

        //- Construct and solve the Y/energy equations on the GPU,
        //  convert to temperature and update properties
        virtual void thermophysicalPredictor();

        //- GPU momentum predictor (stock UEqn, limitedLinearV +
        //  linear-viscous-stress divDevTau — fgmFluid gpuUEqn 이식)
        virtual void momentumPredictor();

        //- GPU pressure corrector (stock non-transonic correctPressure
        //  형태를 rgpPEqnSolve로 — psi/rhorAUf/phiHbyA 매핑)
        virtual void pressureCorrector();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const gpuMulticomponentFluid&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
