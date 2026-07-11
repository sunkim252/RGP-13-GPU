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

    v1 constraints (loud Fatal/fallback, no silent divergence):
    serial-only (auto-disable + warning in parallel), Euler/LTS ddt,
    'Gauss linear orthogonal' laplacian, unityLewis- or Fourier-type
    thermophysical transport (divj/divq are pure laplacians), he == h,
    no fvModels sources / matrix-level fvConstraints / relaxation on
    the GPU-solved fields, no buoyancy with gpuEEqn.

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
            const word& dictBase
        );

        //- v1 가드: relax/fvModels/fvConstraints가 GPU-솔브 필드에
        //  작용하면 Fatal (침묵 오차 방지)
        void checkGpuGuards(const volScalarField& f) const;


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
