/*---------------------------------------------------------------------------*\
Class
    Foam::amgxSolver

Description
    NVIDIA AmgX 기반 lduMatrix 솔버 ("amgx"). fvSolution에서

        p { solver amgx; tolerance 1e-7; relTol 0.01; maxIter 200; }

    로 선택. LDU→CSR 변환 후 AmgX(PCG + classical AMG)로 GPU 해석.
    압력계(대칭) 오프로드가 1차 목적 — fgmFluid 프로파일의 52%.

    v1 제약: serial 전용(processor/cyclic 등 lduInterface 불가 —
    감지 시 FatalError), 대칭/비대칭 모두 지원, 메시 구조 불변 가정
    (구조 변경 시 자동 재빌드).

SourceFiles
    amgxSolver.C
\*---------------------------------------------------------------------------*/

#ifndef amgxSolver_H
#define amgxSolver_H

#include "lduMatrix.H"

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class amgxSolver Declaration
\*---------------------------------------------------------------------------*/

class amgxSolver
:
    public lduMatrix::solver
{
    // Private data

        //- AmgX JSON 설정 (fvSolution 'amgxJson' 항목으로 교체 가능)
        string configJson_;


    // Private Member Functions

        //- 필드별 AmgX 핸들 캐시 접근 (구조 불변 시 계수만 교체)
        void* cacheFor(const label nCells, const label nnz) const;


public:

    //- Runtime type information
    TypeName("amgx");


    // Constructors

        amgxSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~amgxSolver()
    {}


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// ************************************************************************* //
