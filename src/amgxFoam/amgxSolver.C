/*---------------------------------------------------------------------------*\
  amgxSolver — NVIDIA AmgX lduMatrix 솔버 구현.

  구조: 필드명 키의 정적 캐시에 AmgX 핸들(resources/config/solver/
  matrix/vectors)과 CSR 골격을 보관. 메시 구조가 같으면 계수만
  replace_coefficients + solver_setup(AMG 재구성)으로 갱신.
\*---------------------------------------------------------------------------*/

#include "amgxSolver.H"
#include "addToRunTimeSelectionTable.H"
#include "PstreamReduceOps.H"

#include <amgx_c.h>

#include <map>
#include <string>
#include <vector>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(amgxSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<amgxSolver>
        addamgxSolverSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<amgxSolver>
        addamgxSolverAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * AmgX 전역 상태/캐시 * * * * * * * * * * * * * * //

namespace
{

#define AMGX_CHK(call)                                                         \
    {                                                                          \
        AMGX_RC rc_ = (call);                                                  \
        if (rc_ != AMGX_RC_OK)                                                 \
        {                                                                      \
            char msg_[512];                                                    \
            AMGX_get_error_string(rc_, msg_, 512);                             \
            FatalErrorInFunction                                               \
                << "AmgX error: " << msg_ << Foam::exit(Foam::FatalError);     \
        }                                                                      \
    }

    struct AmgxField
    {
        AMGX_config_handle cfg = nullptr;
        AMGX_resources_handle rsrc = nullptr;
        AMGX_solver_handle solver = nullptr;
        AMGX_matrix_handle A = nullptr;
        AMGX_vector_handle x = nullptr, b = nullptr;
        Foam::label nCells = -1, nnz = -1;
        Foam::label solveCount = 0;
        std::vector<int> rowPtr, colInd;
        std::vector<double> vals;
        std::string json;
    };

    bool amgxInitialised = false;
    std::map<std::string, AmgxField> amgxCache;

    void amgxGlobalInit()
    {
        if (!amgxInitialised)
        {
            AMGX_CHK(AMGX_initialize());
            amgxInitialised = true;
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::amgxSolver::amgxSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    configJson_
    (
        solverControls.lookupOrDefault<string>
        (
            "amgxJson",
            string
            (
                "{\"config_version\":2,\"solver\":{"
                "\"solver\":\"PCG\",\"max_iters\":4,\"tolerance\":1e-30,"
                "\"convergence\":\"ABSOLUTE\",\"norm\":\"L1\","
                "\"monitor_residual\":1,"
                "\"preconditioner\":{"
                "\"solver\":\"AMG\",\"algorithm\":\"CLASSICAL\","
                "\"interpolator\":\"D2\",\"aggressive_levels\":2,"
                "\"max_levels\":24,\"cycle\":\"V\","
                "\"matrix_coloring_scheme\":\"MIN_MAX\","
                "\"smoother\":{\"solver\":\"MULTICOLOR_DILU\","
                "\"monitor_residual\":0},"
                "\"presweeps\":1,\"postsweeps\":1,"
                "\"max_iters\":1"
                "}}}"
            )
        )
    )
{
    if (Pstream::parRun())
    {
        FatalErrorInFunction
            << "amgx solver v1 is serial-only" << exit(FatalError);
    }
    forAll(interfaces, i)
    {
        if (interfaces.set(i))
        {
            FatalErrorInFunction
                << "amgx solver v1 does not support coupled interfaces "
                << "(cyclic/processor)" << exit(FatalError);
        }
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::amgxSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf(typeName, fieldName_);

    const label n = matrix_.diag().size();

    // ── 초기 잔차 (OF 관례 정규화) ────────────────────────────────────
    scalarField wA(n), pA(n);
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    scalarField rA(source - wA);
    const scalar normFactor = this->normFactor(psi, source, wA, pA);
    solverPerf.initialResidual() = gSumMag(rA)/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if
    (
        this->minIter_ == 0
     && solverPerf.checkConvergence(this->tolerance_, this->relTol_)
    )
    {
        return solverPerf;
    }

    // ── CSR 조립 ─────────────────────────────────────────────────────
    amgxGlobalInit();
    AmgxField& F = amgxCache[std::string(fieldName_.c_str())];

    const scalarField& diag = matrix_.diag();
    const labelUList& lAddr = matrix_.lduAddr().lowerAddr();
    const labelUList& uAddr = matrix_.lduAddr().upperAddr();
    const scalarField& upper = matrix_.upper();
    const scalarField& lower =
        matrix_.hasLower() ? matrix_.lower() : matrix_.upper();
    const label nFaces = uAddr.size();
    const label nnz = n + 2*nFaces;

    const bool rebuild = (F.nCells != n || F.nnz != nnz);
    if (rebuild)
    {
        // 구조 (재)생성: row 당 [lower cols(<row)] [diag] [upper cols(>row)]
        F.rowPtr.assign(n + 1, 0);
        for (label f = 0; f < nFaces; f++)
        {
            F.rowPtr[uAddr[f] + 1]++;    // row=neighbour, col=owner (<row)
            F.rowPtr[lAddr[f] + 1]++;    // row=owner, col=neighbour (>row)
        }
        for (label r = 0; r < n; r++) { F.rowPtr[r + 1] += F.rowPtr[r] + 1; }

        F.colInd.assign(nnz, -1);
        F.vals.assign(nnz, 0.0);
    }

    // 계수 채우기 (lduAddr는 row 내에서 정렬돼 있음: lower는 owner 오름차순,
    // upper 접근은 losort 없이 직접 — 정렬은 아래 삽입 순서로 보장)
    {
        std::vector<int> fill(n, 0);
        // pass1: lower part (col < row) — face f: row uAddr[f], col lAddr[f]
        for (label f = 0; f < nFaces; f++)
        {
            const label r = uAddr[f];
            const label k = F.rowPtr[r] + fill[r]++;
            F.colInd[k] = lAddr[f];
            F.vals[k] = lower[f];
        }
        // diag
        for (label r = 0; r < n; r++)
        {
            const label k = F.rowPtr[r] + fill[r]++;
            F.colInd[k] = r;
            F.vals[k] = diag[r];
        }
        // upper part (col > row) — face f: row lAddr[f], col uAddr[f]
        for (label f = 0; f < nFaces; f++)
        {
            const label r = lAddr[f];
            const label k = F.rowPtr[r] + fill[r]++;
            F.colInd[k] = uAddr[f];
            F.vals[k] = upper[f];
        }
    }

    // ── AmgX 핸들 준비 ───────────────────────────────────────────────
    if (rebuild)
    {
        if (F.solver)
        {
            AMGX_solver_destroy(F.solver);
            AMGX_matrix_destroy(F.A);
            AMGX_vector_destroy(F.x);
            AMGX_vector_destroy(F.b);
            AMGX_resources_destroy(F.rsrc);
            AMGX_config_destroy(F.cfg);
            F.solver = nullptr;
        }

        // tolerance는 OF 정규화 잔차 기준 → AmgX에는 절대 톨로 전달
        F.json = std::string(configJson_.c_str());
        AMGX_CHK(AMGX_config_create(&F.cfg, F.json.c_str()));
        AMGX_CHK(AMGX_resources_create_simple(&F.rsrc, F.cfg));
        AMGX_CHK(AMGX_matrix_create(&F.A, F.rsrc, AMGX_mode_dDDI));
        AMGX_CHK(AMGX_vector_create(&F.x, F.rsrc, AMGX_mode_dDDI));
        AMGX_CHK(AMGX_vector_create(&F.b, F.rsrc, AMGX_mode_dDDI));
        AMGX_CHK(AMGX_solver_create(&F.solver, F.rsrc, AMGX_mode_dDDI, F.cfg));

        AMGX_CHK(AMGX_matrix_upload_all
        (
            F.A, n, nnz, 1, 1,
            F.rowPtr.data(), F.colInd.data(), F.vals.data(), nullptr
        ));
        F.nCells = n;
        F.nnz = nnz;
    }
    else
    {
        AMGX_CHK(AMGX_matrix_replace_coefficients
        (
            F.A, n, nnz, F.vals.data(), nullptr
        ));
    }

    // AMG 계층 재사용: 구조 재생성/주기적 리프레시 때만 full setup,
    // 그 외에는 resetup(기존 계층에 새 계수만 반영) — setup 비용 제거.
    const label setupInterval =
        controlDict_.lookupOrDefault<label>("setupInterval", 25);
    if (rebuild)
    {
        AMGX_CHK(AMGX_solver_setup(F.solver, F.A));
    }
    else if (setupInterval > 0 && (F.solveCount % setupInterval) == 0)
    {
        AMGX_CHK(AMGX_solver_resetup(F.solver, F.A));
    }
    // 그 외: AMG 계층 동결 — 전처리기만 미세하게 stale, PCG는 새 계수의
    // A로 수렴 판정하므로 정확성 무손실 (flexible preconditioning).
    F.solveCount++;

    AMGX_CHK(AMGX_vector_upload(F.x, n, 1, psi.begin()));
    AMGX_CHK(AMGX_vector_upload(F.b, n, 1, const_cast<scalar*>(source.begin())));

    // ── 고정 4-iter 블록 + 외부(OF 규약) 수렴 판정 ──────────────────
    // AMGX_config_add_parameters의 per-solve tolerance 주입은 라이브
    // 솔버에 반영되지 않는다(rgpPEqnAmgx.C에서 확인된 잠복 이슈) —
    // 내부 톨을 1e-30으로 잠그고 블록 사이에서 OF 정규화 잔차로
    // 판정한다. x는 블록 간 연속이라 정확히 이어서 수렴한다.
    int iters = 0;
    const label maxRounds = max(label(1), (this->maxIter_ + 3)/4);
    for (label round = 0; round < maxRounds; round++)
    {
        AMGX_CHK(AMGX_solver_solve(F.solver, F.b, F.x));

        int it = 0;
        AMGX_solver_get_iterations_number(F.solver, &it);
        iters += it;

        AMGX_CHK(AMGX_vector_download(F.x, psi.begin()));
        matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        const scalar resF = gSumMag(source - wA)/normFactor;
        solverPerf.finalResidual() = resF;

        // NaN은 모든 비교에서 false → 수렴 위장 방지, 명시 탈출
        if (resF != resF)
        {
            WarningInFunction
                << "amgx: diverged (NaN residual) for " << fieldName_
                << " -- frozen AMG hierarchy may be stale; reduce "
                << "setupInterval" << endl;
            break;
        }
        if (solverPerf.checkConvergence(this->tolerance_, this->relTol_))
        {
            break;
        }
    }
    solverPerf.nIterations() = iters;

    return solverPerf;
}


// ************************************************************************* //
