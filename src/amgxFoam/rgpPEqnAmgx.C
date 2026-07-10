/*---------------------------------------------------------------------------*\
  rgpPEqnAmgx — pEqnGPU 디바이스 CSR에 AmgX 직결 (풀-GPU ③)

  thermoGPU(rgpPEqnKernels.cu)가 조립한 디바이스 CSR 값/벡터를 호스트
  왕복 없이 AmgX PCG+AMG로 솔브한다. 구조(rowPtr/colInd 호스트)는 첫
  호출에서 1회 바인딩, 이후 AMGX_matrix_replace_coefficients(디바이스
  값 포인터)만. AMG 계층은 setupInterval번의 압력 솔브마다만 resetup
  (동결 전략, amgxSolver.C와 동일; newSolve 플래그가 솔브 경계 표시).
  수렴 판정은 호출자가 4-iter 블록 사이에서 OF 규약 잔차로 수행.

  fgmFluid는 이 함수를 dlsym으로 찾는다(하드 링크 없음) — controlDict
  libs에 libRGP13amgx.so가 있어야 활성화된다.
\*---------------------------------------------------------------------------*/

#include <amgx_c.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace
{
    char gErr[512] = "no error";

    struct State
    {
        bool globalInit = false;
        AMGX_config_handle cfg = nullptr;
        AMGX_resources_handle rsrc = nullptr;
        AMGX_solver_handle solver = nullptr;
        AMGX_matrix_handle A = nullptr;
        AMGX_vector_handle x = nullptr, b = nullptr;
        int nCells = -1, nnz = -1;
        long solveCount = 0;
    } gS;

    const char* kJson =
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
        "}}}";

#define CHK(call, where)                                                       \
    {                                                                          \
        AMGX_RC rc_ = (call);                                                  \
        if (rc_ != AMGX_RC_OK)                                                 \
        {                                                                      \
            char msg_[400];                                                    \
            AMGX_get_error_string(rc_, msg_, 400);                             \
            snprintf(gErr, sizeof(gErr), "%s: %s", where, msg_);               \
            return -1;                                                         \
        }                                                                      \
    }
}


extern "C"
{

const char* rgpPEqnAmgxErr(void) { return gErr; }


int rgpPEqnAmgxSolve
(
    int nCells, int nnz,
    const int* rowPtrHost, const int* colIndHost,
    void* dValues, void* dB, void* dX,
    int newSolve, int setupInterval, int* nIters
)
{
    if (!gS.globalInit)
    {
        CHK(AMGX_initialize(), "initialize");
        gS.globalInit = true;
    }

    const bool rebuild = (gS.nCells != nCells || gS.nnz != nnz);
    if (rebuild)
    {
        // 실패-후-재시도가 반쯤 파괴된 핸들로 non-rebuild 분기를 타지
        // 않도록 캐시 키를 먼저 무효화 (성공 시 setup 후 재설정)
        gS.nCells = -1;
        gS.nnz = -1;

        if (gS.solver)
        {
            AMGX_solver_destroy(gS.solver);
            AMGX_matrix_destroy(gS.A);
            AMGX_vector_destroy(gS.x);
            AMGX_vector_destroy(gS.b);
            AMGX_resources_destroy(gS.rsrc);
            AMGX_config_destroy(gS.cfg);
            gS.solver = nullptr;
            gS.A = nullptr;
            gS.x = nullptr;
            gS.b = nullptr;
            gS.rsrc = nullptr;
            gS.cfg = nullptr;
        }

        CHK(AMGX_config_create(&gS.cfg, kJson), "config_create");
        CHK(AMGX_resources_create_simple(&gS.rsrc, gS.cfg),
            "resources_create");
        CHK(AMGX_matrix_create(&gS.A, gS.rsrc, AMGX_mode_dDDI),
            "matrix_create");
        CHK(AMGX_vector_create(&gS.x, gS.rsrc, AMGX_mode_dDDI),
            "vector_create x");
        CHK(AMGX_vector_create(&gS.b, gS.rsrc, AMGX_mode_dDDI),
            "vector_create b");
        CHK(AMGX_solver_create(&gS.solver, gS.rsrc, AMGX_mode_dDDI, gS.cfg),
            "solver_create");

        // 구조 바인딩: 값은 디바이스 포인터 그대로 (AmgX는 UVA로 판별)
        CHK(AMGX_matrix_upload_all
            (
                gS.A, nCells, nnz, 1, 1,
                rowPtrHost, colIndHost, dValues, nullptr
            ), "matrix_upload_all");
    }
    else
    {
        CHK(AMGX_matrix_replace_coefficients
            (
                gS.A, nCells, nnz, dValues, nullptr
            ), "replace_coefficients");
    }

    // AMG 계층 동결: 구조 (재)생성 시 full setup, 이후 setupInterval번의
    // '압력 솔브'마다 resetup. 한 압력 솔브가 4-iter 블록을 여러 번
    // 호출하므로 카운트는 newSolve(솔브의 첫 블록)에서만 올린다 —
    // 블록 단위로 세면 resetup이 의도의 ~수 배로 잦아지고 같은 계수의
    // 솔브 중간에 낭비 resetup이 낀다.
    if (rebuild)
    {
        CHK(AMGX_solver_setup(gS.solver, gS.A), "solver_setup");
        gS.nCells = nCells;
        gS.nnz = nnz;
        gS.solveCount = 1;
    }
    else if (newSolve)
    {
        if (setupInterval > 0 && (gS.solveCount % setupInterval) == 0)
        {
            CHK(AMGX_solver_resetup(gS.solver, gS.A), "solver_resetup");
        }
        gS.solveCount++;
    }

    CHK(AMGX_vector_upload(gS.x, nCells, 1, dX), "vector_upload x");
    CHK(AMGX_vector_upload(gS.b, nCells, 1, dB), "vector_upload b");

    CHK(AMGX_solver_solve(gS.solver, gS.b, gS.x), "solver_solve");

    CHK(AMGX_vector_download(gS.x, dX), "vector_download x");

    int it = 0;
    AMGX_solver_get_iterations_number(gS.solver, &it);
    *nIters = it;

    return 0;
}

} // extern "C"

// ************************************************************************* //
