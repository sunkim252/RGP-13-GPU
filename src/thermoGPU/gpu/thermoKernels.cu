/*---------------------------------------------------------------------------*\
  RGP-13 — Host-side wrapper for GPU thermo kernels

  Unified Memory + cudaMemPrefetchAsync로 PCIe 병목 회피.
  OpenFOAM의 scalarField 포인터를 받아 GPU에서 물성을 계산한 뒤 반환.
\*---------------------------------------------------------------------------*/

#include "thermoKernels.cuh"
#include "thermoKernels.H"
#include "cudaDeviceManager.H"

#include <cuda_runtime.h>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define RGP_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess)                                               \
        {                                                                     \
            FatalErrorInFunction                                              \
                << "CUDA error: " << cudaGetErrorString(err)                  \
                << " [" << __FILE__ << ":" << __LINE__ << "]"                 \
                << Foam::abort(FatalError);                                   \
        }                                                                     \
    } while (0)


// ─── Persistent Unified Memory buffers ─────────────────────────────
// Allocated once per process, resized only if nCells changes.

namespace
{
    struct UMBuffers
    {
        double* he    = nullptr;
        double* p     = nullptr;
        double* T     = nullptr;
        double* Cp    = nullptr;
        double* Cv    = nullptr;
        double* psi   = nullptr;
        double* rho   = nullptr;
        double* mu    = nullptr;
        double* kappa = nullptr;
        int     size  = 0;

        void resize(int n)
        {
            if (n <= size) return;
            free();

            cudaMallocManaged(&he,    n * sizeof(double));
            cudaMallocManaged(&p,     n * sizeof(double));
            cudaMallocManaged(&T,     n * sizeof(double));
            cudaMallocManaged(&Cp,    n * sizeof(double));
            cudaMallocManaged(&Cv,    n * sizeof(double));
            cudaMallocManaged(&psi,   n * sizeof(double));
            cudaMallocManaged(&rho,   n * sizeof(double));
            cudaMallocManaged(&mu,    n * sizeof(double));
            cudaMallocManaged(&kappa, n * sizeof(double));
            size = n;
        }

        void free()
        {
            if (!size) return;
            cudaFree(he);    cudaFree(p);     cudaFree(T);
            cudaFree(Cp);    cudaFree(Cv);    cudaFree(psi);
            cudaFree(rho);   cudaFree(mu);    cudaFree(kappa);
            size = 0;
        }

        ~UMBuffers() { free(); }
    };

    static UMBuffers& buffers()
    {
        static UMBuffers b;
        return b;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::rgp13ThermoCalculateGPU
(
    // Host scalarField data pointers
    const double* hostHe,
    const double* hostP,
    double*       hostT,
    double*       hostCp,
    double*       hostCv,
    double*       hostPsi,
    double*       hostRho,
    double*       hostMu,
    double*       hostKappa,
    const int     nCells,
    const RGP13::ThermoCoeffs& coeffs
)
{
    if (nCells <= 0) return;

    const int deviceId = cudaDeviceManager::deviceId();
    UMBuffers& buf = buffers();
    buf.resize(nCells);

    const size_t bytes = nCells * sizeof(double);

    // ── Copy host → UM buffers ─────────────────────────────────────
    memcpy(buf.he, hostHe, bytes);
    memcpy(buf.p,  hostP,  bytes);
    memcpy(buf.T,  hostT,  bytes);

    // ── Prefetch to GPU ────────────────────────────────────────────
    cudaStream_t stream;
    RGP_CUDA_CHECK(cudaStreamCreate(&stream));

    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.he, bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.p,  bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.T,  bytes, deviceId, stream));
    // Output buffers — prefetch to GPU so kernel writes are device-local
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.Cp,    bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.Cv,    bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.psi,   bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.rho,   bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.mu,    bytes, deviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.kappa, bytes, deviceId, stream));

    // ── Launch kernel ──────────────────────────────────────────────
    constexpr int blockSize = 256;
    const int gridSize = (nCells + blockSize - 1) / blockSize;

    RGP13::thermoCalculateKernel<<<gridSize, blockSize, 0, stream>>>
    (
        buf.he, buf.p,
        buf.T,
        buf.Cp, buf.Cv, buf.psi, buf.rho,
        buf.mu, buf.kappa,
        coeffs,
        nCells
    );
    RGP_CUDA_CHECK(cudaGetLastError());

    // ── Prefetch results back to CPU ───────────────────────────────
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.T,     bytes, cudaCpuDeviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.Cp,    bytes, cudaCpuDeviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.Cv,    bytes, cudaCpuDeviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.psi,   bytes, cudaCpuDeviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.rho,   bytes, cudaCpuDeviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.mu,    bytes, cudaCpuDeviceId, stream));
    RGP_CUDA_CHECK(cudaMemPrefetchAsync(buf.kappa, bytes, cudaCpuDeviceId, stream));

    RGP_CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Copy UM buffers → host fields ──────────────────────────────
    memcpy(hostT,     buf.T,     bytes);
    memcpy(hostCp,    buf.Cp,    bytes);
    memcpy(hostCv,    buf.Cv,    bytes);
    memcpy(hostPsi,   buf.psi,   bytes);
    memcpy(hostRho,   buf.rho,   bytes);
    memcpy(hostMu,    buf.mu,    bytes);
    memcpy(hostKappa, buf.kappa, bytes);

    RGP_CUDA_CHECK(cudaStreamDestroy(stream));
}


// ************************************************************************* //
