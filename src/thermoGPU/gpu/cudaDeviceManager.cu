/*---------------------------------------------------------------------------*\
  RGP-13 — cudaDeviceManager implementation
\*---------------------------------------------------------------------------*/

#include "cudaDeviceManager.H"
#include "Pstream.H"
#include "error.H"

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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::cudaDeviceManager::cudaDeviceManager()
:
    deviceId_(0),
    nDevices_(0)
{
    RGP_CUDA_CHECK(cudaGetDeviceCount(&nDevices_));

    if (nDevices_ == 0)
    {
        FatalErrorInFunction
            << "No CUDA-capable devices found."
            << Foam::abort(FatalError);
    }

    // Round-robin: rank % nDevices
    const int rank = Foam::Pstream::myProcNo();
    deviceId_ = rank % nDevices_;

    RGP_CUDA_CHECK(cudaSetDevice(deviceId_));

    // Log device info
    cudaDeviceProp prop;
    RGP_CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId_));

    Foam::Info
        << "RGP-13 CUDA | rank " << rank
        << " → device " << deviceId_
        << " (" << prop.name << ")"
        << ", " << (prop.totalGlobalMem >> 20) << " MiB"
        << ", SM " << prop.major << "." << prop.minor
        << Foam::endl;
}


Foam::cudaDeviceManager& Foam::cudaDeviceManager::instance()
{
    static cudaDeviceManager mgr;
    return mgr;
}


void Foam::cudaDeviceManager::init()
{
    (void)instance();
}


int Foam::cudaDeviceManager::deviceId()
{
    return instance().deviceId_;
}


int Foam::cudaDeviceManager::nDevices()
{
    return instance().nDevices_;
}


// ************************************************************************* //
