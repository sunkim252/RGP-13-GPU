/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | RGP-13: Real Gas Properties for OpenFOAM 13
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2024-2026 RGP-13 Contributors
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    GPLv3 — see LICENSE file
-------------------------------------------------------------------------------
Class
    Foam::cudaDeviceManager

Description
    Singleton that maps MPI rank → CUDA device at construction time.
    Calls cudaSetDevice() once; all subsequent CUDA API calls on this
    rank use the selected device.

    Multi-GPU policy:  rank % nDevices  (round-robin)

\*---------------------------------------------------------------------------*/

#ifndef RGP13_cudaDeviceManager_H
#define RGP13_cudaDeviceManager_H

#include "label.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

class cudaDeviceManager
{
    // Private Data

        //- Selected device ordinal
        int deviceId_;

        //- Total visible devices on this node
        int nDevices_;

    // Private Member Functions

        //- Disallow copy/move
        cudaDeviceManager(const cudaDeviceManager&) = delete;
        void operator=(const cudaDeviceManager&) = delete;

        //- Singleton instance
        static cudaDeviceManager& instance();

        //- Constructor — performs the actual cudaSetDevice
        cudaDeviceManager();

public:

    //- Initialise (idempotent). Call once, typically in thermo constructor.
    static void init();

    //- Currently selected device id
    static int deviceId();

    //- Number of visible CUDA devices
    static int nDevices();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// ************************************************************************* //
