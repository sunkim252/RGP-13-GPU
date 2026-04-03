/*---------------------------------------------------------------------------*\
  RGP-13 — Host-callable interface for GPU thermo kernels

  이 헤더는 .C (g++) 코드에서 include된다. CUDA 헤더를 노출하지 않는다.
\*---------------------------------------------------------------------------*/

#ifndef RGP13_thermoKernels_H
#define RGP13_thermoKernels_H

// Forward-declare the coefficients struct without CUDA dependency
// (matching layout in thermoKernels.cuh)
namespace RGP13
{
    struct JanafCoeffs;
    struct PRCoeffs;
    struct SutherlandCoeffs;
    struct ThermoCoeffs;
}

#include "thermoKernels.cuh"   // Full definition for ThermoCoeffs

namespace Foam
{

//- Calculate thermophysical properties on GPU for all local cells.
//  Arrays are host pointers; internal UM staging handles device transfer.
void rgp13ThermoCalculateGPU
(
    const double* hostHe,       // [in]  sensible enthalpy
    const double* hostP,        // [in]  pressure
    double*       hostT,        // [in/out] temperature (initial guess → result)
    double*       hostCp,       // [out] heat capacity at constant pressure
    double*       hostCv,       // [out] heat capacity at constant volume
    double*       hostPsi,      // [out] compressibility
    double*       hostRho,      // [out] density
    double*       hostMu,       // [out] dynamic viscosity
    double*       hostKappa,    // [out] thermal conductivity
    const int     nCells,       // local cell count (MPI-decomposed)
    const RGP13::ThermoCoeffs& coeffs
);

} // End namespace Foam

#endif

// ************************************************************************* //
