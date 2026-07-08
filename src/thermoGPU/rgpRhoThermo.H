/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | RGP-13: Real Gas Properties for OpenFOAM 13
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2024-2026 RGP-13 Contributors
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    GPLv3
-------------------------------------------------------------------------------
Class
    Foam::rgpRhoThermo

Description
    rhoFluidThermo 래퍼. calculate()만 GPU 커널로 대체한다.
    기존 OF13 thermophysical 프레임워크의 RTS에 등록되어
    physicalProperties 딕셔너리에서 `type rgpRhoThermo`로 선택 가능.

    경계면(boundary) 처리는 CPU fallback (Phase 1).

SourceFiles
    rgpRhoThermo.C

\*---------------------------------------------------------------------------*/

#ifndef rgpRhoThermo_H
#define rgpRhoThermo_H

#include "rhoFluidThermo.H"
#include "gpu/thermoKernels.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                       Class rgpRhoThermo Declaration
\*---------------------------------------------------------------------------*/

class rgpRhoThermo
:
    public rhoFluidThermo::composite
{
    // Private Data

        //- GPU thermo coefficients (read from physicalProperties)
        RGP13::ThermoCoeffs coeffs_;

        //- Minimum cell count to offload to GPU (below this → CPU fallback)
        label cudaMinCells_;


    // Private Member Functions

        //- Read coefficients from the thermo sub-dictionary
        void readCoeffs(const dictionary& dict);

        //- CPU fallback for boundary faces
        void calculateBoundary();

        //- Disallow copy
        rgpRhoThermo(const rgpRhoThermo&) = delete;
        void operator=(const rgpRhoThermo&) = delete;


public:

    //- Runtime type information
    TypeName("rgpRhoThermo");


    // Constructors

        //- Construct from mesh and phase name
        rgpRhoThermo(const fvMesh& mesh, const word& phaseName);


    //- Destructor
    virtual ~rgpRhoThermo();


    // Member Functions

        //- Update thermophysical properties (GPU-accelerated)
        virtual void correct();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

#endif

// ************************************************************************* //
