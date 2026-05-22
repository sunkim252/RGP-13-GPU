/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Class
    Foam::solvers::fgmFluid

Description
    Solver module for steady or transient turbulent flow of compressible
    multicomponent fluids with optional mesh motion and change.

    Uses the flexible PIMPLE (PISO-SIMPLE) solution for time-resolved and
    pseudo-transient and steady simulations.

    Optional fvModels and fvConstraints are provided to enhance the simulation
    in many ways including adding various sources, chemical reactions,
    combustion, Lagrangian particles, radiation, surface film etc. and
    constraining or limiting the solution.

    Reference:
    \verbatim
        Greenshields, C. J., & Weller, H. G. (2022).
        Notes on Computational Fluid Dynamics: General Principles.
        CFD Direct Ltd.: Reading, UK.
    \endverbatim

SourceFiles
    fgmFluid.C

See also
    Foam::solvers::fluidSolver
    Foam::solvers::isothermalFluid

\*---------------------------------------------------------------------------*/

#ifndef fgmFluid_H
#define fgmFluid_H

#include "isothermalFluid.H"
#include "fluidMulticomponentThermo.H"
#include "fluidMulticomponentThermophysicalTransportModel.H"
#include "multivariateScheme.H"
#include "FGMTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{

/*---------------------------------------------------------------------------*\
                          Class fgmFluid Declaration
\*---------------------------------------------------------------------------*/

class fgmFluid
:
    public isothermalFluid
{

protected:

    // Thermophysical properties

        fluidMulticomponentThermo& thermo_;


    // Composition

        PtrList<volScalarField>& Y_;


    // FGM lookup table and control variables

        //- FGM (FPV) lookup table: (Z~, gZ, C~) -> omega_C, T, Y_k
        FGMTable fgmTable_;

        //- Mixture fraction Z~ (transported, no source)
        volScalarField Z_;

        //- Progress variable C~ (transported, source = rho*omega_C)
        volScalarField C_;

        //- Mixture-fraction segregation factor gZ = Zvar/(Z~(1-Z~)) (derived)
        volScalarField gZ_;

        //- Cached volumetric PV source rho*omega_C [kg/m^3/s]
        volScalarField sourcePV_;

        //- LES mixture-fraction variance coefficient (Cv)
        scalar Cv_;

        //- Per-variable Lewis numbers, keyed by field name ("Z", "C", and
        //  -- in the future differential-diffusion extension -- per species).
        //  Defaults to 1 (unity Lewis) for any name not listed.
        HashTable<scalar> Le_;

        //- Turbulent Schmidt number for the control-variable subgrid flux
        scalar Sct_;

        //- Indices of thermo species that are tabulated as Y_<sp>
        labelList tabSpecieIDs_;

        //- Multivariate convection field table (he, Z, C)
        multivariateSurfaceInterpolationScheme<scalar>::fieldTable fields;


    // Thermophysical transport

        autoPtr<fluidMulticomponentThermophysicalTransportModel>
            thermophysicalTransport;


private:

    // Private Member Functions

        //- Set rDeltaT for LTS
        virtual void setRDeltaT();

        //- Update gZ, look up the manifold, set inactive-species Yi and
        //  the cached PV source from the FGM table.
        void updateManifold();

        //- Effective (dynamic) diffusivity [kg/m/s] for a transported control
        //  variable, using its Lewis number:
        //    Deff = alpha_laminar/Le + rho*nut/Sct
        //  Le defaults to 1. This is the single extension point for non-unity
        //  Lewis numbers and future per-species differential diffusion.
        tmp<volScalarField> Deff(const word& var);


public:

    // Public Data

        //- Reference to the fluid thermophysical properties
        const fluidMulticomponentThermo& thermo;

        //- Reference to the composition
        const PtrList<volScalarField>& Y;


    //- Runtime type information
    TypeName("fgmFluid");


    // Constructors

        //- Construct from region mesh
        fgmFluid(fvMesh& mesh);

        //- Disallow default bitwise copy construction
        fgmFluid(const fgmFluid&) = delete;


    //- Destructor
    virtual ~fgmFluid();


    // Member Functions

        //- Predict thermophysical transport
        virtual void thermophysicalTransportPredictor();

        //- Construct and solve the energy equation,
        //  convert to temperature
        //  and update thermophysical and transport properties
        virtual void thermophysicalPredictor();

        //- Correct the thermophysical transport
        virtual void thermophysicalTransportCorrector();


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const fgmFluid&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace solvers
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
