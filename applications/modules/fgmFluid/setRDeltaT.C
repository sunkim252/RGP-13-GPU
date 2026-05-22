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

\*---------------------------------------------------------------------------*/

#include "fgmFluid.H"
#include "fvcSmooth.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::solvers::fgmFluid::setRDeltaT()
{
    volScalarField& rDeltaT = trDeltaT.ref();

    const dictionary& pimpleDict = pimple.dict();

    Info<< "Time scales min/max:" << endl;

    // Cache old reciprocal time scale field
    const volScalarField rDeltaT0("rDeltaT0", rDeltaT);

    // Flow time scale
    {
        // Maximum flow Courant number
        const scalar maxCo(pimpleDict.lookup<scalar>("maxCo"));

        // Set the reciprocal time-step from the local Courant number
        rDeltaT.internalFieldRef() =
            fvc::surfaceSum(mag(phi))/((2*maxCo)*mesh.V()*rho());

        // Clip to user-defined maximum and minimum time-steps
        scalar minRDeltaT = gMin(rDeltaT.primitiveField());
        if (pimpleDict.found("maxDeltaT") || minRDeltaT < rootVSmall)
        {
            const scalar clipRDeltaT = 1/pimpleDict.lookup<scalar>("maxDeltaT");
            rDeltaT.max(clipRDeltaT);
            minRDeltaT = max(minRDeltaT, clipRDeltaT);
        }
        if (pimpleDict.found("minDeltaT"))
        {
            const scalar clipRDeltaT = 1/pimpleDict.lookup<scalar>("minDeltaT");
            rDeltaT.min(clipRDeltaT);
            minRDeltaT = min(minRDeltaT, clipRDeltaT);
        }

        Info<< "Flow time scale min/max = "
            << gMin(1/rDeltaT.primitiveField()) << ", " << 1/minRDeltaT << endl;
    }

    // Heat-release-rate time scale: omitted for the FGM solver (no explicit
    // Qdot; heat release is implicit in the tabulated composition / enthalpy).

    // Progress-variable source time scale (FGM analogue of the reaction-rate
    // constraint). Optional, enabled by the 'alphaC' PIMPLE entry.
    if (pimpleDict.found("alphaC"))
    {
        const scalar alphaC(pimpleDict.lookup<scalar>("alphaC"));
        const scalar Cref(pimpleDict.lookupOrDefault<scalar>("Cref", 0.1));

        volScalarField::Internal rDeltaTC
        (
            mag(sourcePV_()/(rho()*max(alphaC*Cref, small)))
        );

        Info<< "    Progress variable = "
            << 1/(gMax(rDeltaTC.primitiveField()) + vSmall) << ", "
            << 1/(gMin(rDeltaTC.primitiveField()) + vSmall) << endl;

        rDeltaT.internalFieldRef() = max(rDeltaT(), rDeltaTC);
    }

    // Update the boundary values of the reciprocal time-step
    rDeltaT.correctBoundaryConditions();

    // Smoothing parameter (0-1) when smoothing iterations > 0
    const scalar rDeltaTSmoothingCoeff
    (
        pimpleDict.lookupOrDefault<scalar>("rDeltaTSmoothingCoeff", 0.1)
    );

    // Spatially smooth the time scale field
    if (rDeltaTSmoothingCoeff < 1)
    {
        fvc::smooth(rDeltaT, rDeltaTSmoothingCoeff);
    }

    // Limit rate of change of time scale
    // - reduce as much as required
    // - only increase at a fraction of old time scale
    if
    (
        pimpleDict.found("rDeltaTDampingCoeff")
     && runTime.timeIndex() > runTime.startTimeIndex() + 1
    )
    {
        // Damping coefficient (1-0)
        const scalar rDeltaTDampingCoeff
        (
            pimpleDict.lookup<scalar>("rDeltaTDampingCoeff")
        );

        rDeltaT = max
        (
            rDeltaT,
            (scalar(1) - rDeltaTDampingCoeff)*rDeltaT0
        );
    }

    // Update the boundary values of the reciprocal time-step
    rDeltaT.correctBoundaryConditions();

    Info<< "    Overall     = "
        << 1/gMax(rDeltaT.primitiveField())
        << ", " << 1/gMin(rDeltaT.primitiveField()) << endl;
}


// ************************************************************************* //
