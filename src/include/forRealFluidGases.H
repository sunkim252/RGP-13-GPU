/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2020-2023 OpenFOAM Foundation
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

#ifndef forRealFluidGases_H
#define forRealFluidGases_H

#include "rfSpecie.H"
#include "SRKGas.H"
#include "janafThermo.H"

#include "sensibleEnthalpy.H"
#include "sensibleInternalEnergy.H"
#include "absoluteEnthalpy.H"
#include "absoluteInternalEnergy.H"

#include "chungTransport.H"
#include "thermo.H"

#include "forThermo.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#define forRealFluidGasEqns(Mu, He, Cp, Macro, Args...)                       \
    forThermo(Mu, He, Cp, SRKGas, rfSpecie, Macro, Args)

#define forRealFluidGasEnergiesAndThermos(Mu, Macro, Args...)                  \
    forRealFluidGasEqns(Mu, sensibleEnthalpy, janafThermo, Macro, Args);      \
    forRealFluidGasEqns(Mu, sensibleInternalEnergy, janafThermo, Macro, Args);\
    forRealFluidGasEqns(Mu, absoluteEnthalpy, janafThermo, Macro, Args);      \
    forRealFluidGasEqns(Mu, absoluteInternalEnergy, janafThermo, Macro, Args)

#define forRealFluidGasTransports(Macro, Args...)                             \
    forRealFluidGasEnergiesAndThermos(chungTransport, Macro, Args)

#define forRealFluidGases(Macro, Args...)                                     \
    forRealFluidGasTransports(Macro, Args)

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
