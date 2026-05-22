/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Copyright (C) 2024-2025 RGP-13
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#ifndef forRealFluidGasesElyHanley_H
#define forRealFluidGasesElyHanley_H

#include "rfSpecie.H"
#include "SRKGas.H"
#include "janafThermo.H"

#include "sensibleEnthalpy.H"
#include "sensibleInternalEnergy.H"
#include "absoluteEnthalpy.H"
#include "absoluteInternalEnergy.H"

#include "elyHanleyTransport.H"
#include "thermo.H"

#include "forThermo.H"

#define forRealFluidGasEqnsEH(Mu, He, Cp, Macro, Args...)                     \
    forThermo(Mu, He, Cp, SRKGas, rfSpecie, Macro, Args)

#define forRealFluidGasEnergiesAndThermosEH(Mu, Macro, Args...)               \
    forRealFluidGasEqnsEH(Mu, sensibleEnthalpy, janafThermo, Macro, Args);    \
    forRealFluidGasEqnsEH(Mu, sensibleInternalEnergy, janafThermo, Macro, Args);\
    forRealFluidGasEqnsEH(Mu, absoluteEnthalpy, janafThermo, Macro, Args);    \
    forRealFluidGasEqnsEH(Mu, absoluteInternalEnergy, janafThermo, Macro, Args)

#define forRealFluidGasTransportsEH(Macro, Args...)                           \
    forRealFluidGasEnergiesAndThermosEH(elyHanleyTransport, Macro, Args)

#define forRealFluidGasesEH(Macro, Args...)                                   \
    forRealFluidGasTransportsEH(Macro, Args)

#endif
