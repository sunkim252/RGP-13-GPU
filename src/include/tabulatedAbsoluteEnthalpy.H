/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2024 OpenFOAM Foundation
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
    Foam::tabulatedAbsoluteEnthalpy

Description
    Drop-in replacement for Foam::absoluteEnthalpy for tabulated (FGM/FPV)
    solvers, whose enthalpy->temperature inversion (The) is BYPASSED: it
    returns the initial-guess temperature T0 unchanged.

    In the FGM/FPV closure the temperature is an algebraic function of the
    manifold coordinates (Z, gZ, c, dh) and is set directly from the table in
    fgmFluid::updateManifold() BEFORE thermo.correct(); the energy field he is
    then re-seeded to he(p, T_table). T0 (the field's current value passed to
    The()) therefore already carries the tabulated temperature, so the janaf
    he<->T Newton is redundant.

    Bypassing it is essential for cryogenic streams: 100 K LOX is ~200 K below
    O2's janaf Tlow (300 K), so the stock Tha() Newton extrapolates the
    ideal-gas Cp into its unphysical region and overshoots the real-fluid
    pseudo-critical Cp peak, aborting with an FPE in janafThermo::Cp. The
    4D table already provides T(Z=0, dh_ox) ~ 100 K and the SRK EOS gives the
    dense density at that T; only the inversion was broken. Returning T0 closes
    the gap the thermophysicalPredictor.C comment already assumes
    ("T is taken from the table, not inverted from h").

    typeName() deliberately remains "absoluteEnthalpy" so existing FPV cases
    (energy absoluteEnthalpy;) select it unchanged; only the real-fluid thermo
    chain is instantiated with this energy type.

\*---------------------------------------------------------------------------*/

#ifndef tabulatedAbsoluteEnthalpy_H
#define tabulatedAbsoluteEnthalpy_H

#include "scalar.H"
#include "word.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class tabulatedAbsoluteEnthalpy Declaration
\*---------------------------------------------------------------------------*/

template<class Thermo>
class tabulatedAbsoluteEnthalpy
{

public:

    // Constructors

        //- Construct
        tabulatedAbsoluteEnthalpy()
        {}


    // Member Functions

        //- Return the instantiated type name (kept as "absoluteEnthalpy"
        //  so FPV cases select it without any dictionary change)
        static word typeName()
        {
            return "absoluteEnthalpy";
        }


        // Fundamental properties

            static bool enthalpy()
            {
                return true;
            }

            static word energyName()
            {
                return "ha";
            }

            // Heat capacity at constant pressure [J/kg/K]
            scalar Cpv
            (
                const Thermo& thermo,
                const scalar p,
                const scalar T
            ) const
            {
                return thermo.Cp(p, T);
            }

            // Absolute enthalpy [J/kg]
            scalar he
            (
                const Thermo& thermo,
                const scalar p,
                const scalar T
            ) const
            {
                return thermo.ha(p, T);
            }

            //- Temperature from absolute enthalpy given an initial temperature
            //  T0. BYPASSED for the tabulated closure: the manifold lookup has
            //  already set T (= T0); return it unchanged so the janaf he<->T
            //  Newton (which FPEs on cryogenic, sub-Tlow cells) is never run.
            scalar The
            (
                const Thermo& thermo,
                const scalar h,
                const scalar p,
                const scalar T0
            ) const
            {
                return T0;
            }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
