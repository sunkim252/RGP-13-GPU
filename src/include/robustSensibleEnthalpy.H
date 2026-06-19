/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Copyright (C) 2024-2025 RGP-13
    \\  /    A nd           |
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Class
    Foam::robustSensibleEnthalpy

Description
    Drop-in replacement for Foam::sensibleEnthalpy whose enthalpy->temperature
    inversion (The) is a safeguarded Newton + bisection solver instead of the
    plain Newton used by Foam::species::thermo::THs.

    The plain Newton overshoots and limit-cycles near the real-fluid
    pseudo-critical Cp peak (large dCp/dT), producing
    "Maximum number of iterations exceeded" aborts for cold dense mixing
    cells. The bisection fallback on [Tlo, Thi] always converges. Algorithm
    after Nam (UNIST realFluidFoam-8, OpenFOAM-8 species/thermoI.H).

    typeName() deliberately remains "sensibleEnthalpy" so existing cases
    (energy sensibleEnthalpy;) select it unchanged; only the real-fluid
    thermo chain is instantiated with this energy type.

\*---------------------------------------------------------------------------*/

#ifndef robustSensibleEnthalpy_H
#define robustSensibleEnthalpy_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                  Class robustSensibleEnthalpy Declaration
\*---------------------------------------------------------------------------*/

template<class Thermo>
class robustSensibleEnthalpy
{

public:

    // Constructors

        //- Construct
        robustSensibleEnthalpy()
        {}


    // Member Functions

        //- Return the instantiated type name (kept as "sensibleEnthalpy"
        //  so cases select it without any dictionary change)
        static word typeName()
        {
            return "sensibleEnthalpy";
        }


        // Fundamental properties

            static bool enthalpy()
            {
                return true;
            }

            static word energyName()
            {
                return "h";
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

            // Sensible enthalpy [J/kg]
            scalar he
            (
                const Thermo& thermo,
                const scalar p,
                const scalar T
            ) const
            {
                return thermo.hs(p, T);
            }

            //- Temperature from sensible enthalpy given an initial
            //  temperature T0. Safeguarded Newton + bisection.
            scalar The
            (
                const Thermo& thermo,
                const scalar h,
                const scalar p,
                const scalar T0
            ) const
            {
                const int maxIter = 30;
                const scalar Ttol = max(T0, scalar(1))*1e-4;

                // Silent clamp to the thermo's valid range. Calling
                // thermo.limit() here floods the log with janafThermo
                // out-of-range Warnings on intermediate overshooting Newton
                // iterates (negative T on stiff pseudo-critical cells); the
                // bisection fallback below handles genuine non-convergence.
                const scalar Tlo = thermo.Tlow();
                const scalar Thi = thermo.Thigh();

                // 1) Newton iteration (does not abort on non-convergence)
                scalar Test = T0;
                scalar Tnew = T0;
                int iter = 0;
                do
                {
                    Test = Tnew;
                    const scalar cp = thermo.Cp(p, Test);
                    Tnew = max
                    (
                        Tlo,
                        min(Thi, Test - (thermo.hs(p, Test) - h)/cp)
                    );
                    if (++iter > maxIter)
                    {
                        break;
                    }
                } while (mag(Tnew - Test) > Ttol);

                if (mag(Tnew - Test) < Ttol)
                {
                    return Tnew;
                }

                // 2) Bisection fallback on [x1, x2] (root of hs(p,T) - h).
                //    Guaranteed to converge to a bounded T; prevents the
                //    solver from aborting on stiff pseudo-critical cells.
                const scalar biTol = 1e-3;
                scalar x1 = 50;
                scalar x2 = 4000;
                scalar x3 = 0.5*(x1 + x2);
                int n = 0;
                do
                {
                    x3 = 0.5*(x1 + x2);
                    if ((thermo.hs(p, x3) - h)*(thermo.hs(p, x2) - h) < 0)
                    {
                        x1 = x3;
                    }
                    else
                    {
                        x2 = x3;
                    }
                    if (++n > 200)
                    {
                        break;
                    }
                } while (mag(x2 - x1) > biTol);

                return 0.5*(x1 + x2);
            }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
