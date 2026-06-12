/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2014-2023 OpenFOAM Foundation
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

#include "SRKGas.H"
#include "IOstreams.H"
#include "Switch.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class Specie>
Foam::SRKGas<Specie>::SRKGas
(
    const word& name,
    const dictionary& dict
)
:
    Specie(name, dict),
    c_(0),
    cq0_(0), cq1_(0), cq2_(0), cTlo_(0), cThi_(0)
{
    const scalar Tc = this->Tc();
    const scalar Pc = this->Pc();
    const scalar omega = this->omega();

    b_ = 0.08664*Foam::constant::thermodynamic::RR*Tc/Pc;

    const scalar a =
        0.42747*sqr(Foam::constant::thermodynamic::RR*Tc)/Pc;

    const scalar S = 0.48508 + 1.5517*omega - 0.15613*sqr(omega);

    coef1_ = a*sqr(1.0 + S);

    coef2_ = a*2*S*(1 + S)/sqrt(Tc);

    coef3_ = a*sqr(S)/Tc;

    // Optional Peneloux volume translation
    // (Peneloux, Rauzy & Freze, Fluid Phase Equilib. 8 (1982) 7-23).
    // Entry forms accepted inside the species subDict "rfProperties":
    //   1. Explicit shift:   c  <scalar>;            // [m^3/kmol]
    //   2. Automatic form:   penelouxShift true;     // uses Rackett Z_RA
    // Rackett compressibility Z_RA from Spencer & Danner, J. Chem. Eng. Data
    // 17 (1972) 236-241:  Z_RA = 0.29056 - 0.08775 * omega.
    // Recommended shift (Peneloux 1982, Eq. 10):
    //   c = 0.40768 * (0.29441 - Z_RA) * R * Tc / Pc
    const dictionary& rfDict = dict.subDict("rfProperties");
    if (rfDict.found("c"))
    {
        c_ = rfDict.lookup<scalar>("c");
    }
    else if (rfDict.lookupOrDefault<Switch>("penelouxShift", false))
    {
        const scalar Zra = 0.29056 - 0.08775*omega;
        c_ = 0.40768*(0.29441 - Zra)
            *Foam::constant::thermodynamic::RR*Tc/Pc;
    }

    // Optional temperature-dependent Peneloux translation c(T):
    //   penelouxCoeffs (cq0 cq1 cq2 Tlo Thi);   // [m^3/kmol, ..., K, K]
    // Liquid branch c(T) = cq0 + cq1 T + cq2 T^2 for T <= Tlo, smoothstep
    // ramp to the gas baseline c (above) over [Tlo, Thi]. Calibrated for
    // strongly non-ideal fluids (H2O) whose SRK liquid-density error grows
    // with T; see test/proto_penelouxCT.py.
    if (rfDict.found("penelouxCoeffs"))
    {
        const scalarList pc(rfDict.lookup<scalarList>("penelouxCoeffs"));
        if (pc.size() != 5)
        {
            FatalErrorInFunction
                << "penelouxCoeffs for " << name << " must have 5 entries "
                << "(cq0 cq1 cq2 Tlo Thi); got " << pc.size()
                << exit(FatalError);
        }
        cq0_ = pc[0]; cq1_ = pc[1]; cq2_ = pc[2];
        cTlo_ = pc[3]; cThi_ = pc[4];
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class Specie>
void Foam::SRKGas<Specie>::write(Ostream& os) const
{
    Specie::write(os);
}


// * * * * * * * * * * * * * * * Ostream Operator  * * * * * * * * * * * * * //

template<class Specie>
Foam::Ostream& Foam::operator<<
(
    Ostream& os,
    const SRKGas<Specie>& srk
)
{
    srk.write(os);
    return os;
}


// ************************************************************************* //
