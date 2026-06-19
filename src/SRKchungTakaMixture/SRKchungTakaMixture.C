/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2023 OpenFOAM Foundation
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

#include "SRKchungTakaMixture.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * Private Member Functions * * * * * * * * * * * * * //

template<class ThermoType>
void Foam::SRKchungTakaMixture<ThermoType>::calculateRealGas
(
    const List<scalar>& X,
    scalar& bM,
    scalar& coef1,
    scalar& coef2,
    scalar& coef3,
    scalar& cM,
    scalar& sigmaM,
    scalar& epsilonkM,
    scalar& MM,
    scalar& VcM,
    scalar& TcM,
    scalar& omegaM,
    scalar& miuiM,
    scalar& kappaiM
) const
{
    // Linear mixing for SRK co-volume
    forAll(BM_, i)
    {
        bM = bM + X[i]*BM_[i];
    }

    // Linear mole-fraction mixing for Peneloux volume shift.
    // Peneloux et al. (1982) showed the translation c can be applied to
    // mixtures directly as sum_i X_i c_i without affecting vapour
    // pressures.
    forAll(CM_, i)
    {
        cM = cM + X[i]*CM_[i];
    }

    // Quadratic mixing for all pair-interaction parameters
    // Exploit symmetry: matrices are symmetric (geometric/arithmetic means)
    // so sum = diagonal + 2 * upper triangle
    scalar sigma3M = 0, epsilonkM0 = 0, omegaM0 = 0, MM0 = 0, miuiM0 = 0;

    // Diagonal terms (i == j)
    forAll(COEF1_, i)
    {
        const scalar Xi2 = X[i]*X[i];
        coef1      += Xi2*COEF1_[i][i];
        coef2      += Xi2*COEF2_[i][i];
        coef3      += Xi2*COEF3_[i][i];
        sigma3M    += Xi2*SIGMA3M_[i][i];
        epsilonkM0 += Xi2*EPSILONKM0_[i][i];
        omegaM0    += Xi2*OMEGAM0_[i][i];
        MM0        += Xi2*MM0_[i][i];
        miuiM0     += Xi2*MIUIM0_[i][i];
        kappaiM    += Xi2*KAPPAIM_[i][i];
    }

    // Off-diagonal terms (i < j), doubled by symmetry
    for (label i = 0; i < COEF1_.size(); i++)
    {
        for (label j = i + 1; j < COEF1_.size(); j++)
        {
            const scalar twoXiXj = 2.0*X[i]*X[j];
            coef1      += twoXiXj*COEF1_[i][j];
            coef2      += twoXiXj*COEF2_[i][j];
            coef3      += twoXiXj*COEF3_[i][j];
            sigma3M    += twoXiXj*SIGMA3M_[i][j];
            epsilonkM0 += twoXiXj*EPSILONKM0_[i][j];
            omegaM0    += twoXiXj*OMEGAM0_[i][j];
            MM0        += twoXiXj*MM0_[i][j];
            miuiM0     += twoXiXj*MIUIM0_[i][j];
            kappaiM    += twoXiXj*KAPPAIM_[i][j];
        }
    }

    //- For visc. and cond. in Chung's model
    if (sigma3M == 0) { sigma3M = 1e-30; }
    sigmaM = cbrt(sigma3M);

    epsilonkM = epsilonkM0/sigma3M;
    if (epsilonkM == 0) { epsilonkM = 1e-30; }

    const scalar sigmaOverConst = sigmaM/0.809;
    VcM = sigmaOverConst*sigmaOverConst*sigmaOverConst;
    if (VcM == 0) { VcM = 1e-30; }

    TcM = 1.2593*epsilonkM;

    omegaM = omegaM0/sigma3M;

    const scalar MM0_term = MM0/(epsilonkM*sqr(sigmaM));
    MM = sqr(MM0_term);
    if (MM == 0) { MM = 1e-30; }

    miuiM = sqrt(sqrt(miuiM0*sigma3M*epsilonkM));
}


template<class ThermoType>
const typename Foam::SRKchungTakaMixture<ThermoType>::thermoMixtureType&
Foam::SRKchungTakaMixture<ThermoType>::calcMixture
(
    const scalarFieldListSlice& Y
) const
{
    // Mass-fraction weighted mixing of base thermo coefficients
    mixture_ = Y[0]*this->specieThermos()[0];

    for (label n = 1; n < Y.size(); n++)
    {
        mixture_ += Y[n]*this->specieThermos()[n];
    }

    // Convert mass fractions to mole fractions
    const label nSpecies = Y.size();
    List<scalar> X(nSpecies);
    List<scalar> Yl(nSpecies);
    scalar sumXb = 0.0;

    forAll(X, i)
    {
        sumXb = sumXb + Y[i]/ListW_[i];
    }
    if (sumXb == 0) { sumXb = 1e-30; }

    forAll(X, i)
    {
        X[i] = (Y[i]/ListW_[i])/sumXb;
        Yl[i] = Y[i];
        if (X[i] <= 0) { X[i] = 0; }
        if (Yl[i] <= 0) { Yl[i] = 0; }
    }

    // Calculate real-gas mixture parameters
    scalar bM = 0, coef1 = 0, coef2 = 0, coef3 = 0, cM = 0;
    scalar sigmaM = 0, epsilonkM = 0, VcM = 0, TcM = 0;
    scalar omegaM = 0, MM = 0, miuiM = 0, kappaiM = 0;

    calculateRealGas
    (
        X, bM, coef1, coef2, coef3, cM,
        sigmaM, epsilonkM, MM, VcM, TcM, omegaM, miuiM, kappaiM
    );

    // Update coefficients for mixture in SRK
    mixture_.updateEoS(bM, coef1, coef2, coef3, cM);

    // Correct mole fractions for mass diffusivity calculation
    scalar WmixCorrect = 0.0, sumXcorrected = 0.0;
    forAll(X, i)
    {
        X[i] = X[i] + 1e-40;
        sumXcorrected = sumXcorrected + X[i];
    }

    forAll(X, i)
    {
        X[i] = X[i]/sumXcorrected;
        WmixCorrect = WmixCorrect + X[i]*ListW_[i];
    }

    forAll(Yl, i)
    {
        Yl[i] = X[i]*ListW_[i]/WmixCorrect;
    }

    if (computeSpeciesDiffusion_)
    {
        // Compute composition-dependent Tc and Pc for diffusivity (Takahashi)
        // Exploit symmetry: TCMD[i][j] == TCMD[j][i]
        List<List<scalar>> TCMD(nSpecies);
        List<List<scalar>> PCMD(nSpecies);

        // Pre-allocate inner lists
        forAll(TCMD, i)
        {
            TCMD[i].setSize(nSpecies);
            PCMD[i].setSize(nSpecies);
        }

        for (label i = 0; i < nSpecies; i++)
        {
            // Diagonal
            TCMD[i][i] = ListTc_[i];
            if (TCMD[i][i] == 0) { TCMD[i][i] = 1e-40; }
            PCMD[i][i] = ListPc_[i];
            if (PCMD[i][i] == 0) { PCMD[i][i] = 1e-40; }

            // Upper triangle, mirror to lower
            for (label j = i + 1; j < nSpecies; j++)
            {
                const scalar XiPlusXj = X[i] + X[j];
                const scalar invSum =
                    (XiPlusXj == 0) ? 0.0 : 1.0/XiPlusXj;

                scalar tcVal =
                    (X[i]*ListTc_[i] + X[j]*ListTc_[j])*invSum;
                if (tcVal == 0) { tcVal = 1e-40; }

                scalar pcVal =
                    (X[i]*ListPc_[i] + X[j]*ListPc_[j])*invSum;
                if (pcVal == 0) { pcVal = 1e-40; }

                TCMD[i][j] = tcVal;
                TCMD[j][i] = tcVal;
                PCMD[i][j] = pcVal;
                PCMD[j][i] = pcVal;
            }
        }

        // Update coefficients for mixture in Chung's model
        mixture_.updateTRANS
        (
            sigmaM, epsilonkM, MM, VcM, TcM, omegaM, miuiM, kappaiM,
            Yl, X, TCMD, PCMD, MMD_, SIGMD_
        );
    }
    else
    {
        // Tier-0: species diffusivity (Dimix) is never consumed (control-
        // variable FPV, no per-species YEqn), so skip the O(n^2) Takahashi
        // Tcmd/Pcmd build + copies. mu/kappa need only the scalar Chung
        // params + Xmd_(=X)/Ymd_(=Yl) -> transport stays bit-identical.
        mixture_.updateTRANS_noDiffusion
        (
            sigmaM, epsilonkM, MM, VcM, TcM, omegaM, miuiM, kappaiM, Yl, X
        );
    }

    return mixture_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::SRKchungTakaMixture<ThermoType>::SRKchungTakaMixture
(
    const dictionary& dict
)
:
    multicomponentMixture<ThermoType>(dict),
    mixture_("mixture", this->specieThermos()[0]),
    numberOfSpecies_(this->specieThermos().size()),
    ListW_(numberOfSpecies_),
    //- For SRK
    BM_(numberOfSpecies_),
    COEF1_(numberOfSpecies_),
    COEF2_(numberOfSpecies_),
    COEF3_(numberOfSpecies_),
    CM_(numberOfSpecies_, scalar(0)),
    KIJ_(numberOfSpecies_),
    //- For visc. and cond. in Chung's model
    SIGMA3M_(numberOfSpecies_),
    EPSILONKM0_(numberOfSpecies_),
    OMEGAM0_(numberOfSpecies_),
    MM0_(numberOfSpecies_),
    MIUIM0_(numberOfSpecies_),
    KAPPAIM_(numberOfSpecies_),
    //- For diffusion in Takahashi's model
    ListTc_(numberOfSpecies_),
    ListPc_(numberOfSpecies_),
    MMD_(numberOfSpecies_),
    SIGMD_(numberOfSpecies_),
    computeSpeciesDiffusion_
    (
        dict.lookupOrDefault<Switch>("speciesDiffusion", true)
    )
{
    const scalar RR = Foam::constant::thermodynamic::RR;

    Info<< "SRKchungTakaMixture: speciesDiffusion = "
        << computeSpeciesDiffusion_
        << " (Takahashi binary-diffusion matrices "
        << (computeSpeciesDiffusion_ ? "built" : "SKIPPED [Tier-0]") << ")"
        << endl;

    // Pre-compute single-species quantities
    forAll(BM_, i)
    {
        ListW_[i]  = this->specieThermos()[i].W();
        BM_[i]     = 0.08664*RR
                    *this->specieThermos()[i].Tc()
                    /this->specieThermos()[i].Pc();
        ListTc_[i] = this->specieThermos()[i].Tc();
        ListPc_[i] = this->specieThermos()[i].Pc();
        CM_[i]     = this->specieThermos()[i].c();
    }

    // Initialise k_ij matrix with zeros
    forAll(KIJ_, i)
    {
        KIJ_[i].setSize(numberOfSpecies_, scalar(0));
    }

    // Optional binaryInteraction subDict:
    //   binaryInteraction
    //   {
    //       O2_H2O  { kij -0.015; }
    //       H2_N2   { kij  0.093; }
    //       default { kij  0;     }
    //   }
    // Each entry name "A_B" identifies the pair (A, B); the scalar kij is
    // applied symmetrically. Species names are resolved against the
    // multicomponent species list.
    if (dict.found("binaryInteraction"))
    {
        const dictionary& bd = dict.subDict("binaryInteraction");
        forAllConstIter(dictionary, bd, iter)
        {
            if (!iter().isDict())
            {
                continue;
            }
            const word& pair = iter().keyword();
            if (pair == "default")
            {
                continue;
            }
            const std::size_t sep = pair.find('_');
            if (sep == std::string::npos)
            {
                continue;
            }
            const word a = pair.substr(0, sep);
            const word b = pair.substr(sep + 1);
            label ia = -1, ib = -1;
            forAll(this->specieThermos(), s)
            {
                if (this->specieThermos()[s].name() == a) ia = s;
                if (this->specieThermos()[s].name() == b) ib = s;
            }
            if (ia < 0 || ib < 0)
            {
                continue;
            }
            const scalar kij =
                iter().dict().lookupOrDefault<scalar>("kij", 0);
            KIJ_[ia][ib] = kij;
            KIJ_[ib][ia] = kij;
        }
    }

    // Pre-compute pair-interaction matrices
    List<scalar> nCOEF1(numberOfSpecies_);
    List<scalar> nCOEF2(numberOfSpecies_);
    List<scalar> nCOEF3(numberOfSpecies_);

    List<scalar> nSIGMA3M(numberOfSpecies_);
    List<scalar> nEPSILONKM0(numberOfSpecies_);
    List<scalar> nOMEGAM0(numberOfSpecies_);
    List<scalar> nMM0(numberOfSpecies_);
    List<scalar> nMIUIM0(numberOfSpecies_);
    List<scalar> nKAPPAIM(numberOfSpecies_);

    List<scalar> nMMD(numberOfSpecies_);
    List<scalar> nSIGMD(numberOfSpecies_);

    forAll(COEF1_, i)
    {
        forAll(nCOEF1, j)
        {
            // SRK binary interaction factor (1 - k_ij) applied to the
            // cross terms only (i != j). Oefelein Eq. (25).
            const scalar oneMinusKij = (i == j) ? 1.0 : (1.0 - KIJ_[i][j]);

            //- For SRK
            nCOEF1[j] =
                sqrt
                (
                    (0.42747*pow(RR*this->specieThermos()[i].Tc(), 2)
                    /this->specieThermos()[i].Pc())
                   *(0.42747*pow(RR*this->specieThermos()[j].Tc(), 2)
                    /this->specieThermos()[j].Pc())
                )
               *(1 + (0.48508
                    + 1.5517*this->specieThermos()[i].omega()
                    - 0.15613*pow(this->specieThermos()[i].omega(), 2)))
               *(1 + (0.48508
                    + 1.5517*this->specieThermos()[j].omega()
                    - 0.15613*pow(this->specieThermos()[j].omega(), 2)))
               *oneMinusKij;

            nCOEF2[j] =
                sqrt
                (
                    (0.42747*pow(RR*this->specieThermos()[i].Tc(), 2)
                    /this->specieThermos()[i].Pc())
                   *(0.42747*pow(RR*this->specieThermos()[j].Tc(), 2)
                    /this->specieThermos()[j].Pc())
                )
               *(
                  (
                    (1.0 + (0.48508
                          + 1.5517*this->specieThermos()[j].omega()
                          - 0.15613*pow(this->specieThermos()[j].omega(), 2)))
                   *(0.48508
                    + 1.5517*this->specieThermos()[i].omega()
                    - 0.15613*pow(this->specieThermos()[i].omega(), 2))
                   /sqrt(this->specieThermos()[i].Tc())
                  )
                 +(
                    (1.0 + (0.48508
                          + 1.5517*this->specieThermos()[i].omega()
                          - 0.15613*pow(this->specieThermos()[i].omega(), 2)))
                   *(0.48508
                    + 1.5517*this->specieThermos()[j].omega()
                    - 0.15613*pow(this->specieThermos()[j].omega(), 2))
                   /sqrt(this->specieThermos()[j].Tc())
                  )
                )
               *oneMinusKij;

            nCOEF3[j] =
                sqrt
                (
                    (0.42747*pow(RR*this->specieThermos()[i].Tc(), 2)
                    /this->specieThermos()[i].Pc())
                   *(0.42747*pow(RR*this->specieThermos()[j].Tc(), 2)
                    /this->specieThermos()[j].Pc())
                )
               *(0.48508
                + 1.5517*this->specieThermos()[i].omega()
                - 0.15613*pow(this->specieThermos()[i].omega(), 2))
               *(0.48508
                + 1.5517*this->specieThermos()[j].omega()
                - 0.15613*pow(this->specieThermos()[j].omega(), 2))
               /sqrt
                (
                    this->specieThermos()[i].Tc()
                   *this->specieThermos()[j].Tc()
                )
               *oneMinusKij;

            //- For visc. and cond. in Chung's model
            nSIGMA3M[j] =
                pow
                (
                    (0.809*pow(this->specieThermos()[i].Vc(), 1.0/3.0))
                   *(0.809*pow(this->specieThermos()[j].Vc(), 1.0/3.0)),
                    3.0/2
                );

            nEPSILONKM0[j] =
                pow
                (
                    0.809*pow(this->specieThermos()[i].Vc(), 1.0/3)
                   *0.809*pow(this->specieThermos()[j].Vc(), 1.0/3),
                    3.0/2
                )
               *pow
                (
                    (this->specieThermos()[i].Tc()/1.2593)
                   *(this->specieThermos()[j].Tc()/1.2593),
                    1.0/2
                );

            nOMEGAM0[j] =
                pow
                (
                    0.809*pow(this->specieThermos()[i].Vc(), 1.0/3)
                   *0.809*pow(this->specieThermos()[j].Vc(), 1.0/3),
                    3.0/2
                )
               *0.5
               *(this->specieThermos()[i].omega()
                + this->specieThermos()[j].omega());

            nMM0[j] =
                (0.809*pow(this->specieThermos()[i].Vc(), 1.0/3))
               *(0.809*pow(this->specieThermos()[j].Vc(), 1.0/3))
               *pow
                (
                    (this->specieThermos()[i].Tc()/1.2593)
                   *(this->specieThermos()[j].Tc()/1.2593),
                    1.0/2
                )
               *pow
                (
                    2*this->specieThermos()[i].W()
                   *this->specieThermos()[j].W()
                   /(this->specieThermos()[i].W()
                    + this->specieThermos()[j].W()),
                    1.0/2
                );

            nMIUIM0[j] =
                pow(this->specieThermos()[i].miui(), 2)
               *pow(this->specieThermos()[j].miui(), 2)
               /(pow
                 (
                     0.809*pow(this->specieThermos()[i].Vc(), 1.0/3)
                    *0.809*pow(this->specieThermos()[j].Vc(), 1.0/3),
                     3.0/2
                 )
                *pow
                 (
                     (this->specieThermos()[i].Tc()/1.2593)
                    *(this->specieThermos()[j].Tc()/1.2593),
                     0.5
                 )
                );

            nKAPPAIM[j] =
                pow
                (
                    this->specieThermos()[i].kappai()
                   *this->specieThermos()[j].kappai(),
                    1.0/2
                );

            //- For diffusivity
            nMMD[j] =
                1/this->specieThermos()[i].W()
              + 1/this->specieThermos()[j].W();

            nSIGMD[j] =
                pow(this->specieThermos()[i].sigmvi(), 1.0/3)
              + pow(this->specieThermos()[j].sigmvi(), 1.0/3);
        }

        COEF1_[i] = nCOEF1;
        COEF2_[i] = nCOEF2;
        COEF3_[i] = nCOEF3;

        SIGMA3M_[i]    = nSIGMA3M;
        EPSILONKM0_[i] = nEPSILONKM0;
        OMEGAM0_[i]    = nOMEGAM0;
        MM0_[i]        = nMM0;
        MIUIM0_[i]     = nMIUIM0;
        KAPPAIM_[i]    = nKAPPAIM;

        MMD_[i]  = nMMD;
        SIGMD_[i] = nSIGMD;
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ThermoType>
const typename
Foam::SRKchungTakaMixture<ThermoType>::thermoMixtureType&
Foam::SRKchungTakaMixture<ThermoType>::thermoMixture
(
    const scalarFieldListSlice& Y
) const
{
    return calcMixture(Y);
}


template<class ThermoType>
const typename
Foam::SRKchungTakaMixture<ThermoType>::transportMixtureType&
Foam::SRKchungTakaMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice& Y
) const
{
    return thermoMixture(Y);
}


template<class ThermoType>
const typename
Foam::SRKchungTakaMixture<ThermoType>::transportMixtureType&
Foam::SRKchungTakaMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice&,
    const thermoMixtureType& mixture
) const
{
    return mixture;
}


// ************************************************************************* //
