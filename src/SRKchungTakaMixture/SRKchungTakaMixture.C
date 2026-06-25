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
#include "FGMTable.H"
#include <cstdint>

// * * * * * * * * * * * * Private Member Functions * * * * * * * * * * * * * //

template<class ThermoType>
template<class CompType>
void Foam::SRKchungTakaMixture<ThermoType>::compositionToX
(
    const CompType& Y,
    List<scalar>& X,
    List<scalar>& Yl
) const
{
    const label nSpecies = Y.size();
    X.setSize(nSpecies);
    Yl.setSize(nSpecies);

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
}


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
    // Recover the internal cell index up front (if this slice is an internal
    // cell): both the Opt-1 base-blend node interpolation and the Tier-2/Opt-2
    // coefficient lookup key off it. &Y[0] is species 0 at the sliced element;
    // for an internal cell it lies inside species 0's internal field, so celli
    // is recovered by offset and the in-range test rejects patch faces / foreign
    // field sets (different allocation). uintptr_t avoids pointer-provenance UB.
    label celli = -1;
    if (refInternalField_ != nullptr && refInternalField_->size() > 0)
    {
        const std::uintptr_t y =
            reinterpret_cast<std::uintptr_t>(&Y[0]);
        const std::uintptr_t lo =
            reinterpret_cast<std::uintptr_t>(refInternalField_->begin());
        const std::uintptr_t hi =
            lo + std::uintptr_t(refInternalField_->size())*sizeof(scalar);
        if (y >= lo && y < hi)
        {
            celli = label((y - lo)/sizeof(scalar));
        }
    }

    // Mass-fraction weighted mixing of base thermo coefficients. Opt-1: for an
    // internal cell with node interpolation armed, blend the 16 bracketing
    // manifold-node mixtures (pre-built from the tabulated node Y) instead of
    // the 106-species sum. Bit-consistent by linearity: the cell Y is itself the
    // node interpolation of the table Y, so sum_corner w*nodeMix == sum_species
    // Y*thermo (FP order apart). Falls back to the species blend on patch faces
    // / when Opt-1 is off.
    if (useBaseBlendTab_ && celli >= 0 && nodeMixtures_.size())
    {
        label nodes[16];
        scalar weights[16];
        fgmTablePtr_->interpStencil
        (
            (*ZfieldPtr_)[celli], (*gZfieldPtr_)[celli],
            (*CfieldPtr_)[celli], (*chiFieldPtr_)[celli],
            nodes, weights
        );
        mixture_ = weights[0]*nodeMixtures_[nodes[0]];
        for (label m = 1; m < 16; m++)
        {
            if (weights[m] != scalar(0))
            {
                mixture_ += weights[m]*nodeMixtures_[nodes[m]];
            }
        }
    }
    else
    {
        mixture_ = Y[0]*this->specieThermos()[0];
        for (label n = 1; n < Y.size(); n++)
        {
            mixture_ += Y[n]*this->specieThermos()[n];
        }
    }

    // Convert mass fractions to mole fractions (shared with the tabulation
    // utility via compositionToX so live and tabulated coefficients use an
    // identical X).
    const label nSpecies = Y.size();
    List<scalar> X;
    List<scalar> Yl;
    compositionToX(Y, X, Yl);

    // Real-gas mixture parameters: bM/coef1-3/cM (SRK) and sigmaM..kappaiM
    // (Chung mu/kappa). These are pure functions of X and carry the O(n^2)
    // pair-mixing cost. Tier-2: for an internal-cell composition slice they are
    // looked up from the table-filled coeffFields_ instead of being rebuilt.
    scalar bM = 0, coef1 = 0, coef2 = 0, coef3 = 0, cM = 0;
    scalar sigmaM = 0, epsilonkM = 0, VcM = 0, TcM = 0;
    scalar omegaM = 0, MM = 0, miuiM = 0, kappaiM = 0;

    // Tier-2 coefficient lookup is valid for an internal-cell slice (celli was
    // recovered up front, shared with the Opt-1 base blend above).
    const bool useLookup =
        useTabulatedCoeffs_ && coeffFields_.size() == nCoeffs_ && celli >= 0;

    if (useLookup)
    {
        bM        = (*coeffFields_[0])[celli];
        coef1     = (*coeffFields_[1])[celli];
        coef2     = (*coeffFields_[2])[celli];
        coef3     = (*coeffFields_[3])[celli];
        cM        = (*coeffFields_[4])[celli];
        sigmaM    = (*coeffFields_[5])[celli];
        epsilonkM = (*coeffFields_[6])[celli];
        MM        = (*coeffFields_[7])[celli];
        VcM       = (*coeffFields_[8])[celli];
        TcM       = (*coeffFields_[9])[celli];
        omegaM    = (*coeffFields_[10])[celli];
        miuiM     = (*coeffFields_[11])[celli];
        kappaiM   = (*coeffFields_[12])[celli];

        // One-off audit: compare looked-up vs freshly computed coefficients for
        // the first few cells after enabling, to confirm bit-consistency of the
        // tabulated path. Cheap (a handful of cells, once).
        if (coeffDiagCount_ > 0)
        {
            scalar bMc=0, c1=0, c2=0, c3=0, cMc=0, sgM=0, epM=0, MMc=0,
                   VcMc=0, TcMc=0, omM=0, miM=0, kaM=0;
            calculateRealGas
            (
                X, bMc, c1, c2, c3, cMc,
                sgM, epM, MMc, VcMc, TcMc, omM, miM, kaM
            );
            const scalar denom = max(mag(bMc), VSMALL);
            Info<< "[Tier2-DIAG] cell " << celli << " tab/live:"
                << " bM " << bM << "/" << bMc
                << " sigmaM " << sigmaM << "/" << sgM
                << " TcM " << TcM << "/" << TcMc
                << " epsilonkM " << epsilonkM << "/" << epM
                << " (relErr bM " << mag(bM - bMc)/denom << ")" << endl;
            coeffDiagCount_--;
        }
    }
    else
    {
        // Opt-2: patch-face tabulation. A patch composition slice has &Y[0]
        // inside that patch's species-0 boundary field; recover facei by offset
        // (same uintptr_t range trick as the internal cell above) and look the
        // coefficients up from the table-filled boundary fields instead of the
        // live O(n^2) pair sum. Falls through to live mixing for any slice that
        // is neither an internal cell nor a registered patch face.
        bool patchHit = false;
        if (useTabulatedCoeffs_ && patchRefFields_.size())
        {
            const std::uintptr_t y =
                reinterpret_cast<std::uintptr_t>(&Y[0]);
            forAll(patchRefFields_, p)
            {
                const scalarField* pf = patchRefFields_[p];
                if (pf == nullptr || pf->empty())
                {
                    continue;
                }
                const std::uintptr_t lo =
                    reinterpret_cast<std::uintptr_t>(pf->begin());
                const std::uintptr_t hi =
                    lo + std::uintptr_t(pf->size())*sizeof(scalar);
                if (y >= lo && y < hi)
                {
                    const label fi = label((y - lo)/sizeof(scalar));
                    const List<const scalarField*>& pc = patchCoeffFields_[p];
                    bM        = (*pc[0])[fi];
                    coef1     = (*pc[1])[fi];
                    coef2     = (*pc[2])[fi];
                    coef3     = (*pc[3])[fi];
                    cM        = (*pc[4])[fi];
                    sigmaM    = (*pc[5])[fi];
                    epsilonkM = (*pc[6])[fi];
                    MM        = (*pc[7])[fi];
                    VcM       = (*pc[8])[fi];
                    TcM       = (*pc[9])[fi];
                    omegaM    = (*pc[10])[fi];
                    miuiM     = (*pc[11])[fi];
                    kappaiM   = (*pc[12])[fi];
                    patchHit = true;
                    break;
                }
            }
        }
        if (!patchHit)
        {
            calculateRealGas
            (
                X, bM, coef1, coef2, coef3, cM,
                sigmaM, epsilonkM, MM, VcM, TcM, omegaM, miuiM, kappaiM
            );
        }
    }

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
    ),
    useTabulatedCoeffs_(false),
    refInternalField_(nullptr),
    coeffFields_(),
    coeffDiagCount_(0),
    patchRefFields_(),
    patchCoeffFields_(),
    nodeMixtures_(),
    useBaseBlendTab_(false),
    fgmTablePtr_(nullptr),
    ZfieldPtr_(nullptr),
    gZfieldPtr_(nullptr),
    CfieldPtr_(nullptr),
    chiFieldPtr_(nullptr)
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


// - - - - tabulatedRealGasMixture interface (Tier-2 manifold tabulation) - - //

template<class ThermoType>
void Foam::SRKchungTakaMixture<ThermoType>::realGasCoeffs
(
    const List<scalar>& Y,
    List<scalar>& coeffs
) const
{
    List<scalar> X;
    List<scalar> Yl;
    compositionToX(Y, X, Yl);

    // calculateRealGas accumulates onto its outputs, so they must start at 0.
    coeffs.setSize(nCoeffs_);
    coeffs = scalar(0);
    // Order MUST match coeffNames() and the lookup in calcMixture():
    //   bM, coef1, coef2, coef3, cM,
    //   sigmaM, epsilonkM, MM, VcM, TcM, omegaM, miuiM, kappaiM
    calculateRealGas
    (
        X,
        coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4],
        coeffs[5], coeffs[6], coeffs[7], coeffs[8], coeffs[9], coeffs[10],
        coeffs[11], coeffs[12]
    );
}


template<class ThermoType>
void Foam::SRKchungTakaMixture<ThermoType>::enableCoeffTabulation
(
    const scalarField& refInternal,
    const List<const scalarField*>& coeffFields
) const
{
    if (coeffFields.size() != nCoeffs_)
    {
        FatalErrorInFunction
            << "expected " << label(nCoeffs_) << " coefficient fields, got "
            << coeffFields.size() << exit(FatalError);
    }
    refInternalField_ = &refInternal;
    coeffFields_ = coeffFields;
    useTabulatedCoeffs_ = true;
    coeffDiagCount_ = 3;

    Info<< "SRKchungTakaMixture: Tier-2 tabulated real-gas coefficients ENABLED"
        << " (per-cell calculateRealGas O(n^2) replaced by table lookup; "
        << "patch faces use live mixing)" << endl;
}


template<class ThermoType>
void Foam::SRKchungTakaMixture<ThermoType>::enablePatchCoeffTabulation
(
    const List<const scalarField*>& patchRefs,
    const List<List<const scalarField*>>& patchCoeffFields
) const
{
    if (patchRefs.size() != patchCoeffFields.size())
    {
        FatalErrorInFunction
            << "patchRefs (" << patchRefs.size() << ") and patchCoeffFields ("
            << patchCoeffFields.size() << ") size mismatch" << exit(FatalError);
    }
    patchRefFields_ = patchRefs;
    patchCoeffFields_ = patchCoeffFields;

    label nReg = 0;
    forAll(patchRefFields_, p)
    {
        if (patchRefFields_[p] != nullptr && !patchRefFields_[p]->empty())
        {
            nReg++;
        }
    }

    Info<< "SRKchungTakaMixture: Opt-2 patch-face coefficient lookup ENABLED ("
        << nReg << " non-empty patches; live calculateRealGas now skipped on "
        << "boundary faces too)" << endl;
}


template<class ThermoType>
void Foam::SRKchungTakaMixture<ThermoType>::enableBaseBlendTabulation
(
    const List<List<scalar>>& nodeY,
    const FGMTable& table,
    const scalarField& Zfield,
    const scalarField& gZfield,
    const scalarField& Cfield,
    const scalarField& chiField
) const
{
    if (nodeY.size() != numberOfSpecies_)
    {
        FatalErrorInFunction
            << "nodeY species (" << nodeY.size() << ") != numberOfSpecies ("
            << numberOfSpecies_ << ")" << exit(FatalError);
    }
    const label nNodes = table.nTot();

    // Pre-blend the base thermo at every manifold node:
    //   nodeMix[n] = sum_species nodeY[species][n] * specieThermos()[species].
    // Built once (reuses the Tier-3 cheap thermo operator+=); thereafter the
    // per-internal-cell base blend is a 16-corner interpolation of these.
    nodeMixtures_.clear();
    nodeMixtures_.setSize(nNodes);
    for (label n = 0; n < nNodes; n++)
    {
        thermoMixtureType m(nodeY[0][n]*this->specieThermos()[0]);
        for (label s = 1; s < numberOfSpecies_; s++)
        {
            m += nodeY[s][n]*this->specieThermos()[s];
        }
        nodeMixtures_.set(n, new thermoMixtureType(m));
    }

    fgmTablePtr_ = &table;
    ZfieldPtr_   = &Zfield;
    gZfieldPtr_  = &gZfield;
    CfieldPtr_   = &Cfield;
    chiFieldPtr_ = &chiField;
    useBaseBlendTab_ = true;

    Info<< "SRKchungTakaMixture: Opt-1 base-thermo node interpolation ENABLED ("
        << nNodes << " manifold nodes pre-blended; per-internal-cell "
        << numberOfSpecies_ << "-species base blend replaced by 16-corner node "
        << "blend)" << endl;
}


template<class ThermoType>
void Foam::SRKchungTakaMixture<ThermoType>::disableCoeffTabulation() const
{
    useTabulatedCoeffs_ = false;
    refInternalField_ = nullptr;
    coeffFields_.clear();
    coeffDiagCount_ = 0;
    patchRefFields_.clear();
    patchCoeffFields_.clear();
    useBaseBlendTab_ = false;
    nodeMixtures_.clear();
    fgmTablePtr_ = nullptr;
    ZfieldPtr_ = nullptr;
    gZfieldPtr_ = nullptr;
    CfieldPtr_ = nullptr;
    chiFieldPtr_ = nullptr;
}


// ************************************************************************* //
