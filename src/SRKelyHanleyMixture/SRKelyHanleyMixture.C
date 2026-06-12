/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Copyright (C) 2024-2026 RGP-13
    \\  /    A nd           |
     \\/     M anipulation  |
\*---------------------------------------------------------------------------*/

#include "SRKelyHanleyMixture.H"
#include "thermodynamicConstants.H"

// * * * * * * * * * * * * Private Member Functions * * * * * * * * * * * * * //

template<class ThermoType>
void Foam::SRKelyHanleyMixture<ThermoType>::calculateRealGas
(
    const List<scalar>& X,
    scalar& bM,
    scalar& coef1,
    scalar& coef2,
    scalar& coef3,
    scalar& cM,
    scalar& MM,
    scalar& VcM,
    scalar& TcM,
    scalar& omegaM,
    scalar& ZcM
) const
{
    // ---------------- SRK EoS mixing rules (Oefelein 2019 Eqs. 24-25) ----
    //
    //   a_m = SUM_i SUM_j X_i X_j a_ij,
    //         a_ij = sqrt(a_i a_j) (1 - k_ij)        [Eq. 25]
    //   b_m = SUM_i SUM_j X_i X_j b_ij,
    //         b_ij = (1/8)(b_i^1/3 + b_j^1/3)^3 (1 - l_ij)
    //
    // a_m is split across three Soave temperature-functional terms
    // (COEF1..3, with k_ij already baked in by the constructor); b_m
    // uses the precomputed BIJ_ matrix (with l_ij baked in). This
    // replaces the previous linear b_m = SUM X_i b_i (Kay's rule).
    //
    // Peneloux c_m stays linear by Peneloux et al. (1982).
    forAll(CM_, i)
    {
        cM += X[i]*CM_[i];
    }

    forAll(COEF1_, i)
    {
        const scalar Xi2 = X[i]*X[i];
        coef1 += Xi2*COEF1_[i][i];
        coef2 += Xi2*COEF2_[i][i];
        coef3 += Xi2*COEF3_[i][i];
        bM    += Xi2*BIJ_[i][i];
    }
    for (label i = 0; i < COEF1_.size(); i++)
    {
        for (label j = i + 1; j < COEF1_.size(); j++)
        {
            const scalar twoXiXj = 2.0*X[i]*X[j];
            coef1 += twoXiXj*COEF1_[i][j];
            coef2 += twoXiXj*COEF2_[i][j];
            coef3 += twoXiXj*COEF3_[i][j];
            bM    += twoXiXj*BIJ_[i][j];
        }
    }

    // ---------------- Ely-Hanley mixture critical state ------------------
    // ECS reducing-ratio mixing rules of Oefelein 2019 Eqs. (13)-(17),
    // taken in the Leach-Chappelear-Leland shape-factor = 1 limit
    // (the full shape factors are still applied per-species inside
    // elyHanleyTransport when forming F_mu, F_lam, so this is a vdW1f
    // approximation only at the *mixing* step). In that limit
    //   h_i = V_c,i / V_c,o,    f_i = T_c,i / T_c,o,
    //   h_ij = (1/8)(h_i^1/3 + h_j^1/3)^3 (1 - l_ij),
    //   f_ij = sqrt(f_i f_j) (1 - k_ij),
    // so that
    //   V_c,m = h_x V_c,o = SUM SUM X_i X_j V_c,ij
    //   T_c,m V_c,m = (f_x V_c,o) (h_x V_c,o)
    //               = SUM SUM X_i X_j sqrt(T_ci T_cj) (1-k_ij) V_c,ij
    // The pair quantities VCIJ_ (= V_c,ij) and TVCIJ_ (= sqrt(T_ci T_cj)
    // (1-k_ij) V_c,ij) are precomputed in the constructor.
    //
    // omega_m and M_m use the linear Kay's rule (omega is exact under
    // arithmetic-mean omega_ij combining; M_m is the standard ECS
    // simplification).
    scalar PcM    = 0;
    scalar TcVcM  = 0;
    forAll(X, i)
    {
        omegaM += X[i]*ListOmega_[i];
        MM     += X[i]*ListW_[i];
        PcM    += X[i]*ListPc_[i];

        const scalar Xi2 = X[i]*X[i];
        VcM    += Xi2*VCIJ_[i][i];
        TcVcM  += Xi2*TVCIJ_[i][i];
    }
    for (label i = 0; i < VCIJ_.size(); i++)
    {
        for (label j = i + 1; j < VCIJ_.size(); j++)
        {
            const scalar twoXiXj = 2.0*X[i]*X[j];
            VcM   += twoXiXj*VCIJ_[i][j];
            TcVcM += twoXiXj*TVCIJ_[i][j];
        }
    }

    if (VcM == 0) { VcM = 1e-30; }
    if (MM  == 0) { MM  = 1e-30; }
    TcM = TcVcM/VcM;
    if (TcM == 0) { TcM = 1e-30; }

    const scalar Ru    = Foam::constant::thermodynamic::RR;  // J/(kmol K)
    const scalar VcMSI = VcM*1e-3;                           // -> m^3/kmol
    if (PcM > 1e-6 && TcM > 1e-6)
    {
        ZcM = PcM*VcMSI/(Ru*TcM);
    }
    else
    {
        // Fall back to Pitzer if any species lacks P_c
        ZcM = 0.291 - 0.08*omegaM;
        if (ZcM <= 0) ZcM = 0.29;
    }
}


template<class ThermoType>
const typename Foam::SRKelyHanleyMixture<ThermoType>::thermoMixtureType&
Foam::SRKelyHanleyMixture<ThermoType>::calcMixture
(
    const scalarFieldListSlice& Y
) const
{
    // Mass-fraction weighted base thermo
    mixture_ = Y[0]*this->specieThermos()[0];
    for (label n = 1; n < Y.size(); n++)
    {
        mixture_ += Y[n]*this->specieThermos()[n];
    }

    const label nSpecies = Y.size();
    List<scalar> X(nSpecies);
    List<scalar> Yl(nSpecies);
    scalar sumXb = 0.0;

    forAll(X, i)
    {
        sumXb += Y[i]/ListW_[i];
    }
    if (sumXb == 0) { sumXb = 1e-30; }

    forAll(X, i)
    {
        X[i] = (Y[i]/ListW_[i])/sumXb;
        Yl[i] = Y[i];
        if (X[i]  <= 0) X[i]  = 0;
        if (Yl[i] <= 0) Yl[i] = 0;
    }

    // SRK + Ely-Hanley mixture aggregates
    scalar bM = 0, coef1 = 0, coef2 = 0, coef3 = 0, cM = 0;
    scalar MM = 0, VcM = 0, TcM = 0, omegaM = 0, ZcM = 0;

    calculateRealGas
    (
        X, bM, coef1, coef2, coef3, cM,
        MM, VcM, TcM, omegaM, ZcM
    );

    // Composition-weighted temperature-dependent Peneloux c(T) coefficients
    // (Peneloux 1982: c_m linear in mole fraction). The liquid->gas ramp
    // window is taken as the widest among the c(T)-active components.
    scalar cMq0 = 0, cMq1 = 0, cMq2 = 0, cMTlo = 0, cMThi = 0;
    forAll(CMq0_, i)
    {
        cMq0 += X[i]*CMq0_[i];
        cMq1 += X[i]*CMq1_[i];
        cMq2 += X[i]*CMq2_[i];
        cMTlo = max(cMTlo, CMTlo_[i]);
        cMThi = max(cMThi, CMThi_[i]);
    }

    // Push SRK + Peneloux (constant baseline cM + optional c(T)) into the EoS
    mixture_.updateEoS
    (
        bM, coef1, coef2, coef3, cM,
        cMq0, cMq1, cMq2, cMTlo, cMThi
    );

    // Composition-dependent Tc, Pc matrices for Fuller + Takahashi
    // (mole-weighted pair averages; identical to SRKchungTakaMixture)
    scalar WmixCorrect = 0.0, sumXcorrected = 0.0;
    forAll(X, i)
    {
        X[i] += 1e-40;
        sumXcorrected += X[i];
    }
    forAll(X, i)
    {
        X[i] /= sumXcorrected;
        WmixCorrect += X[i]*ListW_[i];
    }
    forAll(Yl, i)
    {
        Yl[i] = X[i]*ListW_[i]/WmixCorrect;
    }

    List<List<scalar>> TCMD(nSpecies);
    List<List<scalar>> PCMD(nSpecies);
    forAll(TCMD, i)
    {
        TCMD[i].setSize(nSpecies);
        PCMD[i].setSize(nSpecies);
    }
    for (label i = 0; i < nSpecies; i++)
    {
        TCMD[i][i] = ListTc_[i];
        if (TCMD[i][i] == 0) TCMD[i][i] = 1e-40;
        PCMD[i][i] = ListPc_[i];
        if (PCMD[i][i] == 0) PCMD[i][i] = 1e-40;

        for (label j = i + 1; j < nSpecies; j++)
        {
            const scalar XiPlusXj = X[i] + X[j];
            const scalar invSum = (XiPlusXj == 0) ? 0.0 : 1.0/XiPlusXj;

            scalar tcVal = (X[i]*ListTc_[i] + X[j]*ListTc_[j])*invSum;
            if (tcVal == 0) tcVal = 1e-40;
            scalar pcVal = (X[i]*ListPc_[i] + X[j]*ListPc_[j])*invSum;
            if (pcVal == 0) pcVal = 1e-40;

            TCMD[i][j] = tcVal; TCMD[j][i] = tcVal;
            PCMD[i][j] = pcVal; PCMD[j][i] = pcVal;
        }
    }

    // Drive elyHanleyTransport via its native 5-parameter signature
    mixture_.updateTRANS
    (
        TcM, VcM*1e-3, MM, omegaM, ZcM,   // VcM -> m^3/kmol for elyHanley
        Yl, X, TCMD, PCMD, MMD_, SIGMD_
    );

    return mixture_;
}


// * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::SRKelyHanleyMixture<ThermoType>::SRKelyHanleyMixture
(
    const dictionary& dict
)
:
    multicomponentMixture<ThermoType>(dict),
    mixture_("mixture", this->specieThermos()[0]),
    numberOfSpecies_(this->specieThermos().size()),
    ListW_(numberOfSpecies_),
    ListTc_(numberOfSpecies_),
    ListPc_(numberOfSpecies_),
    ListVc_(numberOfSpecies_),
    ListOmega_(numberOfSpecies_),
    BM_(numberOfSpecies_),
    COEF1_(numberOfSpecies_),
    COEF2_(numberOfSpecies_),
    COEF3_(numberOfSpecies_),
    CM_(numberOfSpecies_, scalar(0)),
    CMq0_(numberOfSpecies_, scalar(0)),
    CMq1_(numberOfSpecies_, scalar(0)),
    CMq2_(numberOfSpecies_, scalar(0)),
    CMTlo_(numberOfSpecies_, scalar(0)),
    CMThi_(numberOfSpecies_, scalar(0)),
    KIJ_(numberOfSpecies_),
    LIJ_(numberOfSpecies_),
    BIJ_(numberOfSpecies_),
    VCIJ_(numberOfSpecies_),
    TVCIJ_(numberOfSpecies_),
    MMD_(numberOfSpecies_),
    SIGMD_(numberOfSpecies_)
{
    const scalar RR = Foam::constant::thermodynamic::RR;

    // Pure-species pulls
    forAll(BM_, i)
    {
        ListW_[i]     = this->specieThermos()[i].W();
        ListTc_[i]    = this->specieThermos()[i].Tc();
        ListPc_[i]    = this->specieThermos()[i].Pc();
        ListVc_[i]    = this->specieThermos()[i].Vc();
        ListOmega_[i] = this->specieThermos()[i].omega();
        CM_[i]        = this->specieThermos()[i].c();
        CMq0_[i]      = this->specieThermos()[i].cq0();
        CMq1_[i]      = this->specieThermos()[i].cq1();
        CMq2_[i]      = this->specieThermos()[i].cq2();
        CMTlo_[i]     = this->specieThermos()[i].cTlo();
        CMThi_[i]     = this->specieThermos()[i].cThi();
        BM_[i]        = 0.08664*RR
                       *this->specieThermos()[i].Tc()
                       /this->specieThermos()[i].Pc();
    }

    // k_ij and l_ij parsing. k_ij is shared with SRKchungTakaMixture
    // (energy interaction). l_ij is the Oefelein-2019 volume
    // interaction used by the Lorentz combining rule for b_ij and
    // V_c,ij; if absent in the dict it defaults to 0.
    forAll(KIJ_, i)
    {
        KIJ_[i].setSize(numberOfSpecies_, scalar(0));
        LIJ_[i].setSize(numberOfSpecies_, scalar(0));
    }
    if (dict.found("binaryInteraction"))
    {
        const dictionary& bd = dict.subDict("binaryInteraction");
        forAllConstIter(dictionary, bd, iter)
        {
            if (!iter().isDict()) continue;
            const word& pair = iter().keyword();
            if (pair == "default") continue;
            const std::size_t sep = pair.find('_');
            if (sep == std::string::npos) continue;
            const word a = pair.substr(0, sep);
            const word b = pair.substr(sep + 1);
            label ia = -1, ib = -1;
            forAll(this->specieThermos(), s)
            {
                if (this->specieThermos()[s].name() == a) ia = s;
                if (this->specieThermos()[s].name() == b) ib = s;
            }
            if (ia < 0 || ib < 0) continue;
            const scalar kij = iter().dict().lookupOrDefault<scalar>("kij", 0);
            const scalar lij = iter().dict().lookupOrDefault<scalar>("lij", 0);
            KIJ_[ia][ib] = kij;
            KIJ_[ib][ia] = kij;
            LIJ_[ia][ib] = lij;
            LIJ_[ib][ia] = lij;
        }
    }

    // Pre-compute SRK + diffusivity pair matrices, plus the Oefelein
    // 2019 quadratic combining matrices BIJ_, VCIJ_, TVCIJ_ used by
    // the b_m, V_c,m, T_c,m mixing rules.
    List<scalar> nCOEF1(numberOfSpecies_);
    List<scalar> nCOEF2(numberOfSpecies_);
    List<scalar> nCOEF3(numberOfSpecies_);
    List<scalar> nBIJ(numberOfSpecies_);
    List<scalar> nVCIJ(numberOfSpecies_);
    List<scalar> nTVCIJ(numberOfSpecies_);
    List<scalar> nMMD(numberOfSpecies_);
    List<scalar> nSIGMD(numberOfSpecies_);

    forAll(COEF1_, i)
    {
        const auto& si = this->specieThermos()[i];
        const scalar miOmega = si.omega();
        const scalar miTc    = si.Tc();
        const scalar miPc    = si.Pc();
        const scalar miVc    = si.Vc();      // cm^3/mol
        const scalar miW     = si.W();
        const scalar miSigmv = si.sigmvi();

        const scalar mFi = 0.48508 + 1.5517*miOmega - 0.15613*sqr(miOmega);
        const scalar Ai  = 0.42747*sqr(RR*miTc)/miPc;
        const scalar bi  = BM_[i];           // already computed above
        const scalar bi13 = pow(bi,  1.0/3.0);
        const scalar Vci13 = pow(miVc, 1.0/3.0);

        forAll(nCOEF1, j)
        {
            const auto& sj = this->specieThermos()[j];
            const scalar mjOmega = sj.omega();
            const scalar mjTc    = sj.Tc();
            const scalar mjPc    = sj.Pc();
            const scalar mjVc    = sj.Vc();
            const scalar mjW     = sj.W();
            const scalar mjSigmv = sj.sigmvi();

            const scalar mFj = 0.48508 + 1.5517*mjOmega - 0.15613*sqr(mjOmega);
            const scalar Aj  = 0.42747*sqr(RR*mjTc)/mjPc;
            const scalar bj  = BM_[j];
            const scalar bj13 = pow(bj,  1.0/3.0);
            const scalar Vcj13 = pow(mjVc, 1.0/3.0);

            const scalar oneMinusKij = (i == j) ? 1.0 : (1.0 - KIJ_[i][j]);
            const scalar oneMinusLij = (i == j) ? 1.0 : (1.0 - LIJ_[i][j]);
            const scalar sqAij = sqrt(Ai*Aj);

            nCOEF1[j] = sqAij*(1 + mFi)*(1 + mFj)*oneMinusKij;
            nCOEF2[j] =
                sqAij
               *(
                    (1.0 + mFj)*mFi/sqrt(miTc)
                  + (1.0 + mFi)*mFj/sqrt(mjTc)
                )*oneMinusKij;
            nCOEF3[j] = sqAij*mFi*mFj/sqrt(miTc*mjTc)*oneMinusKij;

            // Lorentz combining rule for SRK b_ij and ECS V_c,ij
            // (Oefelein 2019 Eqs. 16, 25).
            const scalar bSum3  = pow3(bi13   + bj13);
            const scalar VcSum3 = pow3(Vci13  + Vcj13);
            nBIJ[j]  = 0.125*bSum3 *oneMinusLij;
            nVCIJ[j] = 0.125*VcSum3*oneMinusLij;
            // Eq. 17 cross term sqrt(T_ci T_cj)(1-k_ij), weighted by
            // V_c,ij so that T_c,m = (sum X_i X_j TVCIJ) / V_c,m.
            nTVCIJ[j] = sqrt(miTc*mjTc)*oneMinusKij*nVCIJ[j];

            nMMD[j]  = 1/miW + 1/mjW;
            nSIGMD[j] = pow(miSigmv, 1.0/3) + pow(mjSigmv, 1.0/3);
        }

        COEF1_[i] = nCOEF1;
        COEF2_[i] = nCOEF2;
        COEF3_[i] = nCOEF3;
        BIJ_[i]   = nBIJ;
        VCIJ_[i]  = nVCIJ;
        TVCIJ_[i] = nTVCIJ;
        MMD_[i]   = nMMD;
        SIGMD_[i] = nSIGMD;
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ThermoType>
const typename Foam::SRKelyHanleyMixture<ThermoType>::thermoMixtureType&
Foam::SRKelyHanleyMixture<ThermoType>::thermoMixture
(
    const scalarFieldListSlice& Y
) const
{
    return calcMixture(Y);
}


template<class ThermoType>
const typename Foam::SRKelyHanleyMixture<ThermoType>::transportMixtureType&
Foam::SRKelyHanleyMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice& Y
) const
{
    return thermoMixture(Y);
}


template<class ThermoType>
const typename Foam::SRKelyHanleyMixture<ThermoType>::transportMixtureType&
Foam::SRKelyHanleyMixture<ThermoType>::transportMixture
(
    const scalarFieldListSlice&,
    const thermoMixtureType& mixture
) const
{
    return mixture;
}


// ************************************************************************* //
