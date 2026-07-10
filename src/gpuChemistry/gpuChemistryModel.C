/*---------------------------------------------------------------------------*\
  gpuChemistryModel — 구현. 배치 계약은 chemistryModel::solve와 동일:
  oldTime (rho, T, p, Y)에서 셀별 [0, deltaT] 적분 후
      RR_i = (c_i,new - c_i,0) * W_i / deltaT   [kg/m^3/s]
\*---------------------------------------------------------------------------*/

#include "gpuChemistryModel.H"
#include "UniformField.H"

#include "gpu/rgpKernelTypes.H"

#include "gpu/rgpChemRHS.H"
#include "OStringStream.H"
#include "IStringStream.H"
#include "Tuple2.H"
#include "thermodynamicConstants.H"
#include <cstring>

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class ThermoType>
Foam::gpuChemistryModel<ThermoType>::gpuChemistryModel
(
    const fluidMulticomponentThermo& thermo
)
:
    chemistryModel<ThermoType>(thermo),
    relTol_(1e-5),
    absTol_(1e-12),
    retrieveTol_(0),
    armed_(false)
{
    if (this->found("gpuCoeffs"))
    {
        const dictionary& dict = this->subDict("gpuCoeffs");
        relTol_ = dict.lookupOrDefault<scalar>("relTol", 1e-5);
        absTol_ = dict.lookupOrDefault<scalar>("absTol", 1e-12);

        // warp 균형: off/auto/always (기본 auto — 불균형일 때만 발동,
        // 게더/스캐터는 값 보존이라 결과 비트-동일)
        const word bal
        (
            dict.lookupOrDefault<word>("balance", "auto")
        );
        rgpChemSetBalance(bal == "off" ? 0 : bal == "always" ? 2 : 1);

        // retrieve 캐시 (opt-in, 정확도-성능 트레이드): 입력 미변화
        // 셀의 재적분 스킵. 0 = off.
        retrieveTol_ = dict.lookupOrDefault<scalar>("retrieveTol", 0);
        if (retrieveTol_ > 0)
        {
            rgpChemSetCacheTol(retrieveTol_);
            Info<< "gpuChemistry: retrieve cache on, tol = "
                << retrieveTol_ << endl;
        }
    }
}


template<class ThermoType>
Foam::gpuChemistryModel<ThermoType>::~gpuChemistryModel()
{}


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

template<class ThermoType>
void Foam::gpuChemistryModel<ThermoType>::armGpu() const
{
    // ── 메커니즘 export: base chemistryModel이 파싱한 종/반응을 POD로 ──
    // 반응 파라미터는 Reaction::write() 라운드트립(dict)에서 추출 —
    // 타입별 멤버 접근 없이 모든 등록 타입을 일반 처리.
    memset(&mech_, 0, sizeof(mech_));

    const PtrList<ThermoType>& sTh = this->specieThermos();
    const label ns = sTh.size();
    if (ns > RGP_CHEM_MAX_SPECIES)
    {
        FatalErrorInFunction
            << "gpu chemistry supports up to " << RGP_CHEM_MAX_SPECIES
            << " species; case has " << ns << exit(FatalError);
    }
    mech_.nSpecies = ns;
    mech_.RR = constant::thermodynamic::RR;
    // OF의 Kc 기준압 (etc/controlDict Pstd, 기본 1e5 Pa — CHEMKIN 101325와
    // 다름). 솔버 경로는 OF와의 정합이 우선이므로 OF 값을 따른다.
    mech_.pRef = constant::thermodynamic::Pstd;

    forAll(sTh, i)
    {
        const scalar W = sTh[i].W();
        mech_.W[i] = W;
        mech_.Tcommon[i] = sTh[i].Tcommon();
        // OF janaf 계수는 질량기준(x R_specific) 저장 — 커널은 무차원
        // NASA7을 기대하므로 역변환.
        const scalar Rinv = W/constant::thermodynamic::RR;
        const auto& hc = sTh[i].highCpCoeffs();
        const auto& lc = sTh[i].lowCpCoeffs();
        for (label c = 0; c < 7; c++)
        {
            mech_.janafHigh[i][c] = hc[c]*Rinv;
            mech_.janafLow[i][c] = lc[c]*Rinv;
        }
    }

    const PtrList<Reaction<ThermoType>>& rxns = this->reactions();
    if (rxns.size() > RGP_CHEM_MAX_REACTIONS)
    {
        FatalErrorInFunction
            << "gpu chemistry supports up to " << RGP_CHEM_MAX_REACTIONS
            << " reactions; mechanism has " << rxns.size()
            << exit(FatalError);
    }
    mech_.nReactions = rxns.size();

    const speciesTable& species = this->thermo().species();

    forAll(rxns, r)
    {
        const Reaction<ThermoType>& rx = rxns[r];

        // 양론 (elementary 가정: exponent == stoichCoeff)
        forAll(rx.lhs(), k)
        {
            mech_.nuL[r][rx.lhs()[k].index] += rx.lhs()[k].stoichCoeff;
        }
        forAll(rx.rhs(), k)
        {
            mech_.nuR[r][rx.rhs()[k].index] += rx.rhs()[k].stoichCoeff;
        }

        const word typ = rx.type();
        mech_.reversible[r] =
            typ.find("irreversible") == string::npos ? 1 : 0;

        // dict 라운드트립
        OStringStream buf;
        rx.write(buf);
        IStringStream is(buf.str());
        dictionary d(is);

        auto readArr = [&](const dictionary& ad, double& A, double& b,
                           double& Ta)
        {
            A = ad.lookup<scalar>("A");
            b = ad.lookup<scalar>("beta");
            Ta = ad.lookup<scalar>("Ta");
        };
        auto readEff = [&](const dictionary& ed)
        {
            for (int s = 0; s < ns; s++) { mech_.eff[r][s] = 1.0; }
            List<Tuple2<word, scalar>> ce(ed.lookup("coeffs"));
            forAll(ce, k)
            {
                const label si = species[ce[k].first()];
                mech_.eff[r][si] = ce[k].second();
            }
        };

        if (d.found("k0"))                       // 폴오프
        {
            readArr(d.subDict("kInf"), mech_.A[r], mech_.beta[r], mech_.Ta[r]);
            readArr(d.subDict("k0"), mech_.A0[r], mech_.beta0[r], mech_.Ta0[r]);
            if (d.found("F") && d.subDict("F").found("alpha"))
            {
                const dictionary& F = d.subDict("F");
                mech_.tbType[r] = RGP_TB_TROE;
                mech_.troe[r][0] = F.lookup<scalar>("alpha");
                mech_.troe[r][1] = F.lookup<scalar>("Tsss");
                mech_.troe[r][2] = F.lookup<scalar>("Ts");
                mech_.troe[r][3] = F.lookupOrDefault<scalar>("Tss", 0);
            }
            else
            {
                mech_.tbType[r] = RGP_TB_LIND;
            }
            readEff(d.subDict("thirdBodyEfficiencies"));
        }
        else if (d.found("coeffs"))              // +M 3체
        {
            mech_.tbType[r] = RGP_TB_M;
            readArr(d, mech_.A[r], mech_.beta[r], mech_.Ta[r]);
            readEff(d);
        }
        else                                     // 일반 Arrhenius
        {
            mech_.tbType[r] = RGP_TB_NONE;
            readArr(d, mech_.A[r], mech_.beta[r], mech_.Ta[r]);
        }
    }

    // 항등 매핑 (mech 종 순서 == 케이스 종 순서)
    map_.setSize(ns);
    forAll(map_, i) { map_[i] = i; }

    const int nDev = max(rgpGpuDeviceCount(), 1);
    if (rgpChemInit(Pstream::parRun() ? (Pstream::myProcNo() % nDev) : -1))
    {
        FatalErrorInFunction
            << "rgpChemInit: " << rgpChemLastError() << exit(FatalError);
    }
    if (rgpChemUpload(&mech_))
    {
        FatalErrorInFunction
            << "rgpChemUpload: " << rgpChemLastError() << exit(FatalError);
    }

    Info<< "gpuChemistryModel: ARMED — " << ns << " species / "
        << mech_.nReactions << " reactions exported from the case "
        << "mechanism (pRef = " << mech_.pRef << " Pa); relTol "
        << relTol_ << ", absTol " << absTol_ << nl << endl;

    armed_ = true;
}


template<class ThermoType>
template<class DeltaTType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::solveBatch
(
    const DeltaTType& deltaT
)
{
    if (!this->chemistry_)
    {
        return great;
    }

    if (!armed_) { armGpu(); }

    const label nc = this->mesh().nCells();
    const int n = mech_.nSpecies;

    const volScalarField& rho0vf =
        this->mesh().template lookupObject<volScalarField>
        (
            this->thermo().phasePropertyName("rho")
        ).oldTime();
    const scalarField& T0 = this->thermo().T().oldTime();
    const scalarField& p0 = this->thermo().p().oldTime();
    const PtrList<volScalarField>& Yvf = this->thermo().Y();

    if (pB_.size() < nc)
    {
        pB_.setSize(nc); TB_.setSize(nc);
        cB_.setSize(nc*n); dtB_.setSize(nc);
    }

    for (label celli = 0; celli < nc; celli++)
    {
        pB_[celli] = p0[celli];
        TB_[celli] = T0[celli];
        dtB_[celli] = deltaT[celli];
    }
    for (int m = 0; m < n; m++)
    {
        const scalarField& Ym = Yvf[map_[m]].oldTime();
        const double Winv = 1.0/mech_.W[m];
        for (label celli = 0; celli < nc; celli++)
        {
            cB_[celli*n + m] = rho0vf[celli]*max(Ym[celli], scalar(0))*Winv;
        }
    }

    long long stats[2] = {0, 0};
    if (rgpChemIntegrate
        (
            nc, dtB_.begin(), relTol_, absTol_,
            pB_.begin(), TB_.begin(), cB_.begin(), stats
        ))
    {
        FatalErrorInFunction
            << "rgpChemIntegrate: " << rgpChemLastError() << exit(FatalError);
    }

    if (retrieveTol_ > 0)
    {
        Info<< "gpuChemistry: retrieve hits " << label(rgpChemCacheHits())
            << " / " << nc << endl;
    }

    // RR_i = (c_new - c_0) W_i / deltaT
    PtrList<volScalarField::Internal>& RRw =
        const_cast<PtrList<volScalarField::Internal>&>(this->RR());

    for (int m = 0; m < n; m++)
    {
        const scalarField& Ym = Yvf[map_[m]].oldTime();
        const double W = mech_.W[m];
        const double Winv = 1.0/W;
        scalarField& RRm = RRw[map_[m]];
        for (label celli = 0; celli < nc; celli++)
        {
            const double c0 = rho0vf[celli]*max(Ym[celli], scalar(0))*Winv;
            RRm[celli] = (cB_[celli*n + m] - c0)*W/dtB_[celli];
        }
    }

    return great;
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::solve(const scalar deltaT)
{
    return solveBatch(UniformField<scalar>(deltaT));
}


template<class ThermoType>
Foam::scalar Foam::gpuChemistryModel<ThermoType>::solve
(
    const scalarField& deltaT
)
{
    return solveBatch(deltaT);
}


template<class ThermoType>
void Foam::gpuChemistryModel<ThermoType>::solve
(
    scalar& p,
    scalar& T,
    scalarField& Y,
    const label li,
    scalar& deltaT,
    scalar& subDeltaT
) const
{
    // 호스트 폴백: 같은 rosenbrock34Step으로 1셀 적분 (Y 공간 왕복)
    if (!armed_) { armGpu(); }

    const int n = mech_.nSpecies;
    const int neq = n + 1;

    // Y -> c (이상기체 rho)
    double sumYoW = 0;
    for (int m = 0; m < n; m++)
    {
        sumYoW += max(Y[map_[m]], scalar(0))/mech_.W[m];
    }
    const double rho = p/(mech_.RR*max(T, small)*max(sumYoW, vSmall));

    List<double> y0(neq), y1(neq), wk(neq*(neq + 8));
    for (int m = 0; m < n; m++)
    {
        y0[m] = rho*max(Y[map_[m]], scalar(0))/mech_.W[m];
    }
    y0[n] = T;

    scalar t = 0, h = min(subDeltaT, deltaT);
    while (t < deltaT)
    {
        if (h > deltaT - t) { h = deltaT - t; }
        const double err = rgpchem::rosenbrock34Step
        (
            mech_, p, h, relTol_, absTol_, y0.begin(), y1.begin(), wk.begin()
        );
        if (err <= 1.0 && y1[n] > 0 && !std::isnan(y1[n]))
        {
            t += h;
            for (int i = 0; i < neq; i++) { y0[i] = y1[i]; }
            h *= min(max(0.9*pow(err, -1.0/3.0), scalar(0.2)), scalar(5.0));
        }
        else
        {
            h *= std::isnan(err) ? 0.25
                : min(max(0.9*pow(err, -1.0/3.0), scalar(0.1)), scalar(0.5));
            if (h < vSmall)
            {
                FatalErrorInFunction
                    << "step underflow" << exit(FatalError);
            }
        }
    }
    subDeltaT = h;

    // c -> Y
    double rhoNew = 0;
    for (int m = 0; m < n; m++) { rhoNew += y0[m]*mech_.W[m]; }
    for (int m = 0; m < n; m++)
    {
        Y[map_[m]] = y0[m]*mech_.W[m]/max(rhoNew, vSmall);
    }
    T = y0[n];
}


// ************************************************************************* //
