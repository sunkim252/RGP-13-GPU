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
#include "PstreamBuffers.H"
#include "PstreamReduceOps.H"
#include <cstring>
#include <algorithm>

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
    dlb_(true),
    dlbThreshold_(1.3),
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

        // 랭크 간 부하균형 (병렬 전용; 화염면 편중 완화)
        dlb_ = dict.lookupOrDefault<Switch>("dlb", true);
        dlbThreshold_ =
            dict.lookupOrDefault<scalar>("dlbThreshold", 1.3);
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

    // ── 랭크 간 부하균형(DLB): 직전 스텝 substep 비용으로 과부하
    //    랭크의 최고비용 셀을 저부하 랭크로 이송. 셀별 적분은 독립·
    //    결정론 — 어느 랭크에서 계산돼도 결과 비트-동일. 매칭은 모든
    //    랭크가 allgather된 부하로 동일하게 계산(결정론 두-포인터) ──
    const bool doDlb = dlb_ && Pstream::parRun() && nc > 0;
    labelList expIds;                 // 내보낸 로컬 셀 (전송 순서 연접)
    DynamicList<label> sendTo, sendCnt;
    DynamicList<label> recvFromRank;
    labelList retained;               // 잔류 로컬 셀 id (배치 앞부분)
    label nImp = 0;                   // 수입 셀 수 (배치 뒷부분)
    List<label> impFromCnt;           // recv별 수입 셀 수
    bool dlbActive = false;

    if (doDlb)
    {
        if (dlbCost_.size() != nc)
        {
            dlbCost_.setSize(nc);
            forAll(dlbCost_, i) { dlbCost_[i] = 1; }
        }
        scalar myLoad = 0;
        forAll(dlbCost_, i) { myLoad += scalar(dlbCost_[i]); }
        List<scalar> loads(Pstream::nProcs(), scalar(0));
        loads[Pstream::myProcNo()] = myLoad;
        Pstream::gatherList(loads);
        Pstream::scatterList(loads);
        scalar total = 0;
        scalar maxLoad = 0;
        forAll(loads, r)
        {
            total += loads[r];
            maxLoad = max(maxLoad, loads[r]);
        }
        const scalar avg = total/Pstream::nProcs();

        // 텔레메트리: 발동 여부와 무관하게 한 줄 (부하비 관찰용)
        Info<< "gpuChemistry DLB: max/avg = "
            << (avg > small ? maxLoad/avg : 1.0)
            << " (thr " << dlbThreshold_ << ")" << endl;

        if (maxLoad > dlbThreshold_*avg + small)
        {
            dlbActive = true;

            // 결정론 두-포인터 매칭: 잉여(donor) → 결핍(receiver)
            List<scalar> excess(loads.size());
            forAll(loads, r) { excess[r] = loads[r] - avg; }
            const scalar minAmt = 0.02*avg + small;
            label d = 0, rv = 0;
            const label nP = Pstream::nProcs();
            const label me = Pstream::myProcNo();
            DynamicList<scalar> myTgtCost;
            while (true)
            {
                while (d < nP && excess[d] <= minAmt) { d++; }
                while (rv < nP && excess[rv] >= -minAmt) { rv++; }
                if (d >= nP || rv >= nP) { break; }
                const scalar amt = min(excess[d], -excess[rv]);
                if (amt >= minAmt)
                {
                    if (d == me) { sendTo.append(rv); myTgtCost.append(amt); }
                    if (rv == me) { recvFromRank.append(d); }
                }
                excess[d] -= amt;
                excess[rv] += amt;
            }

            // donor: 비용 내림차순으로 셀 배정 (stable — 결정론)
            boolList exported(nc, false);
            if (sendTo.size())
            {
                labelList order(nc);
                forAll(order, i) { order[i] = i; }
                std::stable_sort
                (
                    order.begin(), order.end(),
                    [&](label a, label b)
                    {
                        return dlbCost_[a] > dlbCost_[b];
                    }
                );
                DynamicList<label> ids(nc);
                sendCnt.setSize(sendTo.size());
                label cur = 0;
                forAll(sendTo, t)
                {
                    scalar acc = 0;
                    label cnt = 0;
                    while (cur < nc - 1 && acc < myTgtCost[t])
                    {
                        const label id = order[cur++];
                        ids.append(id);
                        exported[id] = true;
                        acc += scalar(dlbCost_[id]);
                        cnt++;
                    }
                    sendCnt[t] = cnt;
                }
                expIds.transfer(ids);
            }

            // 전송: [p,T,dt,c*n]/셀 — 수신은 매칭 순서로 처리(결정론)
            PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
            {
                label off = 0;
                forAll(sendTo, t)
                {
                    List<scalar> pay(sendCnt[t]*(n + 3));
                    for (label k = 0; k < sendCnt[t]; k++)
                    {
                        const label id = expIds[off + k];
                        scalar* q = pay.begin() + k*(n + 3);
                        q[0] = pB_[id];
                        q[1] = TB_[id];
                        q[2] = dtB_[id];
                        for (int m = 0; m < n; m++)
                        {
                            q[3 + m] = cB_[id*n + m];
                        }
                    }
                    UOPstream toR(sendTo[t], pBufs);
                    toR << pay;
                    off += sendCnt[t];
                }
            }
            pBufs.finishedSends();

            // 잔류 로컬 셀
            {
                DynamicList<label> ret(nc);
                for (label i = 0; i < nc; i++)
                {
                    if (!exported[i]) { ret.append(i); }
                }
                retained.transfer(ret);
            }

            // 수입 페이로드 수신 → 컴팩트 배치 [retained | imported]
            List<List<scalar>> impPay(recvFromRank.size());
            impFromCnt.setSize(recvFromRank.size(), 0);
            forAll(recvFromRank, t)
            {
                UIPstream fromR(recvFromRank[t], pBufs);
                fromR >> impPay[t];
                impFromCnt[t] = impPay[t].size()/(n + 3);
                nImp += impFromCnt[t];
            }

            const label nBatch = retained.size() + nImp;
            if (pC_.size() < nBatch)
            {
                pC_.setSize(nBatch); TC_.setSize(nBatch);
                dtC_.setSize(nBatch); cC_.setSize(nBatch*n);
            }
            forAll(retained, k)
            {
                const label id = retained[k];
                pC_[k] = pB_[id];
                TC_[k] = TB_[id];
                dtC_[k] = dtB_[id];
                for (int m = 0; m < n; m++)
                {
                    cC_[k*n + m] = cB_[id*n + m];
                }
            }
            label k = retained.size();
            forAll(recvFromRank, t)
            {
                const scalar* q0 = impPay[t].begin();
                for (label j = 0; j < impFromCnt[t]; j++)
                {
                    const scalar* q = q0 + j*(n + 3);
                    pC_[k] = q[0];
                    TC_[k] = q[1];
                    dtC_[k] = q[2];
                    for (int m = 0; m < n; m++)
                    {
                        cC_[k*n + m] = q[3 + m];
                    }
                    k++;
                }
            }

            Info<< "gpuChemistry DLB: load " << label(myLoad)
                << " (avg " << label(avg) << ", max " << label(maxLoad)
                << ") -- exported " << expIds.size()
                << ", imported " << nImp << " cells" << endl;
        }
    }

    long long stats[2] = {0, 0};
    if (!dlbActive)
    {
        if (rgpChemIntegrate
            (
                nc, dtB_.begin(), relTol_, absTol_,
                pB_.begin(), TB_.begin(), cB_.begin(), stats
            ))
        {
            FatalErrorInFunction
                << "rgpChemIntegrate: " << rgpChemLastError()
                << exit(FatalError);
        }
        if (doDlb)
        {
            dlbCost_.setSize(nc);
            if (rgpChemLastSteps(dlbCost_.begin(), nc))
            {
                forAll(dlbCost_, i) { dlbCost_[i] = 1; }
            }
        }
    }
    else
    {
        const label nBatch = retained.size() + nImp;
        if (nBatch > 0 && rgpChemIntegrate
            (
                nBatch, dtC_.begin(), relTol_, absTol_,
                pC_.begin(), TC_.begin(), cC_.begin(), stats
            ))
        {
            FatalErrorInFunction
                << "rgpChemIntegrate: " << rgpChemLastError()
                << exit(FatalError);
        }

        List<long long> steps(max(nBatch, label(1)), (long long)1);
        if (nBatch > 0) { rgpChemLastSteps(steps.begin(), nBatch); }

        // 잔류 셀 결과·비용을 원래 레이아웃으로 산포
        dlbCost_.setSize(nc);
        forAll(retained, k)
        {
            const label id = retained[k];
            TB_[id] = TC_[k];
            for (int m = 0; m < n; m++)
            {
                cB_[id*n + m] = cC_[k*n + m];
            }
            dlbCost_[id] = steps[k];
        }

        // 수입 셀 결과 반송: [c*n, steps]/셀 (수신 순서 보존)
        PstreamBuffers rBufs(Pstream::commsTypes::nonBlocking);
        {
            label k = retained.size();
            forAll(recvFromRank, t)
            {
                List<scalar> pay(impFromCnt[t]*(n + 1));
                for (label j = 0; j < impFromCnt[t]; j++)
                {
                    scalar* q = pay.begin() + j*(n + 1);
                    for (int m = 0; m < n; m++)
                    {
                        q[m] = cC_[k*n + m];
                    }
                    q[n] = scalar(steps[k]);
                    k++;
                }
                UOPstream toR(recvFromRank[t], rBufs);
                toR << pay;
            }
        }
        rBufs.finishedSends();
        {
            label off = 0;
            forAll(sendTo, t)
            {
                List<scalar> pay;
                UIPstream fromR(sendTo[t], rBufs);
                fromR >> pay;
                for (label j = 0; j < sendCnt[t]; j++)
                {
                    const label id = expIds[off + j];
                    const scalar* q = pay.begin() + j*(n + 1);
                    for (int m = 0; m < n; m++)
                    {
                        cB_[id*n + m] = q[m];
                    }
                    dlbCost_[id] = (long long)q[n];
                }
                off += sendCnt[t];
            }
        }
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
