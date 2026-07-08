/*---------------------------------------------------------------------------*\
  gpuChemistryModel — 구현. 배치 계약은 chemistryModel::solve와 동일:
  oldTime (rho, T, p, Y)에서 셀별 [0, deltaT] 적분 후
      RR_i = (c_i,new - c_i,0) * W_i / deltaT   [kg/m^3/s]
\*---------------------------------------------------------------------------*/

#include "gpuChemistryModel.H"
#include "UniformField.H"

#include "gpu/rgpKernelTypes.H"

#include "gpu/rgpChemRHS.H"
#include "gpu/rgpH2O2Burke.H"

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
    armed_(false)
{
    if (this->found("gpuCoeffs"))
    {
        const dictionary& dict = this->subDict("gpuCoeffs");
        relTol_ = dict.lookupOrDefault<scalar>("relTol", 1e-5);
        absTol_ = dict.lookupOrDefault<scalar>("absTol", 1e-12);
    }
}


template<class ThermoType>
Foam::gpuChemistryModel<ThermoType>::~gpuChemistryModel()
{}


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

template<class ThermoType>
void Foam::gpuChemistryModel<ThermoType>::armGpu() const
{
    static const char* mechNames[nSp9] =
        {"H2", "O2", "H", "O", "OH", "HO2", "H2O2", "H2O", "N2"};

    rgpBuildBurke2012(mech_);

    const speciesTable& species = this->thermo().species();
    if (species.size() != mech_.nSpecies)
    {
        FatalErrorInFunction
            << "gpu chemistry (Burke 2012 H2/O2) requires exactly "
            << mech_.nSpecies << " species; case has " << species.size()
            << exit(FatalError);
    }

    map_.setSize(mech_.nSpecies, -1);
    for (int m = 0; m < mech_.nSpecies; m++)
    {
        forAll(species, i)
        {
            if (species[i] == word(mechNames[m])) { map_[m] = i; break; }
        }
        if (map_[m] < 0)
        {
            FatalErrorInFunction
                << "specie " << mechNames[m]
                << " (Burke 2012 H2/O2) not found in the case species "
                << species << exit(FatalError);
        }
    }

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

    Info<< "gpuChemistryModel: ARMED — Burke 2012 H2/O2 ("
        << mech_.nSpecies << " sp / " << mech_.nReactions
        << " rev. reactions) on the CUDA device; relTol " << relTol_
        << ", absTol " << absTol_ << nl << endl;

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
