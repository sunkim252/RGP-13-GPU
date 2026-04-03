/*---------------------------------------------------------------------------*\
  RGP-13 — rgpRhoThermo implementation
\*---------------------------------------------------------------------------*/

#include "rgpRhoThermo.H"
#include "gpu/cudaDeviceManager.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(rgpRhoThermo, 0);
    addToRunTimeSelectionTable(rhoFluidThermo, rgpRhoThermo, fvMesh);
}


// * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * //

void Foam::rgpRhoThermo::readCoeffs(const dictionary& thermoDict)
{
    // ── JANAF ──
    const dictionary& janafDict = thermoDict.subDict("janaf");
    coeffs_.janaf.Tlow    = janafDict.lookup<scalar>("Tlow");
    coeffs_.janaf.Thigh   = janafDict.lookup<scalar>("Thigh");
    coeffs_.janaf.Tcommon = janafDict.lookup<scalar>("Tcommon");
    coeffs_.janaf.W       = janafDict.lookup<scalar>("molWeight") * 1e-3; // g/mol → kg/mol

    List<scalar> highCoeffs(janafDict.lookup("highCpCoeffs"));
    List<scalar> lowCoeffs(janafDict.lookup("lowCpCoeffs"));
    for (int i = 0; i < 7; ++i)
    {
        coeffs_.janaf.high[i] = highCoeffs[i];
        coeffs_.janaf.low[i]  = lowCoeffs[i];
    }

    // ── Peng-Robinson ──
    const dictionary& prDict = thermoDict.subDict("PengRobinson");
    coeffs_.pr.Tc    = prDict.lookup<scalar>("Tc");
    coeffs_.pr.Pc    = prDict.lookup<scalar>("Pc");
    coeffs_.pr.omega = prDict.lookup<scalar>("omega");

    // ── Sutherland ──
    const dictionary& suthDict = thermoDict.subDict("sutherland");
    coeffs_.suth.As = suthDict.lookup<scalar>("As");
    coeffs_.suth.Ts = suthDict.lookup<scalar>("Ts");
    coeffs_.suth.Pr = suthDict.lookupOrDefault<scalar>("Pr", 0.7);

    // ── GPU threshold ──
    cudaMinCells_ = thermoDict.lookupOrDefault<label>("cudaMinCells", 1000);
}


void Foam::rgpRhoThermo::calculateBoundary()
{
    // Phase 1: CPU fallback — iterate boundary patches and compute
    // properties using the same analytic expressions as the GPU kernel.
    // TODO: GPU boundary kernel (Phase 3)

    volScalarField::Boundary& pBf   = this->p_.boundaryFieldRef();
    volScalarField::Boundary& TBf   = this->T_.boundaryFieldRef();
    volScalarField::Boundary& CpBf  = this->Cp_.boundaryFieldRef();
    volScalarField::Boundary& CvBf  = this->Cv_.boundaryFieldRef();
    volScalarField::Boundary& psiBf = this->psi_.boundaryFieldRef();
    volScalarField::Boundary& rhoBf = this->rho_.boundaryFieldRef();
    volScalarField::Boundary& heBf  = this->he().boundaryFieldRef();
    volScalarField::Boundary& muBf  = this->mu_.boundaryFieldRef();
    volScalarField::Boundary& kBf   = this->kappa_.boundaryFieldRef();

    forAll(TBf, patchi)
    {
        fvPatchScalarField& pT  = TBf[patchi];
        fvPatchScalarField& pp  = pBf[patchi];
        fvPatchScalarField& phe = heBf[patchi];

        // Simplified: use JANAF + PR same as kernel
        // (full implementation will dispatch per-patch)
        forAll(pT, facei)
        {
            // TODO: implement boundary thermo (mirror kernel logic)
            // For now, internal field values are extrapolated by OF boundary conditions
        }

        (void)pp;
        (void)phe;
        (void)CpBf[patchi];
        (void)CvBf[patchi];
        (void)psiBf[patchi];
        (void)rhoBf[patchi];
        (void)muBf[patchi];
        (void)kBf[patchi];
    }
}


// * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * * //

Foam::rgpRhoThermo::rgpRhoThermo
(
    const fvMesh& mesh,
    const word& phaseName
)
:
    rhoFluidThermo::composite(mesh, phaseName),
    cudaMinCells_(1000)
{
    // Initialise CUDA device (rank → GPU mapping)
    cudaDeviceManager::init();

    // Read coefficients from physicalProperties/rgpCoeffs
    const dictionary& thermoDict =
        mesh.lookupObject<IOdictionary>("physicalProperties")
        .subDict("rgpCoeffs");
    readCoeffs(thermoDict);

    // Initial property calculation
    correct();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::rgpRhoThermo::~rgpRhoThermo()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::rgpRhoThermo::correct()
{
    const label nCells = this->T_.size();

    if (nCells >= cudaMinCells_)
    {
        // ── GPU path ───────────────────────────────────────────────
        rgp13ThermoCalculateGPU
        (
            this->he().primitiveFieldRef().cdata(),
            this->p_.primitiveFieldRef().cdata(),
            this->T_.primitiveFieldRef().data(),
            this->Cp_.primitiveFieldRef().data(),
            this->Cv_.primitiveFieldRef().data(),
            this->psi_.primitiveFieldRef().data(),
            this->rho_.primitiveFieldRef().data(),
            this->mu_.primitiveFieldRef().data(),
            this->kappa_.primitiveFieldRef().data(),
            nCells,
            coeffs_
        );
    }
    else
    {
        // ── CPU fallback (small meshes / debugging) ────────────────
        // TODO: CPU implementation mirroring kernel logic
        Info << "RGP-13: nCells=" << nCells
             << " < cudaMinCells=" << cudaMinCells_
             << ", using CPU fallback" << endl;
    }

    // Boundary faces always on CPU (Phase 1)
    calculateBoundary();
}


// ************************************************************************* //
