/*---------------------------------------------------------------------------*\
  gpuMulticomponentFluid — 생성자/아밍/가드 (개요는 헤더 참조)
\*---------------------------------------------------------------------------*/

#include "gpuMulticomponentFluid.H"
#include "addToRunTimeSelectionTable.H"
#include "fvConstraint.H"
#include "gpu/rgpPEqnTypes.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace solvers
{
    defineTypeNameAndDebug(gpuMulticomponentFluid, 0);
    addToRunTimeSelectionTable(solver, gpuMulticomponentFluid, fvMesh);
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::solvers::gpuMulticomponentFluid::gpuMulticomponentFluid(fvMesh& mesh)
:
    multicomponentFluid(mesh),
    gpuYEqn_(false),
    gpuEEqn_(false),
    gpuCheck_(false),
    gpuMeshArmed_(false),
    gpuMeshCells_(-1),
    gpuMeshFaces_(-1),
    gpuMeshStampTime_(-1)
{
    {
        const IOdictionary dict
        (
            IOobject
            (
                "gpuProperties",
                runTime.constant(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            )
        );
        gpuYEqn_ = dict.lookupOrDefault<Switch>("gpuYEqn", false);
        gpuEEqn_ = dict.lookupOrDefault<Switch>("gpuEEqn", false);
        gpuCheck_ = dict.lookupOrDefault<Switch>("gpuCheck", false);
    }

    if (Pstream::parRun() && (gpuYEqn_ || gpuEEqn_))
    {
        WarningInFunction
            << "gpuYEqn/gpuEEqn are serial-only (v1) -- disabled for "
            << "this parallel run (GPU chemistry stays active)" << endl;
        gpuYEqn_ = Switch(false);
        gpuEEqn_ = Switch(false);
    }
    if (gpuEEqn_ && !gpuYEqn_)
    {
        FatalErrorInFunction
            << "gpuEEqn requires gpuYEqn (shared multivariate weights)"
            << exit(FatalError);
    }

    if (gpuYEqn_)
    {
        // v1: divj/divq가 순수 laplacian인 수송 모델만 (Fickian류의
        // 추가 명시 보정항은 미반영 — 침묵 오차 방지)
        const word ttType(thermophysicalTransport->type());
        if
        (
            ttType.find("unityLewis") == std::string::npos
         && ttType.find("Fourier") == std::string::npos
        )
        {
            FatalErrorInFunction
                << "gpuYEqn (v1) supports unityLewis*/Fourier "
                << "thermophysical transport only (divj must be a pure "
                << "laplacian); active model: " << ttType
                << exit(FatalError);
        }
        if (thermo.he().name() != "h" && gpuEEqn_)
        {
            FatalErrorInFunction
                << "gpuEEqn (v1) supports sensible enthalpy (he == h) "
                << "only" << exit(FatalError);
        }

        Info<< "gpuMulticomponentFluid: GPU transport ACTIVE -- "
            << "Y batch" << (gpuEEqn_ ? " + EEqn" : "")
            << " assembled and solved on the device" << nl << endl;
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::solvers::gpuMulticomponentFluid::~gpuMulticomponentFluid()
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::gpuMulticomponentFluid::armGpuMesh()
{
    const label nc = mesh.nCells();
    const label nif = mesh.owner().size();

    // 토폴로지/이동 감지 재아밍 (fgmFluid::armGpuPEqnMesh와 동일 래치)
    if (gpuMeshArmed_)
    {
        const bool flagged =
            (mesh.topoChanged() || mesh.moving())
         && mesh.time().timeIndex() != gpuMeshStampTime_;

        if (nc == gpuMeshCells_ && nif == gpuMeshFaces_ && !flagged)
        {
            return;
        }
        Info<< "gpuMulticomponentFluid: mesh changed -- re-arming"
            << nl << endl;
    }

    label nbf = 0;
    forAll(thermo.T().boundaryField(), patchi)
    {
        if (thermo.T().boundaryField()[patchi].coupled())
        {
            FatalErrorInFunction
                << "gpuYEqn (v1) does not support coupled patches"
                << exit(FatalError);
        }
        nbf += thermo.T().boundaryField()[patchi].size();
    }

    List<int> own(nif), nei(nif);
    forAll(mesh.owner(), f)
    {
        own[f] = mesh.owner()[f];
        nei[f] = mesh.neighbour()[f];
    }
    List<double> gg(nif);
    const scalarField& magSf = mesh.magSf().primitiveField();
    const scalarField& dc = mesh.deltaCoeffs().primitiveField();
    forAll(gg, f) { gg[f] = magSf[f]*dc[f]; }

    List<int> bfc(nbf);
    label off = 0;
    forAll(mesh.boundary(), patchi)
    {
        const labelUList& fc = mesh.boundary()[patchi].faceCells();
        forAll(fc, k) { bfc[off + k] = fc[k]; }
        off += fc.size();
    }

    const scalarField Vc(mesh.V());
    if (rgpPEqnMeshUpload
        (
            nc, nif, own.begin(), nei.begin(), gg.begin(),
            Vc.begin(), nbf, bfc.begin()
        ))
    {
        FatalErrorInFunction
            << "rgpPEqnMeshUpload: " << rgpPEqnLastError()
            << exit(FatalError);
    }

    // 수송용 메시 (선형 가중치, Sf, d, 경계 Sf)
    const surfaceScalarField& wCD = mesh.surfaceInterpolation::weights();
    const surfaceVectorField& Sff = mesh.Sf();
    const volVectorField& CC = mesh.C();

    List<double> wl(nif), sf3(3*nif), d3(3*nif);
    List<double> bsf(3*max(nbf, label(1)), 0.0);
    forAll(mesh.owner(), f)
    {
        wl[f] = wCD.primitiveField()[f];
        const vector d(CC[mesh.neighbour()[f]] - CC[mesh.owner()[f]]);
        for (label k = 0; k < 3; k++)
        {
            sf3[k*nif + f] = Sff.primitiveField()[f][k];
            d3[k*nif + f] = d[k];
        }
    }
    off = 0;
    forAll(mesh.boundary(), patchi)
    {
        const vectorField& Sfb = Sff.boundaryField()[patchi];
        forAll(Sfb, fi)
        {
            for (label k = 0; k < 3; k++)
            {
                bsf[k*nbf + off + fi] = Sfb[fi][k];
            }
        }
        off += Sfb.size();
    }
    if (rgpSTEqnMeshUpload(wl.begin(), sf3.begin(), d3.begin(),
                           bsf.begin()))
    {
        FatalErrorInFunction
            << "rgpSTEqnMeshUpload: " << rgpPEqnLastError()
            << exit(FatalError);
    }

    gpuMeshArmed_ = true;
    gpuMeshCells_ = nc;
    gpuMeshFaces_ = nif;
    gpuMeshStampTime_ = mesh.time().timeIndex();
    Info<< "gpuMulticomponentFluid: GPU transport mesh armed -- "
        << nc << " cells" << nl << endl;
}


void Foam::solvers::gpuMulticomponentFluid::checkGpuGuards
(
    const volScalarField& f
) const
{
    if
    (
        mesh.solution().relaxEquation(f.name())
     && mesh.solution().equationRelaxationFactor(f.name()) != 1
    )
    {
        FatalErrorInFunction
            << "gpuYEqn/gpuEEqn (v1) do not support equation relaxation "
            << "on " << f.name() << exit(FatalError);
    }
    if (fvModels().addsSupToField(f.name()))
    {
        FatalErrorInFunction
            << "gpuYEqn/gpuEEqn (v1) do not support fvModels sources "
            << "on " << f.name() << exit(FatalError);
    }
    const Foam::fvConstraints& cs = fvConstraints();
    forAll(cs, i)
    {
        if (cs[i].constrainsField(f.name()))
        {
            FatalErrorInFunction
                << "gpuYEqn/gpuEEqn (v1) do not support fvConstraints "
                << "on " << f.name() << " (constraint '" << cs[i].name()
                << "')" << exit(FatalError);
        }
    }
}


// ************************************************************************* //
