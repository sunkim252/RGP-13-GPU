/*---------------------------------------------------------------------------*\
  gpuMulticomponentFluid — 생성자/아밍/가드 (개요는 헤더 참조)
\*---------------------------------------------------------------------------*/

#include "gpuMulticomponentFluid.H"
#include "addToRunTimeSelectionTable.H"
#include "fvConstraint.H"
#include "snGradScheme.H"
#include "processorFvPatchFields.H"
#include "processorFvPatch.H"
#include "PstreamReduceOps.H"
#include "gpu/rgpPEqnTypes.H"
#include "gpu/rgpKernelTypes.H"

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
    gpuUEqn_(false),
    gpuPEqn_(false),
    gpuCheck_(false),
    gpuDiffExtract_(false),
    gpuMeshArmed_(false),
    gpuMeshCells_(-1),
    gpuMeshFaces_(-1),
    gpuMeshStampTime_(-1),
    gpuNonOrtho_(false),
    gpuNonOrthoWarned_(false),
    gpuNOSrcDev_(false),
    gpuNOLimitCoeff_(-1)
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
        gpuUEqn_ = dict.lookupOrDefault<Switch>("gpuUEqn", false);
        gpuPEqn_ = dict.lookupOrDefault<Switch>("gpuPEqn", false);
        gpuCheck_ = dict.lookupOrDefault<Switch>("gpuCheck", false);
    }

    // 병렬: fgmFluid와 동일한 processor 인터페이스 halo 인프라
    // (librgpThermoGPU의 gPar/greduce/parB)를 재사용 — 전 스위치 지원.
    if (Pstream::parRun() && gpuCheck_)
    {
        FatalErrorInFunction
            << "gpuCheck is serial-only" << exit(FatalError);
    }
    if (gpuUEqn_ != gpuPEqn_)
    {
        FatalErrorInFunction
            << "gpuUEqn and gpuPEqn must be enabled together (v1): the "
            << "GPU pressure corrector consumes the device-assembled "
            << "UEqn's rAU/H, and the stock corrector needs the CPU "
            << "fvMatrix that the GPU momentum predictor bypasses"
            << exit(FatalError);
    }
    if (gpuEEqn_ && !gpuYEqn_)
    {
        FatalErrorInFunction
            << "gpuEEqn requires gpuYEqn (shared multivariate weights)"
            << exit(FatalError);
    }

    if (gpuYEqn_)
    {
        // 확산 처리 모드: unityLewis*는 gamma=alphaEff 셀 필드의
        // 디바이스 면 보간(고속 경로); 그 외(Fickian/FickianFourier/
        // FickianEddyDiffusivity 등 비-unity Lewis)는 모델이 조립한
        // divj/divq 행렬에서 계수·명시항을 추출해 주입(일반 경로 —
        // 종별 DEff·Soret(DT)·비직교 소스 자동 포섭)
        const word ttType(thermophysicalTransport->type());
        gpuDiffExtract_ =
            ttType.find("unityLewis") == std::string::npos;
        if (gpuDiffExtract_)
        {
            Info<< "gpuMulticomponentFluid: non-unity-Lewis transport ("
                << ttType << ") -- diffusion operators extracted from "
                << "the model's divj/divq matrices" << nl << endl;
        }
        if
        (
            thermo.he().name() != "h"
         && thermo.he().name() != "e"
         && gpuEEqn_
        )
        {
            FatalErrorInFunction
                << "gpuEEqn supports sensible enthalpy (he == h) or "
                << "sensible internal energy (he == e) only"
                << exit(FatalError);
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

    // 랭크→GPU 매핑 + 코히런트 HW 감지(gRgpUnified)를 **버퍼 할당 전**
    // 수행 — 미호출이면 전 랭크가 device 0에 할당(멀티 GPU 노드에서
    // cudaErrorIllegalAddress)하고 GH200 native 감지도 안 된다.
    // (fgmFluid::gpuThermoCorrect 아밍과 동일 규약; 멱등 — 래치 뒤라
    // 아밍/재아밍 시에만 실행)
    {
        const int nDev = max(rgpGpuDeviceCount(), 1);
        const int dev = Pstream::parRun()
            ? (Pstream::myProcNo() % nDev) : 0;
        if (rgpGpuInit(dev))
        {
            FatalErrorInFunction
                << "rgpGpuInit(device " << dev << ") failed"
                << exit(FatalError);
        }
    }

    label nbf = 0;
    forAll(thermo.T().boundaryField(), patchi)
    {
        if
        (
            thermo.T().boundaryField()[patchi].coupled()
         && !isA<processorFvPatchScalarField>
            (
                thermo.T().boundaryField()[patchi]
            )
        )
        {
            FatalErrorInFunction
                << "GPU transport (v1) supports processor coupling only"
                << exit(FatalError);
        }
        nbf += thermo.T().boundaryField()[patchi].size();
    }

    // 비직교 스킴 감지 (fgmFluid::armGpuPEqnMesh 1:1): corrected/limited
    // snGrad면 ⓐ 면 계수는 스킴 deltaCoeffs(=nonOrthDeltaCoeffs)
    // ⓑ 명시 보정 소스는 각 방정식 조립부가 scheme.correction()으로
    // 호스트 계산해 주입 (gaussLaplacianScheme 1:1). gg는 pEqn·수송·
    // UEqn 라플라시안 공용 — 필드별 스킴이 다르면 기본(p) 스킴 기준.
    tmp<fv::snGradScheme<scalar>> tsnP
    (
        fv::snGradScheme<scalar>::New
        (
            mesh, mesh.schemes().snGrad(p_.name())
        )
    );
    gpuNonOrtho_ = tsnP().corrected();
    if (gpuNonOrtho_ && !gpuNonOrthoWarned_)
    {
        Info<< "gpuMulticomponentFluid: corrected snGrad scheme detected"
            << " -- GPU laplacians use scheme deltaCoeffs + explicit "
            << "non-orthogonal correction sources (host-computed, "
            << "1:1 with gaussLaplacianScheme)" << nl << endl;
        gpuNonOrthoWarned_ = true;
    }
    const surfaceScalarField dcsP(tsnP().deltaCoeffs(p_));

    List<int> own(nif), nei(nif);
    forAll(mesh.owner(), f)
    {
        own[f] = mesh.owner()[f];
        nei[f] = mesh.neighbour()[f];
    }
    List<double> gg(nif);
    const scalarField& magSf = mesh.magSf().primitiveField();
    const scalarField& dc = dcsP.primitiveField();
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

    // 비직교 보정 소스 디바이스화 자격 판정·아밍 (fgmFluid 1:1 —
    // corrected/limited-corrected snGrad + Gauss linear grad(p) + 직렬;
    // 부적격/VRAM 부족이면 호스트 fvc 경로 유지)
    gpuNOSrcDev_ = false;
    if (gpuNonOrtho_)
    {
        scalar lc = -1;
        bool devOK = false;
        {
            ITstream& isn = mesh.schemes().snGrad(p_.name());
            isn.rewind();
            const word snType(isn);
            if (snType == "corrected") { devOK = true; }
            else if (snType == "limited")
            {
                token t(isn);
                if (t.isWord() && t.wordToken() == "corrected")
                {
                    lc = readScalar(isn);
                    devOK = true;
                }
                else if (t.isNumber())
                {
                    lc = t.number();
                    devOK = true;
                }
            }
        }
        // grad(p) 스킴: Gauss linear (mdK<0) 또는 cellMDLimited Gauss
        // linear k (결정론 CSR 재생으로 디바이스 리미팅)
        scalar mdK = -1;
        if (devOK)
        {
            ITstream& isg =
                mesh.schemes().grad("grad(" + p_.name() + ')');
            isg.rewind();
            word gd1(isg);
            if (gd1 == "cellMDLimited")
            {
                word gd2;
                if (!isg.eof()) { gd2 = word(isg); }
                word gd3;
                if (!isg.eof()) { gd3 = word(isg); }
                devOK = (gd2 == "Gauss" && gd3 == "linear");
                if (devOK)
                {
                    mdK = readScalar(isg);
                    if (mdK < small) { mdK = -1; }
                }
            }
            else
            {
                word gd2;
                if (!isg.eof()) { gd2 = word(isg); }
                devOK = (gd1 == "Gauss" && gd2 == "linear");
            }
        }
        if (devOK)
        {
            List<double> dno(nif);
            const scalarField& dnof =
                mesh.nonOrthDeltaCoeffs().primitiveField();
            forAll(dno, f) { dno[f] = dnof[f]; }
            const surfaceVectorField& kf =
                mesh.nonOrthCorrectionVectors();
            gpuNOKf3_.setSize(3*nif);
            forAll(mesh.owner(), f)
            {
                for (label k = 0; k < 3; k++)
                {
                    gpuNOKf3_[k*nif + f] = kf.primitiveField()[f][k];
                }
            }
            List<double> C3, CfB3;
            if (mdK >= 0)
            {
                const surfaceVectorField& Cf = mesh.Cf();
                const volVectorField& C = mesh.C();
                gpuNOCf3_.setSize(3*nif);
                C3.setSize(3*nc);
                CfB3.setSize(3*max(nbf, label(1)), 0.0);
                forAll(mesh.owner(), f)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        gpuNOCf3_[k*nif + f] =
                            Cf.primitiveField()[f][k];
                    }
                }
                for (label i = 0; i < nc; i++)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        C3[k*nc + i] = C.primitiveField()[i][k];
                    }
                }
                label off = 0;
                forAll(mesh.boundary(), patchi)
                {
                    const vectorField& pCf =
                        Cf.boundaryField()[patchi];
                    forAll(pCf, k2)
                    {
                        for (label k = 0; k < 3; k++)
                        {
                            CfB3[k*nbf + off + k2] = pCf[k2][k];
                        }
                    }
                    off += pCf.size();
                }
            }
            const int rcNO = rgpPEqnNOArm
            (
                dno.begin(), mdK,
                mdK >= 0 ? C3.begin() : nullptr,
                mdK >= 0 ? CfB3.begin() : nullptr,
                nullptr   // magSf: gmc는 devChainNO 미사용
            );
            if (rcNO == 0)
            {
                gpuNOSrcDev_ = true;
                gpuNOLimitCoeff_ = lc;
                rgpSTDvecRelease();
                Info<< "gpuMulticomponentFluid: non-orthogonal "
                    << "correction source on the device"
                    << (mdK >= 0 ? " (grad cellMDLimited)" : "")
                    << nl << endl;
            }
            else
            {
                WarningInFunction
                    << "device non-ortho source unavailable ("
                    << rgpPEqnLastError()
                    << ") -- keeping the host fvc path" << endl;
            }
        }
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

    // 병렬: processor 패치 구조 아밍 (halo 교환 + 전역 리덕션)
    if (Pstream::parRun())
    {
        DynamicList<int> pNbr, pNF, pFc;
        forAll(mesh.boundary(), patchi)
        {
            if (!thermo.T().boundaryField()[patchi].coupled()) continue;
            const processorFvPatch& pp =
                refCast<const processorFvPatch>(mesh.boundary()[patchi]);
            pNbr.append(pp.neighbProcNo());
            pNF.append(pp.size());
            const labelUList& fc = pp.faceCells();
            forAll(fc, k) { pFc.append(fc[k]); }
        }
        const double gnCells = returnReduce(nc, sumOp<label>());
        if (rgpPEqnParArm
            (
                pNbr.size(),
                pNbr.size() ? pNbr.begin() : nullptr,
                pNbr.size() ? pNF.begin() : nullptr,
                pNbr.size() ? pFc.begin() : nullptr,
                gnCells
            ))
        {
            FatalErrorInFunction
                << "rgpPEqnParArm: " << rgpPEqnLastError()
                << exit(FatalError);
        }
        gpuParB_.setSize(max(pFc.size(), label(1)));
    }

    gpuMeshArmed_ = true;
    gpuMeshCells_ = nc;
    gpuMeshFaces_ = nif;
    gpuMeshStampTime_ = mesh.time().timeIndex();
    {
        const int um = rgpGpuUnifiedMode();
        Info<< "gpuMulticomponentFluid: GPU transport mesh armed -- "
            << nc << " cells; staging "
            << (um == 2 ? "unified-native (coherent, zero-copy)"
              : um == 1 ? "mapped (validation)"
                        : "copy (discrete)")
            << " [coherent HW: " << (rgpGpuCoherentHW() ? "yes" : "no")
            << "]" << nl << endl;
    }
}


void Foam::solvers::gpuMulticomponentFluid::checkGpuGuards
(
    const volScalarField& f
) const
{
    // 방정식 완화는 지원 (rgpSTEqnSolve의 디바이스 fvMatrix::relax 1:1)
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
