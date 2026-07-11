/*---------------------------------------------------------------------------*\
  gpuMulticomponentFluid — GPU Y 배치 + EEqn (stock thermophysicalPredictor
  의 1:1 디바이스 치환; CPU 경로는 base 호출로 유지)
\*---------------------------------------------------------------------------*/

#include "gpuMulticomponentFluid.H"
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "solutionControl.H"
#include "localEulerDdtScheme.H"
#include "gpu/rgpPEqnTypes.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::gpuMulticomponentFluid::solveScalarGpu
(
    volScalarField& psiF,
    const volScalarField& gamma,
    const scalarField& sp,
    const scalarField& src,
    const word& dictBase
)
{
    label nbf = 0;
    forAll(psiF.boundaryField(), patchi)
    {
        nbf += psiF.boundaryField()[patchi].size();
    }
    gpuBuf_.setSize(3*nbf);
    double* bDiagA = gpuBuf_.begin();
    double* bSrcA = bDiagA + nbf;
    double* bPsiA = bSrcA + nbf;

    label off = 0;
    forAll(psiF.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = psiF.boundaryField()[patchi];
        const label np = pp.size();
        if (np == 0) continue;

        const scalarField& phb = phi.boundaryField()[patchi];

        // div: iC = +phi_b*vIC, bC(source) = -phi_b*vBC
        const scalarField vic
        (
            pp.valueInternalCoeffs
            (
                mesh.surfaceInterpolation::weights().boundaryField()[patchi]
            )
        );
        const scalarField vbc
        (
            pp.valueBoundaryCoeffs
            (
                mesh.surfaceInterpolation::weights().boundaryField()[patchi]
            )
        );

        // -laplacian: iC = -gammaMagSf*gic, bC = +gammaMagSf*gbc
        const scalarField gMsf
        (
            gamma.boundaryField()[patchi]
           *mesh.magSf().boundaryField()[patchi]
        );
        const scalarField gic(pp.gradientInternalCoeffs());
        const scalarField gbc(pp.gradientBoundaryCoeffs());

        for (label k = 0; k < np; k++)
        {
            bDiagA[off + k] = phb[k]*vic[k] - gMsf[k]*gic[k];
            bSrcA[off + k] = -phb[k]*vbc[k] + gMsf[k]*gbc[k];
            bPsiA[off + k] = pp[k];
        }
        off += np;
    }

    // fvMatrix::solve(word) 규약: transient 최종 이터레이션 → "Final"
    const dictionary& sd = mesh.solution().solverDict
    (
        !mesh.schemes().steady()
     && solutionControl::finalIteration(mesh)
      ? word(dictBase + "Final")
      : dictBase
    );
    const scalar tol = sd.lookup<scalar>("tolerance");
    const scalar rtol = sd.lookupOrDefault<scalar>("relTol", 0);
    const label maxIter = sd.lookupOrDefault<label>("maxIter", 1000);

    const bool LTS = fv::localEulerDdt::enabled(mesh);

    double res0 = 0, resF = 0;
    int nIter = 0;
    if (rgpSTEqnSolve
        (
            LTS ? 1.0 : 1.0/runTime.deltaTValue(),
            LTS
          ? fv::localEulerDdt::localRDeltaT(mesh).primitiveField().begin()
          : nullptr,
            1,
            rho.primitiveField().begin(),
            rho.oldTime().primitiveField().begin(),
            psiF.oldTime().primitiveField().begin(),
            psiF.primitiveField().begin(),
            phi.primitiveField().begin(),
            gamma.primitiveField().begin(),
            sp.begin(),
            src.begin(),
            bPsiA, bDiagA, bSrcA,
            tol, rtol, maxIter,
            psiF.primitiveFieldRef().begin(),
            &res0, &resF, &nIter
        ))
    {
        FatalErrorInFunction
            << "rgpSTEqnSolve(" << psiF.name() << "): "
            << rgpPEqnLastError() << exit(FatalError);
    }

    Info<< "rgpBiCGStab: Solving for " << psiF.name()
        << ", Initial residual = " << res0
        << ", Final residual = " << resF
        << ", No Iterations " << nIter << endl;

    psiF.correctBoundaryConditions();
}


void Foam::solvers::gpuMulticomponentFluid::thermophysicalPredictor()
{
    if (!gpuYEqn_)
    {
        multicomponentFluid::thermophysicalPredictor();
        return;
    }

    armGpuMesh();

    // ── multivariate limitedLinear 가중치: mvConvection 생성 시점의
    //    필드 값(fields 테이블 = 모든 Y + he — CPU 규약 1:1) ──
    if (rgpSTWeightsBegin(phi.primitiveField().begin()))
    {
        FatalErrorInFunction
            << "rgpSTWeightsBegin: " << rgpPEqnLastError()
            << exit(FatalError);
    }
    label nbfW = 0;
    forAll(thermo.T().boundaryField(), patchi)
    {
        nbfW += thermo.T().boundaryField()[patchi].size();
    }
    gpuBuf_.setSize(3*nbfW);
    auto addWeightField = [&](const volScalarField& f)
    {
        double* bPsiA = gpuBuf_.begin();
        label off = 0;
        forAll(f.boundaryField(), patchi)
        {
            const fvPatchScalarField& pf = f.boundaryField()[patchi];
            forAll(pf, fi) { bPsiA[off + fi] = pf[fi]; }
            off += pf.size();
        }
        if (rgpSTWeightsField(f.primitiveField().begin(), bPsiA))
        {
            FatalErrorInFunction
                << "rgpSTWeightsField(" << f.name() << "): "
                << rgpPEqnLastError() << exit(FatalError);
        }
    };
    forAll(Y, i) { addWeightField(Y[i]); }
    addWeightField(thermo.he());
    if (rgpSTWeightsEnd())
    {
        FatalErrorInFunction
            << "rgpSTWeightsEnd: " << rgpPEqnLastError()
            << exit(FatalError);
    }

    // 검증 모드용 CPU mvConvection (가중치는 위와 같은 시점 값)
    tmp<fv::convectionScheme<scalar>> mvConvection;
    if (gpuCheck_)
    {
        mvConvection = fv::convectionScheme<scalar>::New
        (
            mesh, fields, phi, mesh.schemes().div("div(phi,Yi_h)")
        );
    }

    reaction->correct();

    // ── Y 배치 ──────────────────────────────────────────────────────
    // unityLewis*/Fourier: divj/divq의 gamma = alphaEff = kappa/Cpv
    // (+ 난류 alphat) — 모델 구현(eddyDiffusivity::alphaEff)과 1:1
    volScalarField gammaHe("gammaHe", thermo.kappa()/thermo.Cpv());
    if (mesh.foundObject<volScalarField>("alphat"))
    {
        gammaHe += mesh.lookupObject<volScalarField>("alphat");
    }

    forAll(Y, i)
    {
        volScalarField& Yi = Y_[i];

        if (!thermo_.solveSpecie(i))
        {
            Yi.correctBoundaryConditions();
            continue;
        }

        checkGpuGuards(Yi);

        // R(Yi) 추출: YiEqn == R 이항 규약 — diag += -R.diag,
        // b += -R.source (stCellAssemble의 sp/src는 per-volume)
        tmp<fvScalarMatrix> tR(reaction->R(Yi));
        const scalarField& V = mesh.V();
        // 비-const diag(): 순수 명시 R(diag 미할당)이면 0으로 지연 할당
        scalarField sp(tR.ref().diag());
        scalarField src(-tR().source());
        forAll(sp, c) { sp[c] /= V[c]; src[c] /= V[c]; }

        // unityLewis*: divj(Yi) = -laplacian(alpha*DEff(Yi), Yi);
        // 단상 alpha() = 1, unity Lewis → DEff = alphaEff
        tmp<fvScalarMatrix> tChk;
        if (gpuCheck_)
        {
            tChk = tmp<fvScalarMatrix>
            (
                new fvScalarMatrix
                (
                    fvm::ddt(rho, Yi)
                  + mvConvection->fvmDiv(phi, Yi)
                  + thermophysicalTransport->divj(Yi)
                  - tR()
                )
            );
        }

        solveScalarGpu(Yi, gammaHe, sp, src, word("Yi"));

        // 계수 대조 (경계 iC/bC를 diag/source로 접은 뒤 GPU 덤프와 비교)
        if (gpuCheck_)
        {
            fvScalarMatrix& chk = tChk.ref();
            scalarField diagC(chk.diag());
            scalarField srcC(chk.source());
            forAll(Yi.boundaryField(), patchi)
            {
                const labelUList& fc =
                    mesh.boundary()[patchi].faceCells();
                const scalarField& iC = chk.internalCoeffs()[patchi];
                const scalarField& bC = chk.boundaryCoeffs()[patchi];
                forAll(fc, k)
                {
                    diagC[fc[k]] += iC[k];
                    srcC[fc[k]] += bC[k];
                }
            }
            List<double> dG(mesh.nCells()), uG(mesh.owner().size()),
                         lG(mesh.owner().size()), bG(mesh.nCells());
            rgpSTEqnDump(dG.begin(), uG.begin(), lG.begin(), bG.begin(),
                         nullptr);
            scalar dM = 0, uM = 0, bM = 0;
            forAll(diagC, c)
            {
                dM = max(dM, mag(dG[c] - diagC[c])
                        /max(mag(diagC[c]), small));
                bM = max(bM, mag(bG[c] - srcC[c])
                        /max(mag(srcC[c]), small));
            }
            const scalarField& uC = chk.upper();
            forAll(uC, f)
            {
                uM = max(uM, mag(uG[f] - uC[f])/max(mag(uC[f]), small));
            }
            Info<< "gpuCheck(" << Yi.name() << "): maxRel diag = " << dM
                << ", upper = " << uM << ", source = " << bM << endl;
        }

        fvConstraints().constrain(Yi);
    }

    thermo_.normaliseY();

    // ── EEqn ────────────────────────────────────────────────────────
    volScalarField& he = thermo_.he();

    if (gpuEEqn_)
    {
        checkGpuGuards(he);
        if (buoyancy.valid())
        {
            FatalErrorInFunction
                << "gpuEEqn (v1) does not support buoyancy"
                << exit(FatalError);
        }

        if (mesh.moving())
        {
            FatalErrorInFunction
                << "gpuEEqn (v1) does not support moving meshes "
                << "(pressureWork mesh-flux term)" << exit(FatalError);
        }

        // 명시항 (per-volume): ddt(rho,K) + div(phi,K) - dpdt - Qdot
        // 는 소스로 이항 (divq의 laplacian만 암시 조립)
        const volScalarField expl(fvc::ddt(rho, K) + fvc::div(phi, K));
        scalarField src(reaction->Qdot()().primitiveField());
        src += dpdt;
        src -= expl.primitiveField();

        const scalarField sp(mesh.nCells(), 0.0);
        solveScalarGpu(he, gammaHe, sp, src, he.name());
        fvConstraints().constrain(he);
    }
    else
    {
        fvScalarMatrix EEqn
        (
            fvm::ddt(rho, he)
          + fv::convectionScheme<scalar>::New
            (
                mesh, fields, phi, mesh.schemes().div("div(phi,Yi_h)")
            )->fvmDiv(phi, he)
          + fvc::ddt(rho, K) + fvc::div(phi, K)
          + pressureWork(-dpdt)
          + thermophysicalTransport->divq(he)
         ==
            reaction->Qdot()
          + (
                buoyancy.valid()
              ? fvModels().source(rho, he) + rho*(U & buoyancy->g)
              : fvModels().source(rho, he)
            )
        );
        EEqn.relax();
        fvConstraints().constrain(EEqn);
        EEqn.solve();
        fvConstraints().constrain(he);
    }

    thermo_.correct();
}


// ************************************************************************* //
