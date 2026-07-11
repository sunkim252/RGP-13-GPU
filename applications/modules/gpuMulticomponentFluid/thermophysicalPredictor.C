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
#include "processorFvPatch.H"
#include "PstreamBuffers.H"
#include "gpu/rgpPEqnTypes.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::gpuMulticomponentFluid::solveScalarGpu
(
    volScalarField& psiF,
    const volScalarField& gamma,
    const scalarField& sp,
    const scalarField& src,
    const word& dictBase,
    fvScalarMatrix* diffOp
)
{
    // Fickian류 추출 경로: 모델이 조립한 확산 연산자(divj/divq)에서
    // 면 계수 Γ_f = -upper, 경계 iC/bC를 그대로 주입 — 종별 DEff·
    // 명시 보정항(DT 등)·비직교 소스가 자동 포섭된다. 명시 source는
    // 호출자가 src에 합산.
    List<double> Gface;
    if (diffOp)
    {
        if (diffOp->hasLower())
        {
            FatalErrorInFunction
                << "asymmetric diffusion operator is not supported"
                << exit(FatalError);
        }
        const scalarField& up = diffOp->upper();
        Gface.setSize(up.size());
        forAll(up, f) { Gface[f] = -up[f]; }
    }

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
    label offPar = 0;
    forAll(psiF.boundaryField(), patchi)
    {
        const fvPatchScalarField& pp = psiF.boundaryField()[patchi];
        const label np = pp.size();
        if (np == 0) continue;

        const scalarField& phb = phi.boundaryField()[patchi];

        if (pp.coupled())
        {
            // processor: div 가중치 = multivariate limited(gpuProcW_),
            // 확산 iC/bC는 추출(diffOp) 또는 deltaCoeffs 공식.
            scalarField wP(np);
            for (label k = 0; k < np; k++)
            {
                wP[k] = gpuProcW_[offPar + k];
            }
            const scalarField vic(pp.valueInternalCoeffs(wP));
            const scalarField vbc(pp.valueBoundaryCoeffs(wP));

            if (diffOp)
            {
                const scalarField& oiC =
                    diffOp->internalCoeffs()[patchi];
                const scalarField& obC =
                    diffOp->boundaryCoeffs()[patchi];
                for (label k = 0; k < np; k++)
                {
                    bDiagA[off + k] = phb[k]*vic[k] + oiC[k];
                    bSrcA[off + k] = 0;
                    bPsiA[off + k] = pp[k];
                    gpuParB_[offPar + k] =
                        -phb[k]*vbc[k] + obC[k];
                }
            }
            else
            {
                const scalarField pdc
                (
                    mesh.deltaCoeffs().boundaryField()[patchi]
                );
                const scalarField gic(pp.gradientInternalCoeffs(pdc));
                const scalarField gbc(pp.gradientBoundaryCoeffs(pdc));

                const scalarField cdw
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField gOwn
                (
                    gamma.boundaryField()[patchi].patchInternalField()
                );
                const scalarField& gNei =
                    gamma.boundaryField()[patchi];
                const scalarField& msf =
                    mesh.magSf().boundaryField()[patchi];

                for (label k = 0; k < np; k++)
                {
                    const scalar gMsf =
                        (cdw[k]*gOwn[k] + (1.0 - cdw[k])*gNei[k])
                       *msf[k];
                    bDiagA[off + k] = phb[k]*vic[k] - gMsf*gic[k];
                    bSrcA[off + k] = 0;
                    bPsiA[off + k] = pp[k];
                    gpuParB_[offPar + k] =
                        -phb[k]*vbc[k] + gMsf*gbc[k];
                }
            }
            offPar += np;
        }
        else
        {
            // div: iC = +phi_b*vIC, bC(source) = -phi_b*vBC
            const scalarField vic
            (
                pp.valueInternalCoeffs
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                )
            );
            const scalarField vbc
            (
                pp.valueBoundaryCoeffs
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                )
            );

            if (diffOp)
            {
                const scalarField& oiC =
                    diffOp->internalCoeffs()[patchi];
                const scalarField& obC =
                    diffOp->boundaryCoeffs()[patchi];
                for (label k = 0; k < np; k++)
                {
                    bDiagA[off + k] = phb[k]*vic[k] + oiC[k];
                    bSrcA[off + k] = -phb[k]*vbc[k] + obC[k];
                    bPsiA[off + k] = pp[k];
                }
            }
            else
            {
                // -laplacian: iC = -gammaMagSf*gic, bC = +gMsf*gbc
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
                    bSrcA[off + k] =
                        -phb[k]*vbc[k] + gMsf[k]*gbc[k];
                    bPsiA[off + k] = pp[k];
                }
            }
        }
        off += np;
    }

    if (offPar > 0 && rgpPEqnParCoeffs(gpuParB_.begin()))
    {
        FatalErrorInFunction
            << "rgpPEqnParCoeffs: " << rgpPEqnLastError()
            << exit(FatalError);
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
            diffOp ? nullptr : gamma.primitiveField().begin(),
            diffOp ? Gface.begin() : nullptr,
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

    // ── 병렬: proc-면 리미터 준비 (fgmFluid gpuZC와 동일 구조) ──
    label nPar = 0;
    if (Pstream::parRun())
    {
        forAll(mesh.boundary(), patchi)
        {
            if (thermo.T().boundaryField()[patchi].coupled())
            {
                nPar += mesh.boundary()[patchi].size();
            }
        }
    }
    List<double> procLim(nPar, 1.0);
    gpuProcW_.setSize(max(nPar, label(1)));

    auto exchangeParVec3 = [&](const List<double>& sIn, List<double>& r)
    {
        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
        label offp = 0;
        forAll(mesh.boundary(), patchi)
        {
            if (!thermo.T().boundaryField()[patchi].coupled()) continue;
            const processorFvPatch& pp =
                refCast<const processorFvPatch>(mesh.boundary()[patchi]);
            const label np = pp.size();
            List<vector> sv(np);
            for (label k = 0; k < np; k++)
            {
                sv[k] = vector
                (
                    sIn[offp + k], sIn[nPar + offp + k],
                    sIn[2*nPar + offp + k]
                );
            }
            UOPstream toNbr(pp.neighbProcNo(), pBufs);
            toNbr << sv;
            offp += np;
        }
        pBufs.finishedSends();
        offp = 0;
        forAll(mesh.boundary(), patchi)
        {
            if (!thermo.T().boundaryField()[patchi].coupled()) continue;
            const processorFvPatch& pp =
                refCast<const processorFvPatch>(mesh.boundary()[patchi]);
            const label np = pp.size();
            List<vector> rv;
            UIPstream fromNbr(pp.neighbProcNo(), pBufs);
            fromNbr >> rv;
            for (label k = 0; k < np; k++)
            {
                r[offp + k] = rv[k].x();
                r[nPar + offp + k] = rv[k].y();
                r[2*nPar + offp + k] = rv[k].z();
            }
            offp += np;
        }
    };

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
            if (pf.coupled())
            {
                // gaussGrad coupled: 경계 기여 = 면 값
                const scalarField w
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField own(pf.patchInternalField());
                forAll(pf, fi)
                {
                    bPsiA[off + fi] =
                        w[fi]*own[fi] + (1.0 - w[fi])*pf[fi];
                }
            }
            else
            {
                forAll(pf, fi) { bPsiA[off + fi] = pf[fi]; }
            }
            off += pf.size();
        }
        if (rgpSTWeightsField(f.primitiveField().begin(), bPsiA))
        {
            FatalErrorInFunction
                << "rgpSTWeightsField(" << f.name() << "): "
                << rgpPEqnLastError() << exit(FatalError);
        }

        // 병렬: 이 필드의 proc-면 limiter min-누적
        // (LimitedScheme.C coupled 분기 1:1)
        if (nPar > 0)
        {
            List<double> gradP(3*nPar), gradN(3*nPar);
            if (rgpSTGradAtPar(gradP.begin()))
            {
                FatalErrorInFunction
                    << "rgpSTGradAtPar: " << rgpPEqnLastError()
                    << exit(FatalError);
            }
            exchangeParVec3(gradP, gradN);

            label offp = 0;
            forAll(f.boundaryField(), patchi)
            {
                const fvPatchScalarField& pf =
                    f.boundaryField()[patchi];
                if (!pf.coupled()) continue;
                const label np = pf.size();

                const scalarField phiP(pf.patchInternalField());
                const scalarField& phb = phi.boundaryField()[patchi];
                const vectorField pd(mesh.boundary()[patchi].delta());

                for (label k = 0; k < np; k++)
                {
                    const label j = offp + k;
                    const vector gP
                    (
                        gradP[j], gradP[nPar + j], gradP[2*nPar + j]
                    );
                    const vector gN
                    (
                        gradN[j], gradN[nPar + j], gradN[2*nPar + j]
                    );
                    const scalar gradf = pf[k] - phiP[k];
                    const scalar gradcf =
                        pd[k] & (phb[k] > 0 ? gP : gN);

                    scalar r;
                    if (mag(gradcf) >= 1000.0*mag(gradf))
                    {
                        r = 2.0*1000.0*sign(gradcf)*sign(gradf) - 1.0;
                    }
                    else
                    {
                        r = 2.0*(gradcf/gradf) - 1.0;
                    }
                    procLim[j] = min
                    (
                        procLim[j],
                        max(min(2.0*r, scalar(1)), scalar(0))
                    );
                }
                offp += np;
            }
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

    // 병렬: proc-면 최종 가중치 (w = lim*CD + (1-lim)*upwind)
    if (nPar > 0)
    {
        label offp = 0;
        forAll(mesh.boundary(), patchi)
        {
            if (!thermo.T().boundaryField()[patchi].coupled()) continue;
            const label np = mesh.boundary()[patchi].size();
            const scalarField cdw
            (
                mesh.surfaceInterpolation::weights()
                    .boundaryField()[patchi]
            );
            const scalarField& phb = phi.boundaryField()[patchi];
            for (label k = 0; k < np; k++)
            {
                const double lim = procLim[offp + k];
                gpuProcW_[offp + k] =
                    lim*cdw[k]
                  + (1.0 - lim)*((phb[k] >= 0.0) ? 1.0 : 0.0);
            }
            offp += np;
        }
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

        // Fickian류: divj(Yi) 추출 — 명시 소스는 src로, diag 기여는
        // sp로 (negSum 외 diag: 통상 0), 면/경계 계수는 diffOp로
        tmp<fvScalarMatrix> tDiffOp;
        if (gpuDiffExtract_)
        {
            tDiffOp = thermophysicalTransport->divj(Yi);
            src += tDiffOp().source();
            if (tDiffOp().hasDiag())
            {
                // 행렬 diag = negSum(upper) + 여분 — 디바이스 조립이
                // negSum을 재구성하므로 여분만 sp로
                const scalarField& od = tDiffOp().diag();
                const scalarField& ou = tDiffOp().upper();
                scalarField negSum(od.size(), 0.0);
                forAll(mesh.owner(), f)
                {
                    negSum[mesh.owner()[f]] -= ou[f];
                    negSum[mesh.neighbour()[f]] -= ou[f];
                }
                forAll(sp, c) { sp[c] -= (od[c] - negSum[c]); }
            }
        }
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

        solveScalarGpu
        (
            Yi, gammaHe, sp, src, word("Yi"),
            tDiffOp.valid() ? &tDiffOp.ref() : nullptr
        );

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

        scalarField sp(mesh.nCells(), 0.0);

        // Fickian류: divq(he) 추출 (종 엔탈피 확산 플럭스 보정 포함)
        tmp<fvScalarMatrix> tDivq;
        if (gpuDiffExtract_)
        {
            tDivq = thermophysicalTransport->divq(he);
            const scalarField& V = mesh.V();
            forAll(src, c) { src[c] += tDivq().source()[c]/V[c]; }
        }

        solveScalarGpu
        (
            he, gammaHe, sp, src, he.name(),
            tDivq.valid() ? &tDivq.ref() : nullptr
        );
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
