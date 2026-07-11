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

    // CPU 규약: fvMatrix 생성자가 BC updateCoeffs()를 호출해 phi 의존
    // BC(inletOutlet 등)의 valueFraction을 갱신한다 — 직접 조립 경로도
    // 동일하게 갱신 (누락 시 스테일 계수로 outlet 인접 셀이 K-스케일
    // 이탈 — gmc 벤치 실측)
    psiF.boundaryFieldRef().updateCoeffs();

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

    // ── multivariate limitedLinear 가중치: CPU 스킴 자체로 계산해
    //    업로드. 디바이스 자체 계산은 컴파일러 FMA 축약(gcc
    //    -ffp-contract=fast vs nvcc -fmad=false)의 ULP 차이로 평탄
    //    필드(예: h ~2e6에 1 ULP 요동)에서 limiter의 이산 결정이
    //    뒤집혀 K-스케일 오차로 증폭된다(실측) — 호스트 계산이 유일한
    //    비트-일치 경로. fields 테이블 = 모든 Y + he (CPU 규약 1:1),
    //    coupled(proc) 면 가중치도 스킴이 함께 산출 ──
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
    gpuProcW_.setSize(max(nPar, label(1)));

    {
        ITstream& is = mesh.schemes().div("div(phi,Yi_h)");
        is.rewind();
        const word gaussName(is);   // "Gauss" 소비

        tmp<multivariateSurfaceInterpolationScheme<scalar>> tmvs
        (
            multivariateSurfaceInterpolationScheme<scalar>::New
            (
                mesh, fields, phi, is
            )
        );
        const surfaceScalarField wMv
        (
            tmvs()(Y[0])().weights(Y[0])
        );

        const scalarField& wi = wMv.primitiveField();
        gpuBuf_.setSize(wi.size());
        forAll(wi, f) { gpuBuf_[f] = wi[f]; }
        if (rgpSTWeightsSet(gpuBuf_.begin()))
        {
            FatalErrorInFunction
                << "rgpSTWeightsSet: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        if (nPar > 0)
        {
            label offp = 0;
            forAll(mesh.boundary(), patchi)
            {
                if (!thermo.T().boundaryField()[patchi].coupled())
                {
                    continue;
                }
                const scalarField& wb = wMv.boundaryField()[patchi];
                forAll(wb, k) { gpuProcW_[offp + k] = wb[k]; }
                offp += wb.size();
            }
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
    // unityLewis*: divj의 DEff = kappa/Cp (에너지 형식 무관),
    // divq의 alphahe = kappa/Cpv (h→Cp, e→Cv) — 모델 구현 1:1.
    // he==h 면 두 값이 같지만 he==e 면 다르다 (+ 난류 alphat 공통)
    volScalarField gammaY("gammaY", thermo.kappa()/thermo.Cp());
    volScalarField gammaHe("gammaHe", thermo.kappa()/thermo.Cpv());
    if (mesh.foundObject<volScalarField>("alphat"))
    {
        const volScalarField& alphat =
            mesh.lookupObject<volScalarField>("alphat");
        gammaY += alphat;
        gammaHe += alphat;
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
            Yi, gammaY, sp, src, word("Yi"),
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

        if (mesh.moving())
        {
            FatalErrorInFunction
                << "gpuEEqn (v1) does not support moving meshes "
                << "(pressureWork mesh-flux term)" << exit(FatalError);
        }

        // 명시항 (per-volume): ddt(rho,K) + div(phi,K) + pressureWork
        // - Qdot 는 소스로 이항 (divq의 laplacian만 암시 조립)
        const volScalarField expl(fvc::ddt(rho, K) + fvc::div(phi, K));
        scalarField src(reaction->Qdot()().primitiveField());

        if (he.name() == "e")
        {
            // pressureWork(e-형식) = +mvConvection->fvcDiv(phi, p/rho)
            // (정적 메시: pressureWork(work)=work) — 디바이스
            // multivariate 가중치(결합 min-리미터는 CPU처럼 임의
            // 필드에 적용됨)로 flux-div, proc-면은 호스트 가산
            const scalarField qc
            (
                p.primitiveField()/rho.primitiveField()
            );

            label nbfW = 0;
            forAll(mesh.boundary(), patchi)
            {
                nbfW += mesh.boundary()[patchi].size();
            }
            scalarField bQ(nbfW, 0.0), bPhi(nbfW, 0.0);
            {
                label off = 0;
                forAll(mesh.boundary(), patchi)
                {
                    const fvPatchScalarField& pb =
                        p.boundaryField()[patchi];
                    if (!pb.coupled())
                    {
                        const fvPatchScalarField& rhob =
                            rho.boundaryField()[patchi];
                        const scalarField& phb =
                            phi.boundaryField()[patchi];
                        forAll(pb, fi)
                        {
                            bQ[off + fi] = pb[fi]/rhob[fi];
                            bPhi[off + fi] = phb[fi];
                        }
                    }
                    off += pb.size();
                }
            }

            scalarField divW(mesh.nCells());
            if (rgpSTFluxDiv
                (
                    qc.begin(), bQ.begin(),
                    phi.primitiveField().begin(), bPhi.begin(),
                    divW.begin()
                ))
            {
                FatalErrorInFunction
                    << "rgpSTFluxDiv: " << rgpPEqnLastError()
                    << exit(FatalError);
            }
            src -= divW;

            // proc-면 기여: -phb*(w*q_own + (1-w)*q_nbr)/V
            if (nPar > 0)
            {
                const scalarField& V = mesh.V();
                label offp = 0;
                forAll(mesh.boundary(), patchi)
                {
                    const fvPatchScalarField& pb =
                        p.boundaryField()[patchi];
                    if (!pb.coupled()) continue;
                    const fvPatchScalarField& rhob =
                        rho.boundaryField()[patchi];
                    const scalarField qo
                    (
                        pb.patchInternalField()
                       /rhob.patchInternalField()
                    );
                    const scalarField qn
                    (
                        pb.patchNeighbourField()
                       /rhob.patchNeighbourField()
                    );
                    const scalarField& phb =
                        phi.boundaryField()[patchi];
                    const labelUList& fc =
                        mesh.boundary()[patchi].faceCells();
                    forAll(pb, k)
                    {
                        const scalar w = gpuProcW_[offp + k];
                        src[fc[k]] -=
                            phb[k]*(w*qo[k] + (1.0 - w)*qn[k])
                           /V[fc[k]];
                    }
                    offp += pb.size();
                }
            }
        }
        else
        {
            src += dpdt;   // pressureWork(-dpdt) 이항
        }

        if (buoyancy.valid())
        {
            // 부력 소스: +rho*(U & g)
            const vector gv = buoyancy->g.value();
            const scalarField& rhoc = rho.primitiveField();
            const vectorField& Uc = U.primitiveField();
            forAll(src, i) { src[i] += rhoc[i]*(Uc[i] & gv); }
        }

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

        // 검증 모드: he 조립 비교 (mvConvection = Y솔브 전 가중치 —
        // 디바이스 wLim과 동일 시점; chk 소스에 명시항 fold)
        tmp<fvScalarMatrix> tChkE;
        if (gpuCheck_)
        {
            tChkE = tmp<fvScalarMatrix>
            (
                new fvScalarMatrix
                (
                    fvm::ddt(rho, he)
                  + mvConvection->fvmDiv(phi, he)
                  + thermophysicalTransport->divq(he)
                )
            );
        }

        solveScalarGpu
        (
            he, gammaHe, sp, src, he.name(),
            tDivq.valid() ? &tDivq.ref() : nullptr
        );

        if (gpuCheck_)
        {
            fvScalarMatrix& chk = tChkE.ref();
            scalarField diagC(chk.diag());
            scalarField srcC(chk.source());
            srcC += src*mesh.V();
            forAll(chk.internalCoeffs(), patchi)
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
            scalar dM = 0, uM = 0, bM = 0, lM = 0;
            forAll(diagC, c)
            {
                dM = max(dM, mag(dG[c] - diagC[c])
                        /max(mag(diagC[c]), small));
                bM = max(bM, mag(bG[c] - srcC[c])
                        /max(mag(srcC[c]), small));
            }
            const scalarField& uC = chk.upper();
            const scalarField& lC = chk.lower();
            forAll(uC, f)
            {
                uM = max(uM, mag(uG[f] - uC[f])/max(mag(uC[f]), small));
                lM = max(lM, mag(lG[f] - lC[f])/max(mag(lC[f]), small));
            }
            Info<< "gpuCheck(" << he.name() << "): maxRel diag = " << dM
                << ", upper = " << uM << ", lower = " << lM
                << ", source = " << bM << endl;
        }

        fvConstraints().constrain(he);
    }
    else
    {
        tmp<fv::convectionScheme<scalar>> mvConvE
        (
            fv::convectionScheme<scalar>::New
            (
                mesh, fields, phi, mesh.schemes().div("div(phi,Yi_h)")
            )
        );

        fvScalarMatrix EEqn
        (
            fvm::ddt(rho, he)
          + mvConvE->fvmDiv(phi, he)
          + fvc::ddt(rho, K) + fvc::div(phi, K)
          + pressureWork
            (
                he.name() == "e"
              ? mvConvE->fvcDiv(phi, p/rho)()
              : -dpdt
            )
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
