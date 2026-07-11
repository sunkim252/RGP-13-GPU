/*---------------------------------------------------------------------------*\
  gpuMulticomponentFluid — GPU momentum predictor (stock UEqn의 디바이스
  치환; fgmFluid gpuUEqn 이식, LAD 제외):
      fvm::ddt(rho,U) + fvm::div(phi,U)[limitedLinearV]
    + divDevTau(= -laplacian(muEff,U) - div(muEff*dev2(T(grad U))))
    == -grad(p) [솔브-전용 소스]
  v1: MRF/buoyancy/fvModels/fvConstraints/relax/consistent 미지원(Fatal),
  직렬 전용(생성자에서 병렬 자동 비활성), 정적 메시.
\*---------------------------------------------------------------------------*/

#include "gpuMulticomponentFluid.H"
#include "fvmDdt.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvcGrad.H"
#include "solutionControl.H"
#include "localEulerDdtScheme.H"
#include "processorFvPatch.H"
#include "PstreamBuffers.H"
#include "gpu/rgpPEqnTypes.H"

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::gpuMulticomponentFluid::momentumPredictor()
{
    if (!gpuUEqn_)
    {
        multicomponentFluid::momentumPredictor();
        return;
    }

    volVectorField& U(U_);

    if (MRF.size() > 0 || buoyancy.valid())
    {
        FatalErrorInFunction
            << "gpuUEqn (v1) does not support MRF/buoyancy"
            << exit(FatalError);
    }
    if
    (
        mesh.solution().relaxEquation(U.name())
     && mesh.solution().equationRelaxationFactor(U.name()) != 1
    )
    {
        FatalErrorInFunction
            << "gpuUEqn (v1) does not support relaxation on U"
            << exit(FatalError);
    }
    if (fvModels().addsSupToField(U.name()))
    {
        FatalErrorInFunction
            << "gpuUEqn (v1) does not support fvModels sources on U"
            << exit(FatalError);
    }
    {
        const Foam::fvConstraints& cs = fvConstraints();
        forAll(cs, i)
        {
            if (cs[i].constrainsField(U.name()))
            {
                FatalErrorInFunction
                    << "gpuUEqn (v1) does not support fvConstraints on U"
                    << exit(FatalError);
            }
        }
    }

    // v1: div(phi,U)는 Gauss limitedLinearV 전용 (디바이스 가중치 커널)
    {
        ITstream& is = mesh.schemes().div("div(phi,U)");
        bool ok = false;
        forAll(is, ti)
        {
            if
            (
                is[ti].isWord()
             && is[ti].wordToken().find("limitedLinearV")
             != std::string::npos
            )
            {
                ok = true;
            }
        }
        if (!ok)
        {
            FatalErrorInFunction
                << "gpuUEqn (v1) requires 'div(phi,U) Gauss "
                << "limitedLinearV 1'" << exit(FatalError);
        }
    }
    if (pimple.consistent())
    {
        FatalErrorInFunction
            << "gpuUEqn (v1) does not support 'consistent'"
            << exit(FatalError);
    }

    armGpuMesh();

    const label nc = mesh.nCells();
    label nbf = 0;
    forAll(U.boundaryField(), patchi)
    {
        nbf += U.boundaryField()[patchi].size();
    }

    // fgmFluid gpuUBuf_ 레이아웃과 동일 (15*nc + 22*nbf)
    gpuUBuf_.setSize(15*nc + 22*nbf);
    double* U3 = gpuUBuf_.begin();
    double* U3old = U3 + 3*nc;
    double* U3out = U3 + 6*nc;
    double* srcX3 = U3 + 12*nc;
    double* UB3 = srcX3 + 3*nc;             // [3*nbf]
    double* pB = UB3 + 3*nbf;               // [nbf]
    double* bDiag3 = pB + nbf;              // [3*nbf]
    double* bSrc3 = bDiag3 + 3*nbf;
    double* bFlux3 = bSrc3 + 3*nbf;
    double* bGrad9 = bFlux3 + 3*nbf;        // [9*nbf]
    (void)srcX3;

    const volScalarField muEff(rho*momentumTransport->nuEff());

    {
        const vectorField& Uc = U.primitiveField();
        const vectorField& Uo = U.oldTime().primitiveField();
        for (label i = 0; i < nc; i++)
        {
            for (label k = 0; k < 3; k++)
            {
                U3[k*nc + i] = Uc[i][k];
                U3old[k*nc + i] = Uo[i][k];
            }
        }
    }
    {
        label off = 0;
        forAll(U.boundaryField(), patchi)
        {
            const fvPatchVectorField& pp = U.boundaryField()[patchi];
            const fvPatchScalarField& ppp = p_.boundaryField()[patchi];
            if (pp.coupled())
            {
                const scalarField w
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const vectorField Uo(pp.patchInternalField());
                const scalarField po(ppp.patchInternalField());
                forAll(pp, f)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        UB3[k*nbf + off + f] =
                            w[f]*Uo[f][k] + (1.0 - w[f])*pp[f][k];
                    }
                    pB[off + f] = w[f]*po[f] + (1.0 - w[f])*ppp[f];
                }
            }
            else
            {
                forAll(pp, f)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        UB3[k*nbf + off + f] = pp[f][k];
                    }
                    pB[off + f] = ppp[f];
                }
            }
            off += pp.size();
        }
    }

    if (rgpUEqnGrad(U3, UB3, bGrad9))
    {
        FatalErrorInFunction
            << "rgpUEqnGrad: " << rgpPEqnLastError() << exit(FatalError);
    }

    // 병렬: proc-면 이웃 gradU 교환 → limitedLinearV 리미터 + dev2용
    label nPar = 0;
    forAll(U.boundaryField(), patchi)
    {
        if (U.boundaryField()[patchi].coupled())
        {
            nPar += U.boundaryField()[patchi].size();
        }
    }
    List<double> gradN9(9*max(nPar, label(1)), 0.0);
    List<double> wPar(max(nPar, label(1)), 0.0);
    if (nPar > 0)
    {
        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
        label off = 0;
        forAll(U.boundaryField(), patchi)
        {
            const fvPatchVectorField& pp = U.boundaryField()[patchi];
            const label np = pp.size();
            if (pp.coupled())
            {
                const processorFvPatch& prp =
                    refCast<const processorFvPatch>
                    (
                        mesh.boundary()[patchi]
                    );
                List<tensor> sv(np);
                for (label f = 0; f < np; f++)
                {
                    for (label m = 0; m < 9; m++)
                    {
                        sv[f][m] = bGrad9[m*nbf + off + f];
                    }
                }
                UOPstream toNbr(prp.neighbProcNo(), pBufs);
                toNbr << sv;
            }
            off += np;
        }
        pBufs.finishedSends();
        off = 0;
        label offp = 0;
        forAll(U.boundaryField(), patchi)
        {
            const fvPatchVectorField& pp = U.boundaryField()[patchi];
            const label np = pp.size();
            if (pp.coupled())
            {
                const processorFvPatch& prp =
                    refCast<const processorFvPatch>
                    (
                        mesh.boundary()[patchi]
                    );
                List<tensor> rv;
                UIPstream fromNbr(prp.neighbProcNo(), pBufs);
                fromNbr >> rv;
                for (label f = 0; f < np; f++)
                {
                    for (label m = 0; m < 9; m++)
                    {
                        gradN9[m*nPar + offp + f] = rv[f][m];
                    }
                }

                const scalarField cdw
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField& phb = phi.boundaryField()[patchi];
                const vectorField pd(mesh.boundary()[patchi].delta());
                const vectorField Uo(pp.patchInternalField());

                for (label f = 0; f < np; f++)
                {
                    const label j = offp + f;
                    vector gfv;
                    scalar gradf = 0;
                    for (label k = 0; k < 3; k++)
                    {
                        gfv[k] = pp[f][k] - Uo[f][k];
                        gradf += gfv[k]*gfv[k];
                    }
                    scalar gradcf = 0;
                    for (label k = 0; k < 3; k++)
                    {
                        scalar gcfk = 0;
                        for (label i = 0; i < 3; i++)
                        {
                            const scalar g =
                                (phb[f] > 0)
                              ? bGrad9[(3*i + k)*nbf + off + f]
                              : gradN9[(3*i + k)*nPar + j];
                            gcfk += pd[f][i]*g;
                        }
                        gradcf += gfv[k]*gcfk;
                    }
                    scalar r;
                    if (mag(gradcf) >= 1000.0*mag(gradf))
                    {
                        r = 2.0*1000.0*sign(gradcf)*sign(gradf) - 1.0;
                    }
                    else
                    {
                        r = 2.0*(gradcf/gradf) - 1.0;
                    }
                    const scalar lim =
                        max(min(2.0*r, scalar(1)), scalar(0));
                    wPar[j] =
                        lim*cdw[f]
                      + (1.0 - lim)*((phb[f] >= 0) ? 1.0 : 0.0);
                }
                offp += np;
            }
            off += np;
        }
    }

    // 경계 dev2 명시 플럭스 (gaussGrad 경계 보정 → X_b → Sf&X_b)
    {
        label off = 0;
        label offp = 0;
        forAll(U.boundaryField(), patchi)
        {
            const fvPatchVectorField& pp = U.boundaryField()[patchi];
            const label np = pp.size();
            if (np == 0) continue;

            const vectorField& Sfb = mesh.Sf().boundaryField()[patchi];
            const scalarField& mub = muEff.boundaryField()[patchi];

            if (pp.coupled())
            {
                // coupled 면: 양측 X = mu*dev2(T(gradU))를 선형 보간
                // (uDevTauFlux 내부면 산술 1:1)
                const scalarField cdw
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField muo
                (
                    muEff.boundaryField()[patchi].patchInternalField()
                );
                for (label f = 0; f < np; f++)
                {
                    tensor go, gn;
                    for (label m = 0; m < 9; m++)
                    {
                        go[m] = bGrad9[m*nbf + off + f];
                        gn[m] = gradN9[m*nPar + offp + f];
                    }
                    const tensor Xo(muo[f]*dev2(go.T()));
                    const tensor Xn(mub[f]*dev2(gn.T()));
                    const tensor Xf(cdw[f]*Xo + (1.0 - cdw[f])*Xn);
                    const vector flux = Sfb[f] & Xf;
                    for (label k = 0; k < 3; k++)
                    {
                        bFlux3[k*nbf + off + f] = flux[k];
                    }
                }
                offp += np;
            }
            else
            {
                const vectorField nHat(mesh.boundary()[patchi].nf());
                const vectorField snU(pp.snGrad());

                for (label f = 0; f < np; f++)
                {
                    tensor g;
                    for (label i = 0; i < 3; i++)
                    {
                        for (label j = 0; j < 3; j++)
                        {
                            g[3*i + j] =
                                bGrad9[(3*i + j)*nbf + off + f];
                        }
                    }
                    const vector n = nHat[f];
                    const vector nDotG = n & g;
                    const tensor gc(g - n*nDotG + n*snU[f]);
                    const tensor X(mub[f]*dev2(gc.T()));
                    const vector flux = Sfb[f] & X;
                    for (label k = 0; k < 3; k++)
                    {
                        bFlux3[k*nbf + off + f] = flux[k];
                    }
                }
            }
            off += np;
        }
    }

    if (rgpUEqnPrep2
        (
            p_.primitiveField().begin(), pB,
            muEff.primitiveField().begin(),
            phi.primitiveField().begin(), bFlux3
        ))
    {
        FatalErrorInFunction
            << "rgpUEqnPrep2: " << rgpPEqnLastError() << exit(FatalError);
    }

    // 경계 행렬 계수
    {
        label off = 0;
        label offp = 0;
        forAll(U.boundaryField(), patchi)
        {
            const fvPatchVectorField& pp = U.boundaryField()[patchi];
            const label np = pp.size();
            if (np == 0) continue;

            const scalarField& phb = phi.boundaryField()[patchi];

            if (pp.coupled())
            {
                // processor: div 가중치 = 벡터 limited(wPar),
                // laplacian은 deltaCoeffs 오버로드; mu는 면 보간.
                // bC는 인터페이스 계수(성분 공유) — gpuParB_
                scalarField wV(np);
                for (label f = 0; f < np; f++)
                {
                    wV[f] = wPar[offp + f];
                }
                const vectorField vic(pp.valueInternalCoeffs(wV));
                const vectorField vbc(pp.valueBoundaryCoeffs(wV));

                const scalarField pdc
                (
                    mesh.deltaCoeffs().boundaryField()[patchi]
                );
                const vectorField gic(pp.gradientInternalCoeffs(pdc));
                const vectorField gbc(pp.gradientBoundaryCoeffs(pdc));

                const scalarField cdw
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField muo
                (
                    muEff.boundaryField()[patchi].patchInternalField()
                );
                const scalarField& mun =
                    muEff.boundaryField()[patchi];
                const scalarField& msf =
                    mesh.magSf().boundaryField()[patchi];

                for (label f = 0; f < np; f++)
                {
                    const scalar gMsf =
                        (cdw[f]*muo[f] + (1.0 - cdw[f])*mun[f])*msf[f];
                    for (label k = 0; k < 3; k++)
                    {
                        bDiag3[k*nbf + off + f] =
                            phb[f]*vic[f][k] - gMsf*gic[f][k];
                        bSrc3[k*nbf + off + f] = 0;
                    }
                    gpuParB_[offp + f] =
                        -phb[f]*vbc[f][0] + gMsf*gbc[f][0];
                }
                offp += np;
            }
            else
            {
                const scalarField& wb =
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi];
                const vectorField vic(pp.valueInternalCoeffs(wb));
                const vectorField vbc(pp.valueBoundaryCoeffs(wb));
                const scalarField gMsf
                (
                    muEff.boundaryField()[patchi]
                   *mesh.magSf().boundaryField()[patchi]
                );
                const vectorField gic(pp.gradientInternalCoeffs());
                const vectorField gbc(pp.gradientBoundaryCoeffs());

                for (label f = 0; f < np; f++)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        bDiag3[k*nbf + off + f] =
                            phb[f]*vic[f][k] - gMsf[f]*gic[f][k];
                        bSrc3[k*nbf + off + f] =
                            -phb[f]*vbc[f][k] + gMsf[f]*gbc[f][k];
                    }
                }
            }
            off += np;
        }

        if (offp > 0 && rgpUEqnParCoeffs(gpuParB_.begin()))
        {
            FatalErrorInFunction
                << "rgpUEqnParCoeffs: " << rgpPEqnLastError()
                << exit(FatalError);
        }
    }

    const dictionary& sd = mesh.solution().solverDict
    (
        U.select
        (
            !mesh.schemes().steady()
         && solutionControl::finalIteration(mesh)
        )
    );
    const scalar tol = sd.lookup<scalar>("tolerance");
    const scalar rtol = sd.lookupOrDefault<scalar>("relTol", 0);
    const label maxIter = sd.lookupOrDefault<label>("maxIter", 1000);

    int solveCmpt[3];
    for (label k = 0; k < 3; k++)
    {
        solveCmpt[k] =
            (mesh.solutionD()[k] == 1 && pimple.momentumPredictor())
          ? 1 : 0;
    }

    const bool LTS = fv::localEulerDdt::enabled(mesh);

    double res0[3], resF[3];
    int iters[3];
    if (rgpUEqnSolve
        (
            LTS ? 1.0 : 1.0/runTime.deltaTValue(),
            LTS
          ? fv::localEulerDdt::localRDeltaT(mesh)
               .primitiveField().begin()
          : nullptr,
            rho.primitiveField().begin(),
            rho.oldTime().primitiveField().begin(),
            U3old, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            bDiag3, bSrc3, solveCmpt,
            tol, rtol, maxIter,
            U3out, res0, resF, iters
        ))
    {
        FatalErrorInFunction
            << "rgpUEqnSolve: " << rgpPEqnLastError() << exit(FatalError);
    }

    if (pimple.momentumPredictor())
    {
        vectorField& Uc = U.primitiveFieldRef();
        for (label i = 0; i < nc; i++)
        {
            for (label k = 0; k < 3; k++)
            {
                if (solveCmpt[k]) { Uc[i][k] = U3out[k*nc + i]; }
            }
        }
        for (label k = 0; k < 3; k++)
        {
            if (!solveCmpt[k]) continue;
            Info<< "rgpBiCGStab: Solving for U"
                << char('x' + k) << ", Initial residual = " << res0[k]
                << ", Final residual = " << resF[k]
                << ", No Iterations " << iters[k] << endl;
        }
        U.correctBoundaryConditions();
        fvConstraints().constrain(U);
        K = 0.5*magSqr(U);
    }
}


// ************************************************************************* //
