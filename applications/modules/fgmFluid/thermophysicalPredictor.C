/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2024 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "fgmFluid.H"
#include "fvcDdt.H"
#include "fvcDiv.H"
#include "fvmSup.H"
#include "fvmLaplacian.H"
#include "fvcGrad.H"
#include "zeroGradientFvPatchFields.H"
#include "solutionControl.H"
#include "localEulerDdtScheme.H"
#include "processorFvPatch.H"
#include "PstreamBuffers.H"
#include "snGradScheme.H"
#include "gpu/rgpPEqnTypes.H"

#include <chrono>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::armGpuSTMesh()
{
    // pEqn 메시 아밍이 토폴로지 변경을 감지하면 gpuZCArmed_를 리셋함
    armGpuPEqnMesh();

    if (gpuZCArmed_) return;

    const label nif = mesh.owner().size();
    label nbf = 0;
    forAll(Z_.boundaryField(), patchi)
    {
        nbf += Z_.boundaryField()[patchi].size();
    }

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
    label off = 0;
    forAll(Z_.boundaryField(), patchi)
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
    gpuZCArmed_ = true;
    Info<< "fgmFluid: GPU transport mesh armed" << nl << endl;
}


void Foam::solvers::fgmFluid::thermophysicalPredictor()
{
    std::chrono::steady_clock::time_point tTot;
    if (thermoTimings_) { tTot = std::chrono::steady_clock::now(); }

    tmp<fv::convectionScheme<scalar>> mvConvection;
    if (!gpuZC_ || gpuPEqnCheck_)   // check 모드: CPU 대조용으로도 구성
    {
        mvConvection =
            fv::convectionScheme<scalar>::New
            (
                mesh,
                fields,
                phi,
                mesh.schemes().div("div(phi,Yi_h)")
            );
    }

    // ── 병렬 gpuZC: processor 면의 multivariate 경계 리미터/가중치 ──
    // (LimitedScheme.C coupled 분기 1:1 — 이웃 grad는 랭크 간 교환)
    label nPar = 0;
    if (gpuZC_ && Pstream::parRun())
    {
        forAll(Z_.boundaryField(), patchi)
        {
            if (Z_.boundaryField()[patchi].coupled())
            {
                nPar += Z_.boundaryField()[patchi].size();
            }
        }
    }
    List<double> procLim(nPar, 1.0);
    List<double> procW(nPar, 0.0);

    // 3성분 벡터의 proc-패치 교환 (SoA [3*nPar] ↔ 이웃 랭크)
    auto exchangeParVec3 = [&](const List<double>& s, List<double>& r)
    {
        PstreamBuffers pBufs(Pstream::commsTypes::nonBlocking);
        label offp = 0;
        forAll(Z_.boundaryField(), patchi)
        {
            if (!Z_.boundaryField()[patchi].coupled()) continue;
            const processorFvPatch& pp =
                refCast<const processorFvPatch>(mesh.boundary()[patchi]);
            const label np = pp.size();
            List<vector> sv(np);
            for (label k = 0; k < np; k++)
            {
                sv[k] = vector
                (
                    s[offp + k], s[nPar + offp + k], s[2*nPar + offp + k]
                );
            }
            UOPstream toNbr(pp.neighbProcNo(), pBufs);
            toNbr << sv;
            offp += np;
        }
        pBufs.finishedSends();
        offp = 0;
        forAll(Z_.boundaryField(), patchi)
        {
            if (!Z_.boundaryField()[patchi].coupled()) continue;
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

    if (gpuZC_)
    {
        // ZCGPU: multivariate limitedLinear 가중치를 mvConvection 생성
        // 시점(= manifold 갱신 전 필드 값)에 GPU에서 준비. limiter =
        // fields 테이블(he, Z, C[, h, W]) 각 필드 limiter의 min.
        armGpuSTMesh();

        if (rgpSTWeightsBegin(phi.primitiveField().begin()))
        {
            FatalErrorInFunction
                << "rgpSTWeightsBegin: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        label nbfW = 0;
        forAll(Z_.boundaryField(), patchi)
        {
            nbfW += Z_.boundaryField()[patchi].size();
        }
        gpuZCBuf_.setSize(3*nbfW);

        auto addWeightField = [&](const volScalarField& f)
        {
            double* bPsiA = gpuZCBuf_.begin();
            label off = 0;
            forAll(f.boundaryField(), patchi)
            {
                const fvPatchScalarField& pf = f.boundaryField()[patchi];
                if (pf.coupled())
                {
                    // gaussGrad coupled 규약: 경계 기여는 면 값
                    // w*psi_own + (1-w)*psi_nei (pf가 이웃 셀 값 보유)
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
                    << rgpPEqnLastError()
                    << exit(FatalError);
            }

            // 병렬: 이 필드의 proc-면 limiter를 min-누적
            // (LimitedScheme.C coupled 분기 1:1 — NVDTVD::r + limitedLinear)
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
        // 수송 스칼라 목록: Z, C [+ h(비단열) + W(희석)] — h/W는 Z와
        // 동일한 이류-확산 기계(소스 없음, gamma만 상이)라 같은 배치로
        DynamicList<word> zcNames;
        zcNames.append(Z_.name());
        zcNames.append(C_.name());
        if (hPtr_.valid()) { zcNames.append(hPtr_().name()); }
        if (WPtr_.valid()) { zcNames.append(WPtr_().name()); }

        forAll(zcNames, zi)
        {
            const word& nm = zcNames[zi];
            if
            (
                mesh.solution().relaxEquation(nm)
             && mesh.solution().equationRelaxationFactor(nm) != 1
            )
            {
                FatalErrorInFunction
                    << "gpuZC (v1) does not support equation relaxation "
                    << "on " << nm << exit(FatalError);
            }
            if (fvModels().addsSupToField(nm))
            {
                FatalErrorInFunction
                    << "gpuZC (v1) does not support fvModels sources on "
                    << nm << exit(FatalError);
            }
            // 행렬-레벨 fvConstraint 차단 (CPU 경로는 constrain(Eqn)을
            // 적용 — GPU 조립은 미반영이므로 켜져 있으면 침묵 발산)
            checkGpuConstraints(nm, "gpuZC");
        }
        if (Pstream::parRun() && gpuPEqnCheck_)
        {
            FatalErrorInFunction
                << "gpuPEqnCheck is serial-only"
                << exit(FatalError);
        }

        addWeightField(thermo.he());
        addWeightField(Z_);
        addWeightField(C_);
        if (hPtr_.valid()) { addWeightField(hPtr_()); }
        if (WPtr_.valid()) { addWeightField(WPtr_()); }

        if (rgpSTWeightsEnd())
        {
            FatalErrorInFunction
                << "rgpSTWeightsEnd: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        // 병렬: proc-면 최종 가중치 (multivariateScheme 규약 1:1 —
        // w = lim*CD + (1-lim)*upwind)
        if (nPar > 0)
        {
            label offp = 0;
            forAll(Z_.boundaryField(), patchi)
            {
                if (!Z_.boundaryField()[patchi].coupled()) continue;
                const label np = Z_.boundaryField()[patchi].size();
                const scalarField cdw
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField& phb = phi.boundaryField()[patchi];
                for (label k = 0; k < np; k++)
                {
                    const double lim = procLim[offp + k];
                    procW[offp + k] =
                        lim*cdw[k]
                      + (1.0 - lim)*((phb[k] >= 0.0) ? 1.0 : 0.0);
                }
                offp += np;
            }
        }
    }

    // --- FGM manifold update: gZ, composition Y_k, PV source (from Z, C) ---
    {
        const auto tManifold0 = std::chrono::steady_clock::now();
        updateManifold();
        if (thermoTimings_)
        {
            Info<< "manifold update (incl. he re-seed) = "
                << std::chrono::duration<double>
                   (
                       std::chrono::steady_clock::now() - tManifold0
                   ).count() << " s" << endl;
        }
    }
    // NOTE: thermo_.normaliseY() REMOVED -- with every specie marked inactive
    // it sets the default specie (N2) = 1 - sum(ACTIVE) = 1 (no active species),
    // spuriously diluting the tabulated composition (which already sums to 1,
    // with N2 = 0 from the table) by ~50% N2 after the mixture renormalises.
    // The FGM manifold already delivers a normalised composition, so the call
    // is unnecessary and corrupts it.

    // --- Localized Artificial Diffusivity (LAD) -----------------------------
    // Smear the steep transcritical density interface (recess-tip LOx/gas, rho
    // jump ~60x at x~25 mm) with a density-gradient-sensed artificial mass
    // diffusivity, damping at source the spurious pressure/velocity overshoots
    // that throttle the time step (diagnosed bottleneck). Kawai, Terashima &
    // Negishi, J. Comput. Phys. 300:116-135 (2015). Sized to the local cell,
    //   Dart = LADCoeff * V^(2/3) * |U| * |grad(rho)|   [kg/(m s)],
    // so the artificial diffusive time step stays of order the convective one
    // and smooth regions (|grad(rho)| ~ 0) are untouched. LADCoeff is read
    // from the PIMPLE dict each step (runTimeModifiable); default 0 = LAD off.
    // The same Dart is added to the Z, C and h diffusivities so the manifold
    // coordinates stay mutually consistent under the smoothing.
    const scalar LADCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADCoeff", scalar(0))
    );
    volScalarField Dart
    (
        IOobject
        (
            "Dart",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1, -1, -1, 0, 0, 0, 0), 0),
        zeroGradientFvPatchScalarField::typeName
    );
    if (LADCoeff > 0)
    {
        const scalarField V23(pow(scalarField(mesh.V()), 2.0/3.0));
        Dart.primitiveFieldRef() =
            LADCoeff*V23
           *mag(U_)().primitiveField()
           *mag(fvc::grad(rho))().primitiveField();
        Dart.correctBoundaryConditions();
        Info<< "LAD: Dart max = " << gMax(Dart.primitiveField())
            << " kg/(m s)" << endl;
    }

    // Continuity-error compensation for the manifold scalars (Z, C, h).
    // The conservative ddt(rho,phi)+div(phi,phi) pair rebinds any LOCAL
    // continuity residual into the transported scalar: at a flame-zone
    // pressure spike the step-to-step density churn drives h (and Z, C)
    // out of physical bounds (observed: h to the manifold dh floor -> T
    // pinned at the 80 K table edge -> rho ~1300 pockets beside the flame
    // -> the density contrast re-excites the spike). Subtracting
    // phi*(ddt(rho)+div(phi)) turns the equations into the bounded pure-
    // advection form rho*Dphi/Dt (identical when continuity is satisfied,
    // bounded when it is not) -- the standard bounded-transport correction.
    // Switch 'contErrCompensation' (PIMPLE dict, default on; read each step).
    // NOTE: a DISCONTINUOUS field seed (e.g. an igniter kernel written by
    // setFields) creates a locally huge contErr for a few steps; the Sp term
    // then dominates/flips the matrix diagonal and wipes the seeded scalar.
    // Disable while seeding, or keep off if it proves harmful.
    const Switch contErrComp
    (
        pimple.dict().lookupOrDefault<Switch>("contErrCompensation", true)
    );
    const volScalarField contErr
    (
        "contErr",
        contErrComp
      ? (fvc::ddt(rho) + fvc::div(phi))()
      : volScalarField::New
        (
            "zeroContErr", mesh, dimensionedScalar(dimDensity/dimTime, 0)
        )()
    );

    // --- Mixture-fraction transport (conserved scalar, no source) ---
    std::chrono::steady_clock::time_point tZC;
    if (thermoTimings_) { tZC = std::chrono::steady_clock::now(); }

    // ZCGPU 공통 솔브: 조립(ddt+div[준비된 가중치]+Sp+laplacian)과
    // Jacobi-BiCGStab를 디바이스에서. 경계 기여는 fvPatchField API.
    auto solveScalarGpu = [&]
    (
        volScalarField& psiF,
        const volScalarField& gamma,
        const volScalarField& sp,
        const volScalarField* srcPtr
    )
    {
        // CPU 규약: fvMatrix 생성자의 BC updateCoeffs() 상응 (phi
        // 의존 BC의 valueFraction 갱신 — 직접 조립 경로 필수)
        psiF.boundaryFieldRef().updateCoeffs();

        // 비직교: 면 계수는 gg(=magSf·scheme.deltaCoeffs, 아밍 시
        // 업로드)로 이미 반영 — 여기서는 coupled 경계 pdc와
        // 명시 보정 소스(gaussLaplacianScheme 1:1: −fvm::laplacian이
        // 방정식에 −로 들어가므로 src += div(ΓmagSf·correction(ψ)))
        tmp<fv::snGradScheme<scalar>> tsnZC
        (
            fv::snGradScheme<scalar>::New
            (
                mesh, mesh.schemes().snGrad(psiF.name())
            )
        );
        const surfaceScalarField dcsZC(tsnZC().deltaCoeffs(psiF));

        tmp<volScalarField> tCorrDiv;
        if (gpuNonOrtho_)
        {
            const surfaceScalarField gammaMagSf
            (
                fvc::interpolate(gamma)*mesh.magSf()
            );
            tCorrDiv =
                fvc::div(gammaMagSf*tsnZC().correction(psiF));
        }

        label nbf = 0;
        forAll(psiF.boundaryField(), patchi)
        {
            nbf += psiF.boundaryField()[patchi].size();
        }
        gpuZCBuf_.setSize(3*nbf);
        double* bDiagA = gpuZCBuf_.begin();
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
                // processor 패치: div의 가중치는 multivariate limited
                // 가중치(procW — CPU mvConvection과 동일), laplacian은
                // deltaCoeffs 오버로드. iC는 diag로, bC(= -phb*vbc +
                // gMsf*gbc — uncoupled의 소스 공식과 동일한 양)는
                // 인터페이스 계수로 (소스에 접지 않음).
                scalarField wP(np);
                for (label k = 0; k < np; k++)
                {
                    wP[k] = procW[offPar + k];
                }
                const scalarField vic(pp.valueInternalCoeffs(wP));
                const scalarField vbc(pp.valueBoundaryCoeffs(wP));

                const scalarField pdc
                (
                    dcsZC.boundaryField()[patchi]
                );
                const scalarField gic(pp.gradientInternalCoeffs(pdc));
                const scalarField gbc(pp.gradientBoundaryCoeffs(pdc));

                // CPU interpolate(gamma) coupled = w*own + (1-w)*nei
                const scalarField cdw
                (
                    mesh.surfaceInterpolation::weights()
                        .boundaryField()[patchi]
                );
                const scalarField gOwn
                (
                    gamma.boundaryField()[patchi].patchInternalField()
                );
                const scalarField& gNei = gamma.boundaryField()[patchi];
                const scalarField& msf =
                    mesh.magSf().boundaryField()[patchi];

                for (label k = 0; k < np; k++)
                {
                    const scalar gMsf =
                        (cdw[k]*gOwn[k] + (1.0 - cdw[k])*gNei[k])*msf[k];
                    bDiagA[off + k] = phb[k]*vic[k] - gMsf*gic[k];
                    bSrcA[off + k] = 0;
                    bPsiA[off + k] = pp[k];
                    gpuPEqnParB_[offPar + k] =
                        -phb[k]*vbc[k] + gMsf*gbc[k];
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
            }
            off += np;
        }

        // 병렬: 이 방정식의 인터페이스 계수 업로드 (gamma가 방정식마다
        // 달라 솔브 직전 갱신 — pEqn과 gPar.bC 버퍼 공유)
        if (offPar > 0 && rgpPEqnParCoeffs(gpuPEqnParB_.begin()))
        {
            FatalErrorInFunction
                << "rgpPEqnParCoeffs: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        // fvMatrix::solve(word) 규약과 1:1 — transient 최종 이터레이션이면
        // "YiFinal" 딕셔너리 선택 (CPU 경로 ZEqn.solve("Yi")와 동일)
        // 명시 소스 결합 (반응 소스 + 비직교 보정)
        scalarField srcTot;
        if (tCorrDiv.valid())
        {
            srcTot =
                srcPtr
              ? scalarField(srcPtr->primitiveField())
              : scalarField(mesh.nCells(), 0.0);
            srcTot += tCorrDiv().primitiveField();
        }

        const dictionary& sd = mesh.solution().solverDict
        (
            !mesh.schemes().steady()
         && solutionControl::finalIteration(mesh)
          ? word("YiFinal")
          : word("Yi")
        );
        const scalar tol = sd.lookup<scalar>("tolerance");
        const scalar rtol = sd.lookupOrDefault<scalar>("relTol", 0);
        const label maxIter = sd.lookupOrDefault<label>("maxIter", 1000);

        double res0 = 0, resF = 0;
        int nIter = 0;
        const int rc = rgpSTEqnSolve
        (
            LTS ? 1.0 : 1.0/runTime.deltaTValue(),
            LTS
          ? fv::localEulerDdt::localRDeltaT(mesh)
               .primitiveField().begin()
          : nullptr,
            (srcPtr || tCorrDiv.valid()) ? 1 : 0,
            rho.primitiveField().begin(),
            rho.oldTime().primitiveField().begin(),
            psiF.oldTime().primitiveField().begin(),
            psiF.primitiveField().begin(),
            phi.primitiveField().begin(),
            gamma.primitiveField().begin(),
            nullptr,   // gammaFace (Fickian 추출 경로 미사용)
            sp.primitiveField().begin(),
            srcTot.size() ? srcTot.begin()
          : (srcPtr ? srcPtr->primitiveField().begin() : nullptr),
            bPsiA, bDiagA, bSrcA,
            tol, rtol, maxIter,
            psiF.primitiveFieldRef().begin(),
            &res0, &resF, &nIter
        );
        if (rc)
        {
            FatalErrorInFunction
                << "rgpSTEqnSolve(" << psiF.name() << "): "
                << rgpPEqnLastError()
                << exit(FatalError);
        }

        Info<< "rgpBiCGStab: Solving for " << psiF.name()
            << ", Initial residual = " << res0
            << ", Final residual = " << resF
            << ", No Iterations " << nIter << endl;

        psiF.correctBoundaryConditions();
    };

    // ZCGPU 검증: CPU fvMatrix와 계수 대조 (gpuPEqnCheck 재사용)
    auto checkSTMatrix = [&]
    (
        fvScalarMatrix& chk, const volScalarField& psiF,
        const scalarField& psi0
    )
    {
        const label ncc = mesh.nCells();
        const label nif = mesh.owner().size();
        scalarField diagC(chk.diag());
        scalarField srcC(chk.source());
        forAll(psiF.boundaryField(), patchi)
        {
            const labelUList& fc = mesh.boundary()[patchi].faceCells();
            const scalarField& iC = chk.internalCoeffs()[patchi];
            const scalarField& bC = chk.boundaryCoeffs()[patchi];
            forAll(fc, k)
            {
                diagC[fc[k]] += iC[k];
                srcC[fc[k]] += bC[k];
            }
        }
        List<double> dG(ncc), uG(nif), lG(nif), bG(ncc);
        rgpSTEqnDump(dG.begin(), uG.begin(), lG.begin(), bG.begin(),
                     nullptr);
        scalar dM = 0, uM = 0, lM = 0, bM = 0;
        forAll(diagC, i)
        {
            dM = max(dM, mag(dG[i] - diagC[i])/max(mag(diagC[i]), small));
            bM = max(bM, mag(bG[i] - srcC[i])/max(mag(srcC[i]), small));
        }
        const scalarField& uC = chk.upper();
        const scalarField& lC = chk.lower();
        forAll(uC, f)
        {
            uM = max(uM, mag(uG[f] - uC[f])/max(mag(uC[f]), small));
            lM = max(lM, mag(lG[f] - lC[f])/max(mag(lC[f]), small));
        }
        Info<< "gpuZCCheck(" << psiF.name() << "): maxRel diag = " << dM
            << ", upper = " << uM << ", lower = " << lM
            << ", source = " << bM << endl;

        // CPU 공식 그대로 초기 잔차 재계산 (pre-solve psi0 기준)
        scalarField Apsi(ncc);
        scalarField rs(ncc);
        forAll(Apsi, i) { Apsi[i] = diagC[i]*psi0[i]; rs[i] = diagC[i]; }
        forAll(mesh.owner(), f)
        {
            const label o = mesh.owner()[f], n = mesh.neighbour()[f];
            Apsi[o] += uC[f]*psi0[n];
            Apsi[n] += lC[f]*psi0[o];
            rs[o] += uC[f];
            rs[n] += lC[f];
        }
        const scalar xRef = gAverage(psi0);
        scalar nF = 0, num = 0;
        forAll(Apsi, i)
        {
            nF += mag(Apsi[i] - rs[i]*xRef) + mag(srcC[i] - rs[i]*xRef);
            num += mag(srcC[i] - Apsi[i]);
        }
        nF += 1e-20;
        Info<< "gpuZCCheck(" << psiF.name() << "): CPU-formula initRes = "
            << num/nF << " (normFactor " << nF << ", |rA| " << num << ")"
            << endl;
    };

    if (gpuZC_)
    {
        const volScalarField DZ("DZ", Deff("Z"));
        const volScalarField DZeff(DZ + Dart);
        const scalarField Z0(gpuPEqnCheck_ ? Z_.primitiveField() : scalarField());
        solveScalarGpu(Z_, DZeff, contErr, nullptr);
        if (gpuPEqnCheck_)
        {
            fvScalarMatrix chk
            (
                fvm::ddt(rho, Z_)
              + mvConvection->fvmDiv(phi, Z_)
              - fvm::Sp(contErr, Z_)
              - fvm::laplacian(DZeff, Z_)
            );
            checkSTMatrix(chk, Z_, Z0);
        }
        fvConstraints().constrain(Z_);

        Z_ = max(min(Z_, scalar(1)), scalar(0));
    }
    else
    {
        const volScalarField DZ("DZ", Deff("Z"));
        fvScalarMatrix ZEqn
        (
            fvm::ddt(rho, Z_)
          + mvConvection->fvmDiv(phi, Z_)
          - fvm::Sp(contErr, Z_)
          - fvm::laplacian(DZ + Dart, Z_)
         ==
            fvModels().source(rho, Z_)
        );

        ZEqn.relax();
        fvConstraints().constrain(ZEqn);
        ZEqn.solve("Yi");
        fvConstraints().constrain(Z_);

        Z_ = max(min(Z_, scalar(1)), scalar(0));
    }

    // --- Progress-variable transport (source = rho*omega_C from table) ---
    if (gpuZC_)
    {
        const volScalarField DC("DC", Deff("C"));
        const volScalarField DCeff(DC + Dart);
        const scalarField C0(gpuPEqnCheck_ ? C_.primitiveField() : scalarField());
        solveScalarGpu(C_, DCeff, contErr, &sourcePV_);
        if (gpuPEqnCheck_)
        {
            fvScalarMatrix chk
            (
                fvm::ddt(rho, C_)
              + mvConvection->fvmDiv(phi, C_)
              - fvm::Sp(contErr, C_)
              - fvm::laplacian(DCeff, C_)
             ==
                sourcePV_
            );
            checkSTMatrix(chk, C_, C0);
        }
        fvConstraints().constrain(C_);

        if (thermoTimings_)
        {
            Info<< "Z+C transport total = "
                << std::chrono::duration<double>
                   (std::chrono::steady_clock::now() - tZC).count()
                << " s" << endl;
        }

        C_ = max(min(C_, scalar(1)), scalar(0));
    }
    else
    {
        const volScalarField DC("DC", Deff("C"));
        fvScalarMatrix CEqn
        (
            fvm::ddt(rho, C_)
          + mvConvection->fvmDiv(phi, C_)
          - fvm::Sp(contErr, C_)
          - fvm::laplacian(DC + Dart, C_)
         ==
            sourcePV_
          + fvModels().source(rho, C_)
        );

        CEqn.relax();
        fvConstraints().constrain(CEqn);
        {
            std::chrono::steady_clock::time_point tS;
            if (thermoTimings_) { tS = std::chrono::steady_clock::now(); }
            CEqn.solve("Yi");
            if (thermoTimings_)
            {
                Info<< "CEqn solve-only = "
                    << std::chrono::duration<double>
                       (std::chrono::steady_clock::now() - tS).count()
                    << " s" << endl;
            }
        }
        fvConstraints().constrain(C_);

        if (thermoTimings_)
        {
            Info<< "Z+C transport total = "
                << std::chrono::duration<double>
                   (std::chrono::steady_clock::now() - tZC).count()
                << " s" << endl;
        }

        // The transported progress variable is the NORMALIZED c in [0, 1]
        // (table closure: omega = 0 at c = 1, the equilibrium/envelope
        // boundary). The explicit source can step THROUGH the c = 1 zero in
        // a single dt (omega ~ 1e4 1/s), so clamp both ends -- the standard
        // practice layered on top of the equilibrium closure (Pierce 2004's
        // library truncation; cf. solver-side bounding in tabulated codes).
        C_ = max(min(C_, scalar(1)), scalar(0));
    }

    // --- Total-enthalpy transport (non-adiabatic FPV, method b) ---
    // Total (absolute) enthalpy is a CONSERVED scalar (no reaction source:
    // chemical heat release is internal to h), so this is a plain advection-
    // diffusion equation -- NOT the unstable he equation. It carries the cold-
    // inlet / heat-loss enthalpy in from the boundaries; the manifold lookup
    // then reads T(Z,gZ,c,dh) at the local defect dh = h - h_ad(Z). T is taken
    // from the table (not inverted from h), so the he<->T drift that forbids an
    // EEqn does not occur. Solved only when the table carries an enthalpy axis.
    if (fgmTable_.useEnthalpy())
    {
        volScalarField& h = hPtr_();
        const volScalarField Dh("Dh", Deff("h"));

        if (gpuZC_)
        {
            const volScalarField Dheff(Dh + Dart);
            const scalarField h0
            (
                gpuPEqnCheck_ ? h.primitiveField() : scalarField()
            );
            solveScalarGpu(h, Dheff, contErr, nullptr);
            if (gpuPEqnCheck_)
            {
                fvScalarMatrix chk
                (
                    fvm::ddt(rho, h)
                  + mvConvection->fvmDiv(phi, h)
                  - fvm::Sp(contErr, h)
                  - fvm::laplacian(Dheff, h)
                );
                checkSTMatrix(chk, h, h0);
            }
            fvConstraints().constrain(h);
        }
        else
        {
        fvScalarMatrix hEqn
        (
            fvm::ddt(rho, h)
          + mvConvection->fvmDiv(phi, h)
          - fvm::Sp(contErr, h)
          - fvm::laplacian(Dh + Dart, h)
         ==
            fvModels().source(rho, h)
        );

        hEqn.relax();
        fvConstraints().constrain(hEqn);
        hEqn.solve("Yi");
        fvConstraints().constrain(h);
        }

        // Bound h to the manifold-representable enthalpy band
        // [hAd(Z)+dh_min, hAd(Z)+dh_max] -- the same table-consistency
        // bounding applied to C above. Outside the band T/composition already
        // clamp at the table edge, but the TRANSPORTED h keeps drifting: at a
        // flame-zone pressure spike the step-to-step density churn makes the
        // conservative ddt(rho,h) rebind mass and corrupt h (observed 2026-07:
        // h below the dh floor -> T pinned at the 80 K table edge -> EOS rho
        // ~1300 pockets BESIDE ~100 kg/m3 flame gas at the faceplate wall ->
        // the density contrast re-excites the pressure spike, closing a
        // self-sustaining loop). Bounding h severs that feedback leg.
        {
            const List<scalar>& dhAxis = fgmTable_.chiAxis();
            const scalar dhMin = min(dhAxis.first(), dhAxis.last());
            const scalar dhMax = max(dhAxis.first(), dhAxis.last());
            const scalar hOxb = fgmTable_.hOx();
            const scalar hFuelb = fgmTable_.hFuel();

            scalarField& hc = h.primitiveFieldRef();
            const scalarField& Zbnd = Z_.primitiveField();
            forAll(hc, celli)
            {
                const scalar Zi = max(min(Zbnd[celli], scalar(1)), scalar(0));
                const scalar hAd = (scalar(1) - Zi)*hOxb + Zi*hFuelb;
                hc[celli] = max(min(hc[celli], hAd + dhMax), hAd + dhMin);
            }
            h.correctBoundaryConditions();
        }
    }

    // Steam-dilution scalar W is a CONSERVED mixing scalar (injected steam is
    // inert as a stream tracer -- combustion-produced H2O is NOT counted in
    // this v1 model), so it advects/diffuses like Z with no source. It carries
    // the local steam-in-oxidiser fraction in from the steam inlet; the
    // manifold's 4th axis is then queried at the local W. Solved only for a
    // steam-diluted (dilution) table.
    if (fgmTable_.useDilution())
    {
        volScalarField& W = WPtr_();
        const volScalarField DW("DW", Deff("Z"));   // W diffuses like Z

        if (gpuZC_)
        {
            const volScalarField DWeff(DW + Dart);
            const scalarField W0
            (
                gpuPEqnCheck_ ? W.primitiveField() : scalarField()
            );
            solveScalarGpu(W, DWeff, contErr, nullptr);
            if (gpuPEqnCheck_)
            {
                fvScalarMatrix chk
                (
                    fvm::ddt(rho, W)
                  + mvConvection->fvmDiv(phi, W)
                  - fvm::Sp(contErr, W)
                  - fvm::laplacian(DWeff, W)
                );
                checkSTMatrix(chk, W, W0);
            }
            fvConstraints().constrain(W);
        }
        else
        {
        fvScalarMatrix WEqn
        (
            fvm::ddt(rho, W)
          + mvConvection->fvmDiv(phi, W)
          - fvm::Sp(contErr, W)
          - fvm::laplacian(DW + Dart, W)
         ==
            fvModels().source(rho, W)
        );

        WEqn.relax();
        fvConstraints().constrain(WEqn);
        WEqn.solve("Yi");
        fvConstraints().constrain(W);
        }

        // Bound W to the tabulated dilution range (outside it the manifold
        // clamps at the edge slice anyway; keep the transported field in-band).
        {
            const List<scalar>& Wax = fgmTable_.chiAxis();
            const scalar Wmin = min(Wax.first(), Wax.last());
            const scalar Wmax = max(Wax.first(), Wax.last());
            scalarField& Wc = W.primitiveFieldRef();
            forAll(Wc, celli)
            {
                Wc[celli] = max(min(Wc[celli], Wmax), Wmin);
            }
            W.correctBoundaryConditions();
        }
    }


    // --- No transported energy equation (adiabatic FPV/FGM closure) ---
    // The thermochemical state (T, he, Y) is a tabulated function of the
    // manifold coordinates and was set in updateManifold() above: he has been
    // re-seeded to he(p, T_table) on the looked-up composition. We therefore
    // do NOT advance an EEqn for he here. The reason is essential, not an
    // optimisation: under fully-compressible acoustics a transported he is
    // perturbed off the manifold (the pressure-work/dilatation and convective
    // fluxes inject/remove energy that the algebraic composition update does
    // not see), so the he->T inversion drifts and the strained flame
    // eventually collapses. flameletFoam/FPVFoam transport only Z and c and
    // read T/he from the table for exactly this reason.
    //
    // thermo.correct() inverts the manifold he straight back to T_table and
    // refreshes rho, psi, mu, ... at the new (p, T, Y) for the pressure solve.
    // thermoGPU: the same property refresh, batched on the CUDA device (the
    // he->T inversion is skipped -- T stays the manifold temperature).
    {
        const auto tRefresh0 = std::chrono::steady_clock::now();

        if (gpuThermo_)
        {
            gpuThermoCorrect();
        }
        else
        {
            thermo_.correct();
        }

        if (thermoTimings_)
        {
            const scalar dtRefresh =
                std::chrono::duration<double>
                (
                    std::chrono::steady_clock::now() - tRefresh0
                ).count();
            Info<< "thermo refresh ("
                << (gpuThermo_ ? "GPU" : "CPU") << ") = "
                << dtRefresh << " s" << endl;
        }
    }

    if (thermoTimings_)
    {
        Info<< "thermophysical predictor total = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tTot).count()
            << " s" << endl;
    }
}


// ************************************************************************* //
