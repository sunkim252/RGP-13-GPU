/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2022-2025 OpenFOAM Foundation
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
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvcGrad.H"
#include "fvcDiv.H"
#include "zeroGradientFvPatchFields.H"
#include "limitedSurfaceInterpolationScheme.H"
#include "solutionControl.H"
#include "processorFvPatch.H"
#include "PstreamBuffers.H"
#include "localEulerDdtScheme.H"
#include "snGradScheme.H"
#include "fvcDiv.H"
#include "gpu/rgpPEqnTypes.H"

#include <chrono>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::momentumPredictor()
{
    std::chrono::steady_clock::time_point tTot;
    if (thermoTimings_) { tTot = std::chrono::steady_clock::now(); }

    volVectorField& U(U_);

    // --- Localized Artificial (shear) Viscosity on the momentum -------------
    // Companion to the scalar mass-diffusivity LAD: damps at source the
    // spurious VELOCITY overshoot at the transcritical density interface
    // (diagnosed recess-tip |U| spike, limitMag-capped at 500, that throttles
    // the time step). A density-gradient-sensed artificial dynamic viscosity
    //   muArt = LADUCoeff * V^(2/3) * |U| * |grad(rho)|   [kg/(m s)]
    // is added to the viscous stress (-fvm::laplacian(muArt, U)); cell-sized so
    // its own viscous time step stays of order the convective one and smooth
    // regions (|grad(rho)| ~ 0) are untouched -- the LES subgrid stress is
    // unaffected away from the interface. Cook & Cabot, J. Comput. Phys. 195
    // (2004); Kawai & Lele, J. Comput. Phys. 227 (2008); Kawai, Terashima &
    // Negishi, J. Comput. Phys. 300 (2015). LADUCoeff read from the PIMPLE
    // dict each step (runTimeModifiable); default 0 = off.
    const scalar LADUCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADUCoeff", scalar(0))
    );
    volScalarField muArt
    (
        IOobject
        (
            "muArt",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimensionSet(1, -1, -1, 0, 0, 0, 0), 0),
        zeroGradientFvPatchScalarField::typeName
    );
    if (LADUCoeff > 0)
    {
        const scalarField V23(pow(scalarField(mesh.V()), 2.0/3.0));
        muArt.primitiveFieldRef() =
            LADUCoeff*V23
           *mag(U)().primitiveField()
           *mag(fvc::grad(rho))().primitiveField();
        muArt.correctBoundaryConditions();
        Info<< "LAD-U: muArt max = " << gMax(muArt.primitiveField())
            << " kg/(m s)" << endl;
    }

    // Artificial BULK viscosity (Cook-Cabot) -- damps the DILATATIONAL /
    // compressive part directly, targeting the injector pressure oscillation
    // (p +/- spike and |U| overshoot at the fine tangential-hole cells) that
    // the shear muArt and the Rhie-Chow fixes do not reach. Dilatation-sensed
    //   betaArt = LADbulkCoeff * rho * V^(2/3) * |div U|   [kg/(m s)]
    // added as -grad(betaArt*div(U)) to the momentum. Cook & Cabot, J. Comput.
    // Phys. 195 (2004) 594; Kawai, Terashima & Negishi, J. Comput. Phys. 300
    // (2015) 116. Read each step (runTimeModifiable); default 0 = off.
    const scalar LADbulkCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADbulkCoeff", scalar(0))
    );

    std::chrono::steady_clock::time_point tAsm;
    if (thermoTimings_) { tAsm = std::chrono::steady_clock::now(); }

    if (gpuUEqn_)
    {
        // --- UEqnGPU: 물리 필드(가중치·muEff·dev2 명시항·grad p)는 OF
        // 스킴으로 호스트가 정확 계산 → GPU가 LDU 조립+성분별 BiCGStab.
        // 행렬은 스텝 내내 디바이스 상주(pCorr가 rAU/HbyA를 GPU 산출).
        if (MRF.size() > 0)
        {
            FatalErrorInFunction
                << "gpuUEqn (v1) does not support MRF"
                << exit(FatalError);
        }
        // 침묵 오차 방지: CPU 경로의 relax/fvModels/fvConstraints(행렬)
        // 는 GPU 조립에 미반영 — U에 실제로 작용하는 것만 차단 (계수 1의
        // no-op relax, 다른 필드의 모델, 필드-레벨 제약은 무해하므로 허용)
        // 방정식 완화: fvMatrix::relax(alpha) 1:1을 디바이스에서 수행
        // (rgpUEqnSolve relaxAlpha/bRelax3 — 완화된 행렬이 pCorr의
        // rAU/H에도 그대로 흘러가는 CPU 규약 유지). 계수는
        // relaxationFactor()의 "<name>Final"→평이름 fallback (α==1 스킵)
        scalar relaxAlpha = -1;
        {
            const word fname(U.name() + "Final");
            if
            (
                solutionControl::finalIteration(mesh)
             && mesh.solution().relaxEquation(fname)
            )
            {
                relaxAlpha =
                    mesh.solution().equationRelaxationFactor(fname);
            }
            else if (mesh.solution().relaxEquation(U.name()))
            {
                relaxAlpha =
                    mesh.solution().equationRelaxationFactor(U.name());
            }
            if (relaxAlpha == 1) { relaxAlpha = -1; }
        }
        if (fvModels().addsSupToField(U.name()))
        {
            FatalErrorInFunction
                << "gpuUEqn (v1) does not support fvModels sources on U"
                << exit(FatalError);
        }
        checkGpuConstraints(U.name(), "gpuUEqn");

        armGpuSTMesh();

        const label nc = mesh.nCells();
        label nbf = 0;
        forAll(U.boundaryField(), patchi)
        {
            nbf += U.boundaryField()[patchi].size();
        }

        // muEff = (rho*nuEff) — dev2 명시항용. 라플라시안 계수는
        // muEff+muArt (LAD-U는 implicit laplacian에만 들어감 —
        // fvm::laplacian(a,U)+fvm::laplacian(b,U) == fvm::laplacian(a+b,U))
        // CPU 규약: fvMatrix 생성자의 BC updateCoeffs() 상응 (phi
        // 의존 BC의 valueFraction 갱신 — 직접 조립 경로 필수)
        U.boundaryFieldRef().updateCoeffs();

        // ── div(phi,U) 가중치: CPU 스킴(limitedLinearV) 자체로 계산해
        //    업로드 — 디바이스 자체 계산은 컴파일러 FMA 축약(gcc
        //    -ffp-contract=fast vs nvcc -fmad=false)의 ULP 차이로 평탄
        //    필드에서 리미터의 이산 결정이 뒤집힐 수 있다(h-수송
        //    ULP-플립과 동일 클래스) — 호스트 계산이 유일한 비트-일치
        //    경로. coupled proc-면 가중치도 스킴이 함께 산출 ──
        tmp<surfaceScalarField> twU;
        {
            ITstream& is = mesh.schemes().div("div(phi,U)");
            is.rewind();
            const word gaussName(is);   // "Gauss" 소비
            twU = surfaceInterpolationScheme<vector>::New
            (
                mesh, phi, is
            )().weights(U);
        }
        const surfaceScalarField& wU = twU();

        const volScalarField muEff(rho*momentumTransport->nuEff());
        tmp<volScalarField> tmuLap;
        if (LADUCoeff > 0)
        {
            tmuLap = muEff + muArt;
        }
        const volScalarField& muLap =
            tmuLap.valid() ? tmuLap() : muEff;

        // LAD-bulk: -fvc::grad(betaArt*divU)는 저장 소스(H() 포함)
        tmp<volVectorField> tsrcBulk;
        if (LADbulkCoeff > 0)
        {
            const volScalarField divU(fvc::div(U));
            volScalarField betaArt
            (
                IOobject
                (
                    "betaArt", mesh.time().name(), mesh,
                    IOobject::NO_READ, IOobject::NO_WRITE
                ),
                mesh,
                dimensionedScalar(dimensionSet(1, -1, -1, 0, 0, 0, 0), 0),
                zeroGradientFvPatchScalarField::typeName
            );
            const scalarField V23(pow(scalarField(mesh.V()), 2.0/3.0));
            betaArt.primitiveFieldRef() =
                LADbulkCoeff*rho.primitiveField()*V23
               *mag(divU.primitiveField());
            betaArt.correctBoundaryConditions();
            Info<< "LAD-bulk: betaArt max = "
                << gMax(betaArt.primitiveField()) << " kg/(m s)" << endl;
            // UEqn -= grad(...) → 저장 소스 += grad(...)
            tsrcBulk = fvc::grad(betaArt*divU);
        }

        // 비직교 명시 보정: −fvm::laplacian(muLap,U)의 corrected 소스
        // = +div(ΓmagSf·correction(U)) (gaussLaplacianScheme 1:1) —
        // 저장 소스(H() 포함)라 LAD-bulk와 같은 srcX3 슬롯에 합산
        tmp<volVectorField> tCorrU;
        if (gpuNonOrtho_)
        {
            tmp<fv::snGradScheme<vector>> tsnU
            (
                fv::snGradScheme<vector>::New
                (
                    mesh, mesh.schemes().snGrad(U.name())
                )
            );
            const surfaceScalarField muMagSf
            (
                fvc::interpolate(muLap)*mesh.magSf()
            );
            tCorrU = fvc::div(muMagSf*tsnU().correction(U));
        }

        // SoA gather (v2: 물리 필드는 GPU — 호스트는 U/경계값만)
        gpuUBuf_.setSize(15*nc + 22*nbf);
        double* U3 = gpuUBuf_.begin();          // [3*nc]
        double* U3old = U3 + 3*nc;
        double* U3out = U3old + 3*nc;
        double* H3 = U3out + 3*nc;              // [3*nc] (pCorr)
        double* srcX3 = H3 + 3*nc;              // [3*nc] LAD-bulk 소스
        double* UB3 = srcX3 + 3*nc;             // [3*nbf]
        double* pB = UB3 + 3*nbf;               // [nbf]
        double* bDiag3 = pB + nbf;              // [3*nbf]
        double* bSrc3 = bDiag3 + 3*nbf;
        double* bFlux3 = bSrc3 + 3*nbf;         // [3*nbf] 명시항 경계
        double* bGrad9 = bFlux3 + 3*nbf;        // [9*nbf]
        (void)H3;

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
            if (tsrcBulk.valid() || tCorrU.valid())
            {
                for (label i = 0; i < nc; i++)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        srcX3[k*nc + i] = 0.0;
                    }
                }
                if (tsrcBulk.valid())
                {
                    const vectorField& sb = tsrcBulk().primitiveField();
                    for (label i = 0; i < nc; i++)
                    {
                        for (label k = 0; k < 3; k++)
                        {
                            srcX3[k*nc + i] += sb[i][k];
                        }
                    }
                }
                if (tCorrU.valid())
                {
                    const vectorField& cu = tCorrU().primitiveField();
                    for (label i = 0; i < nc; i++)
                    {
                        for (label k = 0; k < 3; k++)
                        {
                            srcX3[k*nc + i] += cu[i][k];
                        }
                    }
                }
            }
        }
        {
            label off = 0;
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp = U.boundaryField()[patchi];
                const fvPatchScalarField& ppp = p.boundaryField()[patchi];
                if (pp.coupled())
                {
                    // gaussGrad coupled: 경계 기여 = 면 값
                    // w*own + (1-w)*nei (패치 필드가 이웃 셀 값 보유)
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
                        pB[off + f] =
                            w[f]*po[f] + (1.0 - w[f])*ppp[f];
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

        // 1) grad(U) 디바이스 + 경계셀 gradU 채집
        if (rgpUEqnGrad(U3, UB3, bGrad9))
        {
            FatalErrorInFunction
                << "rgpUEqnGrad: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        // 1b) 병렬: proc-면 이웃 gradU 교환 → 벡터 리미터(NVDVTVDV
        //     1:1, uLimVWeights의 호스트 포트) + dev2 명시 플럭스용
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
            // 자기 쪽 경계셀 gradU(bGrad9)의 proc 슬라이스를 텐서로
            // 패킹해 이웃 랭크와 교환
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

                    // proc-면 가중치: CPU 스킴이 산출한 coupled 면
                    // 가중치 그대로 (수제 리미터 재현 은퇴 — ULP 방어)
                    const scalarField& wUb =
                        wU.boundaryField()[patchi];
                    for (label f = 0; f < np; f++)
                    {
                        wPar[offp + f] = wUb[f];
                    }
                    offp += np;
                }
                off += np;
            }
        }

        // 2) 명시항 경계 기여 (호스트, 패치 크기): gaussGrad 경계 보정
        //    (gGrad_b = gGradI − n⊗(n·gGradI) + n⊗snGrad(U)) →
        //    X_b = mu_b*dev2(T(gGrad_b)) → Sf_b & X_b
        {
            label off = 0;
            label offp = 0;
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp = U.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const vectorField& Sfb =
                    mesh.Sf().boundaryField()[patchi];
                const scalarField& mub =
                    muEff.boundaryField()[patchi];

                if (pp.coupled())
                {
                    // fvc::div(mu*dev2(T(gradU))) coupled 면: 양측 X를
                    // 선형 보간 (uDevTauFlux 내부면 산술과 동일) —
                    // 이웃 gradU는 1b)에서 교환, 이웃 mu는 패치 값
                    const scalarField cdw
                    (
                        mesh.surfaceInterpolation::weights()
                            .boundaryField()[patchi]
                    );
                    const scalarField muo
                    (
                        muEff.boundaryField()[patchi]
                            .patchInternalField()
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
                        const tensor Xf
                        (
                            cdw[f]*Xo + (1.0 - cdw[f])*Xn
                        );
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
                        // gaussGrad::correctBoundaryConditions
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

        // 3) 가중치/명시항/grad p 디바이스 준비
        if
        (
            rgpUEqnPrep2
            (
                p.primitiveField().begin(), pB,
                muEff.primitiveField().begin(),
                phi.primitiveField().begin(), bFlux3
            )
        )
        {
            FatalErrorInFunction
                << "rgpUEqnPrep2: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        // 4) 경계 행렬 계수 (BC 가중치: 본 케이스 BC 타입들은 w 미사용
        //    — CD 가중치 전달)
        const surfaceScalarField dcsU
        (
            fv::snGradScheme<vector>::New
            (
                mesh, mesh.schemes().snGrad(U.name())
            )().deltaCoeffs(U)
        );
        List<double> bRelaxA;
        {
            label off = 0;
            label offp = 0;
            List<double> parB(max(nPar, label(1)), 0.0);
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp = U.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const scalarField& phb = phi.boundaryField()[patchi];

                if (pp.coupled())
                {
                    // processor: div 가중치 = 1b)의 limited 가중치,
                    // laplacian은 deltaCoeffs 오버로드; mu는 면 보간.
                    // iC → diag(bDiag3), bC → 인터페이스(parB, 성분 공유)
                    scalarField wV(np);
                    for (label f = 0; f < np; f++)
                    {
                        wV[f] = wPar[offp + f];
                    }
                    const vectorField vic(pp.valueInternalCoeffs(wV));
                    const vectorField vbc(pp.valueBoundaryCoeffs(wV));

                    const scalarField pdc
                    (
                        dcsU.boundaryField()[patchi]
                    );
                    const vectorField gic
                    (
                        pp.gradientInternalCoeffs(pdc)
                    );
                    const vectorField gbc
                    (
                        pp.gradientBoundaryCoeffs(pdc)
                    );

                    const scalarField cdw
                    (
                        mesh.surfaceInterpolation::weights()
                            .boundaryField()[patchi]
                    );
                    const scalarField muo
                    (
                        muLap.boundaryField()[patchi]
                            .patchInternalField()
                    );
                    const scalarField& mun =
                        muLap.boundaryField()[patchi];
                    const scalarField& msf =
                        mesh.magSf().boundaryField()[patchi];

                    for (label f = 0; f < np; f++)
                    {
                        const scalar gMsf =
                            (cdw[f]*muo[f] + (1.0 - cdw[f])*mun[f])
                           *msf[f];
                        for (label k = 0; k < 3; k++)
                        {
                            bDiag3[k*nbf + off + f] =
                                phb[f]*vic[f][k] - gMsf*gic[f][k];
                            bSrc3[k*nbf + off + f] = 0;
                        }
                        // coupled vic/vbc/gic/gbc는 등방(성분 동일)
                        parB[offp + f] =
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
                        muLap.boundaryField()[patchi]
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

            if (offp > 0 && rgpUEqnParCoeffs(parB.begin()))
            {
                FatalErrorInFunction
                    << "rgpUEqnParCoeffs: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            // relax 경계 배열 [add|rem|soff] — fvMatrix::relax의
            // 경계 취급 1:1: uncoupled add=cmptMax(cmptMag(iC)),
            // rem=cmptMin(iC); coupled add=rem=iC성분0, soff=|bC|
            if (relaxAlpha > 0)
            {
                bRelaxA.setSize(3*max(nbf, label(1)), 0.0);
                label off2 = 0;
                label offp2 = 0;
                forAll(U.boundaryField(), patchi)
                {
                    const fvPatchVectorField& pp =
                        U.boundaryField()[patchi];
                    const label np = pp.size();
                    if (np == 0) continue;

                    if (pp.coupled())
                    {
                        for (label f = 0; f < np; f++)
                        {
                            const scalar iC0 =
                                bDiag3[off2 + f];
                            bRelaxA[off2 + f] = iC0;
                            bRelaxA[nbf + off2 + f] = iC0;
                            bRelaxA[2*nbf + off2 + f] =
                                mag(parB[offp2 + f]);
                        }
                        offp2 += np;
                    }
                    else
                    {
                        for (label f = 0; f < np; f++)
                        {
                            const scalar c0 = bDiag3[off2 + f];
                            const scalar c1 =
                                bDiag3[nbf + off2 + f];
                            const scalar c2 =
                                bDiag3[2*nbf + off2 + f];
                            bRelaxA[off2 + f] =
                                max(mag(c0), max(mag(c1), mag(c2)));
                            bRelaxA[nbf + off2 + f] =
                                min(c0, min(c1, c2));
                            // soff = 0 (uncoupled)
                        }
                    }
                    off2 += np;
                }
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

        double res0[3], resF[3];
        int iters[3];
        const int rc = rgpUEqnSolve
        (
            LTS ? 1.0 : 1.0/runTime.deltaTValue(),
            LTS
          ? fv::localEulerDdt::localRDeltaT(mesh)
               .primitiveField().begin()
          : nullptr,
            rho.primitiveField().begin(),
            rho.oldTime().primitiveField().begin(),
            U3old, nullptr /*U3: 디바이스 상주*/,
            nullptr /*phi*/,
            wU.primitiveField().begin() /*w: 호스트 CPU-스킴 가중치*/,
            tmuLap.valid()
          ? muLap.primitiveField().begin() : nullptr /*mu: LAD-U 합산*/,
            nullptr /*srcExp3*/,
            (tsrcBulk.valid() || tCorrU.valid())
          ? srcX3 : nullptr /*LAD-bulk + 비직교 보정*/,
            nullptr /*gradP3*/,
            bDiag3, bSrc3, solveCmpt,
            relaxAlpha,
            relaxAlpha > 0 ? bRelaxA.begin() : nullptr,
            tol, rtol, maxIter,
            U3out, res0, resF, iters
        );
        if (rc)
        {
            FatalErrorInFunction
                << "rgpUEqnSolve: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        if (pimple.momentumPredictor())
        {
            vectorField& Uc = U.primitiveFieldRef();
            for (label k = 0; k < 3; k++)
            {
                if (!solveCmpt[k]) continue;
                Info<< "rgpBiCGStab: Solving for "
                    << word(U.name() + vector::componentNames[k])
                    << ", Initial residual = " << res0[k]
                    << ", Final residual = " << resF[k]
                    << ", No Iterations " << iters[k] << endl;
                for (label i = 0; i < nc; i++)
                {
                    Uc[i][k] = U3out[k*nc + i];
                }
            }

            U.correctBoundaryConditions();
            fvConstraints().constrain(U);
            K = 0.5*magSqr(U);
        }

        if (thermoTimings_)
        {
            Info<< "momentum predictor total = "
                << std::chrono::duration<double>
                   (std::chrono::steady_clock::now() - tTot).count()
                << " s" << endl;
        }
        return;
    }

    tUEqn =
    (
        fvm::ddt(rho, U) + fvm::div(phi, U)
      + MRF.DDt(rho, U)
      + momentumTransport->divDevTau(U)
     ==
        fvModels().source(rho, U)
    );
    fvVectorMatrix& UEqn = tUEqn.ref();

    // LAD 항은 계수가 켜졌을 때만 조립: 계수 0이어도 0-계수 laplacian/grad의
    // LDU 전체 조립이 스텝의 ~0.1-0.2s를 소모한다. 0 기여이므로 생략은
    // 비트-동일.
    if (LADUCoeff > 0)
    {
        UEqn -= fvm::laplacian(muArt, U);
    }
    if (LADbulkCoeff > 0)
    {
        const volScalarField divU(fvc::div(U));
        volScalarField betaArt
        (
            IOobject
            (
                "betaArt",
                mesh.time().name(),
                mesh,
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar(dimensionSet(1, -1, -1, 0, 0, 0, 0), 0),
            zeroGradientFvPatchScalarField::typeName
        );
        const scalarField V23(pow(scalarField(mesh.V()), 2.0/3.0));
        betaArt.primitiveFieldRef() =
            LADbulkCoeff*rho.primitiveField()*V23
           *mag(divU.primitiveField());
        betaArt.correctBoundaryConditions();
        Info<< "LAD-bulk: betaArt max = " << gMax(betaArt.primitiveField())
            << " kg/(m s)" << endl;

        UEqn -= fvc::grad(betaArt*divU);
    }

    UEqn.relax();

    fvConstraints().constrain(UEqn);

    if (thermoTimings_)
    {
        Info<< "UEqn assembly = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tAsm).count()
            << " s" << endl;
        tAsm = std::chrono::steady_clock::now();
    }

    if (pimple.momentumPredictor())
    {
        if (buoyancy.valid())
        {
            solve
            (
                UEqn
             ==
                netForce()
            );
        }
        else
        {
            solve(UEqn == -fvc::grad(p));
        }

        fvConstraints().constrain(U);
        K = 0.5*magSqr(U);

        if (thermoTimings_)
        {
            Info<< "UEqn solve = "
                << std::chrono::duration<double>
                   (std::chrono::steady_clock::now() - tAsm).count()
                << " s" << endl;
        }
    }

    if (thermoTimings_)
    {
        Info<< "momentum predictor total = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tTot).count()
            << " s" << endl;
    }
}


// ************************************************************************* //
