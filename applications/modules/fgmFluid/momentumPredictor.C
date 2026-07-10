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
#include "localEulerDdtScheme.H"
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
        if
        (
            mesh.solution().relaxEquation(U.name())
         && mesh.solution().equationRelaxationFactor(U.name()) != 1
        )
        {
            FatalErrorInFunction
                << "gpuUEqn (v1) does not support equation relaxation "
                << "on U" << exit(FatalError);
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
            if (tsrcBulk.valid())
            {
                const vectorField& sb = tsrcBulk().primitiveField();
                for (label i = 0; i < nc; i++)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        srcX3[k*nc + i] = sb[i][k];
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
                forAll(pp, f)
                {
                    for (label k = 0; k < 3; k++)
                    {
                        UB3[k*nbf + off + f] = pp[f][k];
                    }
                    pB[off + f] = ppp[f];
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

        // 2) 명시항 경계 기여 (호스트, 패치 크기): gaussGrad 경계 보정
        //    (gGrad_b = gGradI − n⊗(n·gGradI) + n⊗snGrad(U)) →
        //    X_b = mu_b*dev2(T(gGrad_b)) → Sf_b & X_b
        {
            label off = 0;
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp = U.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const vectorField nHat(mesh.boundary()[patchi].nf());
                const vectorField snU(pp.snGrad());
                const vectorField& Sfb =
                    mesh.Sf().boundaryField()[patchi];
                const scalarField& mub =
                    muEff.boundaryField()[patchi];

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
        {
            label off = 0;
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp = U.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const scalarField& phb = phi.boundaryField()[patchi];
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
                off += np;
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
            nullptr /*phi*/, nullptr /*w*/,
            tmuLap.valid()
          ? muLap.primitiveField().begin() : nullptr /*mu: LAD-U 합산*/,
            nullptr /*srcExp3*/,
            tsrcBulk.valid() ? srcX3 : nullptr /*LAD-bulk*/,
            nullptr /*gradP3*/,
            bDiag3, bSrc3, solveCmpt,
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
