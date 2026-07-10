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
        if (LTS || LADUCoeff > 0 || LADbulkCoeff > 0 || MRF.size() > 0)
        {
            FatalErrorInFunction
                << "gpuUEqn (v1) does not support LTS/LAD/MRF"
                << exit(FatalError);
        }

        armGpuSTMesh();

        const label nc = mesh.nCells();
        label nbf = 0;
        forAll(U.boundaryField(), patchi)
        {
            nbf += U.boundaryField()[patchi].size();
        }

        // div(phi,U) 스킴의 실제 가중치 (Gauss 토큰 소비 후 구성)
        ITstream divIs(mesh.schemes().div("div(phi,U)"));
        const word gaussWord(divIs);
        tmp<surfaceScalarField> twU
        (
            limitedSurfaceInterpolationScheme<vector>::New
            (
                mesh, phi, divIs
            )().weights(U)
        );
        const surfaceScalarField& wU = twU();

        // muEff = (rho*nuEff) — divDevTau/laplacian 계수 (이름 규약 유지:
        // fvc::div 스킴 조회가 "div(((rho*nuEff)*dev2(T(grad(U)))))"와
        // 일치해야 함)
        const volScalarField muEff(rho*momentumTransport->nuEff());

        // 저장 소스: divDevTau의 fvc 항만. −grad p는 solve() 전용 임시
        // 소스라 행렬 소스(H()에 반영되는)에 넣으면 안 된다 —
        // OF의 solve(UEqn == -fvc::grad(p)) 의미론 그대로.
        const volVectorField srcExp
        (
            fvc::div(muEff*dev2(T(fvc::grad(U))))
        );
        const volVectorField gradP(fvc::grad(p));

        // SoA gather + 경계 계수
        gpuUBuf_.setSize(15*nc + 6*nbf + 3*nc);
        double* U3 = gpuUBuf_.begin();          // [3*nc]
        double* U3old = U3 + 3*nc;
        double* src3 = U3old + 3*nc;
        double* U3out = src3 + 3*nc;
        double* gp3 = U3out + 3*nc;             // [3*nc] grad p
        double* bDiag3 = gp3 + 3*nc;            // [3*nbf]
        double* bSrc3 = bDiag3 + 3*nbf;
        double* H3 = bSrc3 + 3*nbf;             // [3*nc] (pCorr에서 사용)
        (void)H3;

        {
            const vectorField& Uc = U.primitiveField();
            const vectorField& Uo = U.oldTime().primitiveField();
            const vectorField& sc = srcExp.primitiveField();
            const vectorField& gp = gradP.primitiveField();
            for (label i = 0; i < nc; i++)
            {
                for (label k = 0; k < 3; k++)
                {
                    U3[k*nc + i] = Uc[i][k];
                    U3old[k*nc + i] = Uo[i][k];
                    src3[k*nc + i] = sc[i][k];
                    gp3[k*nc + i] = gp[i][k];
                }
            }
        }

        {
            label off = 0;
            forAll(U.boundaryField(), patchi)
            {
                const fvPatchVectorField& pp = U.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const scalarField& phb = phi.boundaryField()[patchi];
                const vectorField vic
                (
                    pp.valueInternalCoeffs
                    (
                        wU.boundaryField()[patchi]
                    )
                );
                const vectorField vbc
                (
                    pp.valueBoundaryCoeffs
                    (
                        wU.boundaryField()[patchi]
                    )
                );
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
            1.0/runTime.deltaTValue(),
            rho.primitiveField().begin(),
            rho.oldTime().primitiveField().begin(),
            U3old, U3,
            phi.primitiveField().begin(), wU.primitiveField().begin(),
            muEff.primitiveField().begin(),
            src3, gp3, bDiag3, bSrc3, solveCmpt,
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
