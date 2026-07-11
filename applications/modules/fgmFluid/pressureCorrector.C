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

Description
    fgmFluid pressure corrector -- overrides isothermalFluid::pressureCorrector
    so the pressure equation can be replaced by a pressure-equilibrium-
    preserving (pressure-evolution) formulation. STEP 1 (this file, first
    build): correctPressurePEP() is a FAITHFUL COPY of the base
    isothermalFluid::correctPressure (non-buoyant, SIMPLErho path), so the 1-D
    advecting-interface benchmark must reproduce the baseline spurious pressure
    spike unchanged -- confirming the override mechanism is isolated before the
    PEP modification is introduced. Reference for the PEP target:
    Terashima & Koshi, J. Comput. Phys. 231 (2012) 6907; Kai/Kurose PEQSI,
    Phys. Fluids 36 (2024) 116104.

\*---------------------------------------------------------------------------*/

#include "fgmFluid.H"
#include "constrainHbyA.H"
#include "constrainPressure.H"
#include "adjustPhi.H"
#include "fvcMeshPhi.H"
#include "fvcFlux.H"
#include "fvcDdt.H"
#include "fvcGrad.H"
#include "fvcSnGrad.H"
#include "fvcReconstruct.H"
#include "fvcVolumeIntegrate.H"
#include "fvmDiv.H"
#include "fvmLaplacian.H"
#include "fvcLaplacian.H"
#include "fvcAverage.H"
#include "zeroGradientFvPatchFields.H"
#include "extrapolatedCalculatedFvPatchFields.H"
#include "solutionControl.H"
#include "localEulerDdtScheme.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "processorFvPatchFields.H"
#include "processorFvPatch.H"
#include "PstreamReduceOps.H"
#include "gpu/rgpPEqnTypes.H"

#include <chrono>
#include <dlfcn.h>

// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::solvers::fgmFluid::armGpuPEqnMesh()
{
    const label nc = mesh.nCells();
    const label nif = mesh.owner().size();

    // AMR/재분배로 토폴로지가 바뀌면 stale owner/neigh/V/가중치로
    // 조용히 틀린 행렬을 조립하게 된다 — 상태 불일치 시 전체 재아밍
    // (rgpPEqnMeshUpload가 디바이스 버퍼 전체 해제 후 재할당).
    // 셀/면 수 비교만으로는 (a) 수 보존 topo 변화(refine+unrefine 상쇄)
    // (b) 메시 이동(연결성 불변, V/Sf/deltaCoeffs 변화)을 놓치므로
    // topoChanged()/moving() 플래그도 본다. 플래그는 스텝 내내 유지되고
    // 이 함수는 스텝 중간에도 불리므로(재아밍이 devChain의 조립된 UEqn
    // 행렬까지 해제) timeIndex 래치로 스텝당 1회만 재아밍한다 — 메시
    // 변화(preSolve의 topo, firstIter의 move)는 predictor들보다 먼저
    // 일어나므로 스텝 첫 아밍 호출에서의 재아밍이 정확하다.
    if (gpuPEqnArmed_)
    {
        const bool flagged =
            (mesh.topoChanged() || mesh.moving())
         && mesh.time().timeIndex() != gpuMeshStampTime_;

        if (nc == gpuMeshCells_ && nif == gpuMeshFaces_ && !flagged)
        {
            return;
        }
        Info<< "fgmFluid: mesh topology/geometry changed ("
            << gpuMeshCells_ << " -> " << nc
            << " cells) -- re-arming GPU mesh structures" << nl << endl;
        gpuZCArmed_ = false;   // ST 메시(가중치/Sf/d)도 재업로드 필요
    }

    // 외부 반복마다 이동하는 mover는 스텝-단위 래치 가정을 깬다 (v1)
    if
    (
        mesh.dynamic()
     && pimple.dict().lookupOrDefault<Switch>
        (
            "moveMeshOuterCorrectors", false
        )
    )
    {
        FatalErrorInFunction
            << "gpuPEqn/gpuZC/gpuUEqn (v1) do not support "
            << "moveMeshOuterCorrectors (mesh geometry would change "
            << "between outer iterations while the GPU copy is armed "
            << "once per time step)"
            << exit(FatalError);
    }

    label nbf = 0;
    forAll(p_.boundaryField(), patchi)
    {
        if (p_.boundaryField()[patchi].coupled())
        {
            // processor 패치는 병렬 pEqn이 halo 교환으로 지원 —
            // 그 외 커플드(cyclic/AMI)는 v1 미지원
            if
            (
                !isA<processorFvPatchScalarField>
                (
                    p_.boundaryField()[patchi]
                )
            )
            {
                FatalErrorInFunction
                    << "gpuPEqn/gpuZC (v1) support processor coupling "
                    << "only (no cyclic/AMI)"
                    << exit(FatalError);
            }
        }
        nbf += p_.boundaryField()[patchi].size();
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
    forAll(p_.boundaryField(), patchi)
    {
        const labelUList& fc = mesh.boundary()[patchi].faceCells();
        forAll(fc, k) { bfc[off + k] = fc[k]; }
        off += fc.size();
    }

    const scalarField Vc(mesh.V());
    const int rc = rgpPEqnMeshUpload
    (
        nc, nif, own.begin(), nei.begin(), gg.begin(),
        Vc.begin(), nbf, bfc.begin()
    );
    if (rc)
    {
        FatalErrorInFunction
            << "rgpPEqnMeshUpload: " << rgpPEqnLastError()
            << exit(FatalError);
    }
    // CSR 구조는 솔버 무관 준비: amgx 직결 + PCG의 무-atomics SpMV 겸용
    {
        int nnz = 0;
        if (rgpPEqnCsrPrepare(&nnz))
        {
            FatalErrorInFunction
                << "rgpPEqnCsrPrepare: " << rgpPEqnLastError()
                << exit(FatalError);
        }
        gpuPEqnNnz_ = nnz;
    }

    // ── 병렬: processor 패치 구조 아밍 (halo 교환 + 전역 리덕션) ──
    if (Pstream::parRun())
    {
        DynamicList<int> pNbr, pNF;
        DynamicList<int> pFc;
        forAll(p_.boundaryField(), patchi)
        {
            if (!p_.boundaryField()[patchi].coupled()) continue;
            const processorFvPatch& pp =
                refCast<const processorFvPatch>
                (
                    mesh.boundary()[patchi]
                );
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
        gpuPEqnParB_.setSize(max(pFc.size(), label(1)));
        gpuPEqnParPhb_.setSize(max(pFc.size(), label(1)));
        gpuPEqnParRhof_.setSize(max(pFc.size(), label(1)));
    }

    gpuPEqnArmed_ = true;
    gpuMeshCells_ = nc;
    gpuMeshFaces_ = nif;
    gpuMeshStampTime_ = mesh.time().timeIndex();
    Info<< "fgmFluid: GPU pEqn mesh armed -- " << nc << " cells, "
        << nif << " internal faces, " << nbf
        << " boundary faces (pEqn solver: " << gpuPEqnSolver_ << ")"
        << nl << endl;
}


void Foam::solvers::fgmFluid::correctPressurePEP()
{
    volScalarField& rho(rho_);
    volScalarField& p(p_);
    volVectorField& U(U_);
    surfaceScalarField& phi(phi_);

    // --- RANK 1: refresh the real-fluid rho, psi at the CURRENT pressure each
    // pressure corrector (not once per outer). A stale SRK compressibility
    // psi = (drho/dp)_T at the stiff cold-LOX injector cells lets the pressure
    // corrector run against a near-singular diagonal and generate the injector
    // pressure spike -- the pressure-velocity / EOS-stiffness ill-conditioning
    // that neither PEP nor LAD addresses (Ma, Lv & Ihme, J. Comput. Phys. 340
    // (2017) 330). Property-update-per-corrector is the realFluidFoam recipe
    // (Nguyen & Yoo, Comput. Phys. Commun. 312 (2025) 109600). thermo_.correct()
    // inverts the manifold-seeded he to T and refreshes rho/psi/mu at (p,T,Y).
    // (b) he->T-drift-free variant: re-seed he from the manifold at the CURRENT
    // pressure (updateManifold) BEFORE correct(), so the he->T inversion returns
    // T_table exactly (no drift) while rho/psi still refresh at the new p. This
    // fixes the RANK-1 side effect where bare thermo_.correct() inverted a stale
    // he against the new p and drifted T, moving the spike into the chamber.
    std::chrono::steady_clock::time_point tPh, tTot;
    if (thermoTimings_) { tPh = tTot = std::chrono::steady_clock::now(); }

    updateManifold();

    if (thermoTimings_)
    {
        Info<< "pCorr manifold = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tPh).count()
            << " s" << endl;
        tPh = std::chrono::steady_clock::now();
    }

    // 예측자(thermophysicalPredictor)와 동일한 GPU 물성 refresh 대체.
    // updateManifold가 직전에 he를 T_table로 재시드하므로 CPU 경로의
    // he->T 역산은 T_table을 되돌려줄 뿐 — GPU 경로(역산 생략, (p,T,Y)
    // 일괄 물성)와 의미가 같다. 보정자당 1회 x nCorr회 호출되는 CPU
    // calculate() 루프가 제거된다.
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
        Info<< "pCorr thermo correct = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tPh).count()
            << " s" << endl;
        tPh = std::chrono::steady_clock::now();
    }

    // Per-corrector transported-density re-sync (modified-PIMPLE, cf.
    // realFluidFoam/Jarczyk-Pfitzner). The transported rho_ is otherwise
    // advanced only by the correctRho(psi*dp) increments; at a flame-zone
    // pressure spike those increments are huge and (with the pMinPa clamp)
    // inconsistent, so rho_ drifts from the EOS state and accumulates error
    // -- observed: rho_ down to -1175 kg/m3 at the ox tangential holes while
    // the EOS density stayed positive. Snapping rho_ to the just-corrected
    // thermo state each corrector removes the drift (rho.oldTime() is
    // untouched, so the ddt history stays consistent).
    rho_ = thermo.rho();
    rho_.correctBoundaryConditions();

    const volScalarField& psi = thermo.psi();
    rho = thermo.rho();
    rho.relax();

    // pCorrGPU 디바이스 체인: gpuUEqn(rAU/HbyA 디바이스) + gpuPEqn일 때
    // fvc 준비체인(rhof/rhorAUf/rAUf/psis/phiHbyAv)을 GPU 상주로 대체
    const bool devChain = gpuUEqn_ && gpuPEqn_;

    tmp<surfaceScalarField> trhof;
    if (!devChain)
    {
        trhof = surfaceScalarField::New("rhof", fvc::interpolate(rho));
    }

    // UEqnGPU: 행렬이 디바이스 상주이므로 rAU=1/A(), H(U)를 GPU에서 산출
    // (fvMatrix::D/H 1:1). 그 외에는 기존 CPU 경로.
    tmp<volScalarField> trAU;
    tmp<volVectorField> tHbyA;
    if (gpuUEqn_)
    {
        if (pimple.consistent())
        {
            FatalErrorInFunction
                << "gpuUEqn (v1) does not support 'consistent'"
                << exit(FatalError);
        }

        const label nc = mesh.nCells();
        label nbf = 0;
        forAll(U.boundaryField(), patchi)
        {
            nbf += U.boundaryField()[patchi].size();
        }

        double* U3 = gpuUBuf_.begin();
        double* rAUh = U3 + 6*nc;               // U3out 슬롯 재사용
        double* H3 = U3 + 9*nc;

        {
            const vectorField& Uc = U.primitiveField();
            for (label i = 0; i < nc; i++)
            {
                for (label k = 0; k < 3; k++)
                {
                    U3[k*nc + i] = Uc[i][k];
                }
            }
        }

        // 병렬: fvMatrix::H()의 coupled addBoundarySource(H += bC*U_nbr)
        // 용 이웃 U — momentum cBC 직후라 proc 패치가 이웃 값 보유
        List<double> UNbr3;
        if (Pstream::parRun())
        {
            label nPar = 0;
            forAll(U.boundaryField(), patchi)
            {
                if (U.boundaryField()[patchi].coupled())
                {
                    nPar += U.boundaryField()[patchi].size();
                }
            }
            if (nPar > 0)
            {
                UNbr3.setSize(3*nPar);
                label offp = 0;
                forAll(U.boundaryField(), patchi)
                {
                    const fvPatchVectorField& pp =
                        U.boundaryField()[patchi];
                    if (!pp.coupled()) continue;
                    forAll(pp, f)
                    {
                        for (label k = 0; k < 3; k++)
                        {
                            UNbr3[k*nPar + offp + f] = pp[f][k];
                        }
                    }
                    offp += pp.size();
                }
            }
        }

        const int rc = rgpUEqnAH
        (
            U3, UNbr3.size() ? UNbr3.begin() : nullptr, rAUh, H3
        );
        if (rc)
        {
            FatalErrorInFunction
                << "rgpUEqnAH: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        trAU = volScalarField::New
        (
            "rAU", mesh,
            dimensionedScalar(dimVolume*dimTime/dimMass, 0),
            extrapolatedCalculatedFvPatchScalarField::typeName
        );
        {
            scalarField& r = trAU.ref().primitiveFieldRef();
            for (label i = 0; i < nc; i++) { r[i] = rAUh[i]; }
        }
        trAU.ref().correctBoundaryConditions();

        tmp<volVectorField> tH = volVectorField::New
        (
            "H(U)", mesh,
            dimensionedVector(dimDensity*dimVelocity/dimTime, Zero),
            extrapolatedCalculatedFvPatchScalarField::typeName
        );
        {
            vectorField& h = tH.ref().primitiveFieldRef();
            for (label i = 0; i < nc; i++)
            {
                for (label k = 0; k < 3; k++)
                {
                    h[i][k] =
                        (mesh.solutionD()[k] == 1) ? H3[k*nc + i] : 0.0;
                }
            }
        }
        tH.ref().correctBoundaryConditions();

        tHbyA = constrainHbyA(trAU()*tH, U, p);
    }
    else
    {
        fvVectorMatrix& UEqn = tUEqn.ref();
        trAU = volScalarField::New("rAU", 1.0/UEqn.A());
        tHbyA = constrainHbyA(trAU()*UEqn.H(), U, p);
    }

    const volScalarField& rAU = trAU();
    tmp<surfaceScalarField> trhorAUf;
    if (!devChain)
    {
        trhorAUf = surfaceScalarField::New
        (
            "rhorAUf", fvc::interpolate(rho*rAU)
        );
    }

    tmp<volScalarField> rAtU
    (
        (!gpuUEqn_ && pimple.consistent())
      ? volScalarField::New("rAtU", 1.0/(1.0/rAU - tUEqn.ref().H1()))
      : tmp<volScalarField>(nullptr)
    );

    tmp<surfaceScalarField> rhorAtUf
    (
        (!gpuUEqn_ && pimple.consistent())
      ? surfaceScalarField::New("rhoRAtUf", fvc::interpolate(rho*rAtU()))
      : tmp<surfaceScalarField>(nullptr)
    );

    const volScalarField& rAAtU = pimple.consistent() ? rAtU() : rAU;

    volVectorField HbyA(tHbyA);

    if (!gpuUEqn_ && pimple.nCorrPiso() <= 1)
    {
        tUEqn.clear();
    }

    // --- PEP (pressure-evolution) pressure equation -------------------------
    // Replace the base continuity pEqn's fvc::ddt(rho) -- whose Eulerian
    // thermal density change radiates the spurious contact pressure spike --
    // with the pressure-evolution form  psis*dp/dt + div(u) = 0, where psis =
    // 1/(rho c^2) is the isentropic compressibility and div(u) is the
    // VOLUMETRIC velocity divergence (NOT the mass-flux divergence). For a
    // uniform-p, uniform-u contact div(u) = 0 and dp/dt = 0 by construction, so
    // the density jump advects without radiating pressure (Terashima & Koshi,
    // J. Comput. Phys. 231 (2012) 6907; Kai/Kurose PEQSI, Phys. Fluids 36
    // (2024) 116104). ATTEMPT 1: ideal-gas isentropic compressibility
    // 1/(gamma p) -- validates the STRUCTURE (spike removal); refined to the
    // real SRK sound speed next. constrainPressure / MRF-mass / consistent
    // branches dropped for this first cyclic-benchmark attempt.
    // C1a: HARMONIC-mean face mobility (Rhie-Chow across the stiff density
    // jump). rAU = 1/A() ~ dt/rho jumps ~60x cell-to-cell at the recess-tip
    // LOx/gas interface; the arithmetic fvc::interpolate(rAU) over-weights the
    // low-density (large rAU) side and over-stiffens the pEqn face coefficient
    // by ~O(30), so a small dp drives a huge face flux -> the spurious pressure
    // spike with |U| pinned at the limiter (co-located max(p)/|U|-cap). The
    // harmonic (series-resistance-correct) mean is the density-jump-robust
    // choice for a face mobility (Ferziger & Peric; Rhie & Chow, AIAA J. 21
    // (1983) 1525). 1/rAU = A() > 0 (momentum diagonal) so no floor needed.
    tmp<surfaceScalarField> trAUf;
    if (!devChain)
    {
        trAUf = surfaceScalarField::New
        (
            "rAUf", 1.0/fvc::interpolate(1.0/rAU)
        );
    }
    // ATTEMPT 2: real SRK isothermal compressibility psis = (drho/dp)_T/rho =
    // kappa_T (thermo.psi() now returns the real (drho/dp)_T -- see SRKGasI.H).
    // This supplies the true dense-fluid stiffness that the ideal-gas 1/(gamma
    // p) lacked (~100x too soft for liquid LOX).
    // C3: pressure-based double-flux analogue -- bound the cell-to-cell
    // CONTRAST of the pEqn diagonal compressibility psis = kappa_T across the
    // stiff-LOX / soft-gas interface. psis jumps enormously there (dense LOX
    // stiff, warm gas soft); a large diagonal-ratio next to the (now harmonic)
    // off-diagonal still lets a residual dp blow up. Floor each cell's psis at
    // (face-averaged psis)/psisCapRatio so the interface diagonal ratio is
    // capped -- the honest pressure-based cousin of freezing a local effective
    // gamma* (Ma, Lv & Ihme, JCP 340 (2017) 330). psisCapRatio default GREAT
    // (off); read each step. Conservative at convergence (psis*ddt(p) -> 0).
    const scalar psisCapRatio
    (
        pimple.dict().lookupOrDefault<scalar>("psisCapRatio", GREAT)
    );
    // psisIsentropic: use the ISENTROPIC compressibility kappa_s = kappa_T/gamma
    // = 1/(rho c^2) instead of the isothermal kappa_T = thermo.psi()/rho. For a
    // pressure-EVOLUTION (PEP) equation the acoustically-correct diagonal is the
    // isentropic one; the isothermal psi() over-stiffens the dense cold LOX
    // (Terashima-Koshi; design-agent refinement). Default off.
    const Switch psisIsentropic
    (
        pimple.dict().lookupOrDefault<Switch>("psisIsentropic", false)
    );
    tmp<volScalarField> tpsis;
    if (!devChain)
    {
        tpsis = volScalarField::New
        (
            "psis",
            psisIsentropic
          ? volScalarField(thermo.psi()/(rho*thermo.gamma()))
          : volScalarField(thermo.psi()/rho)
        );
        if (psisCapRatio < GREAT)
        {
            volScalarField& psis = tpsis.ref();
            const volScalarField psisSm
            (
                fvc::average(fvc::interpolate(psis))
            );
            psis = max(psis, psisSm/psisCapRatio);
            psis.correctBoundaryConditions();
        }
    }

    // Volumetric predicted face flux with the RHO-CONSISTENT transient
    // Rhie-Chow correction (C2 proper). The base compressible form is the MASS
    // flux  rhof*fvc::flux(HbyA) + rhorAUf*fvc::ddtCorr(rho,U,phi,rhoUf)
    // (isothermalFluid::correctPressure); the volumetric PEP is its /rhof. The
    // rho-aware ddtCorr(rho,U,phi,rhoUf) (rhoUf is null on a static mesh, where
    // the overload falls back to the rho-weighted form) keeps the transient RC
    // CONSISTENT with the momentum flux fvm::div(phi,U) across the ~60x rho jump
    // -- unlike the earlier ddtCorr(U, phi/rhof), whose arithmetic rhof in the
    // denominator injected a spurious face velocity (the recess-tip/injector
    // spike). This replaces the rcDdtScale 0 workaround, which suppressed the
    // spurious RC flux but left the injector fine cells to CHECKERBOARD (a new
    // spike re-formed there). rcDdtScale (default 1) still gates the term for
    // A/B testing. Read each step (runTimeModifiable).
    const scalar rcDdtScale
    (
        pimple.dict().lookupOrDefault<scalar>("rcDdtScale", scalar(1))
    );
    tmp<surfaceScalarField> tphiHbyAv;
    if (!devChain)
    {
        tphiHbyAv = surfaceScalarField::New
        (
            "phiHbyAv",
            fvc::flux(HbyA)
          + rcDdtScale*trhorAUf()*fvc::ddtCorr(rho, U, phi, rhoUf)
           /trhof()  // rho-consistent RC
        );
        MRF.makeRelative(tphiHbyAv.ref());

        // Update the pressure BCs for flux consistency (3D:
        // waveTransmissive outlet, fixedFluxPressure walls).
        // 체적 플럭스(phiHbyAv[m3/s])에는 rho 인자 없는 오버로드가 정합
        // (rho 버전은 질량 플럭스용 — fixedFluxPressure에서 rho_b배 오차)
        constrainPressure(p, U, tphiHbyAv(), trAUf(), MRF);
    }
    else
    {
        // pCorrGPU: rhof/rAUf(조화)/phiHbyAv(flux(HbyA)+Euler ddtCorr)/
        // psis를 디바이스 상주 생성 (경계 ddtCorr=0은 OF 규약 그대로 —
        // phiHbyAv 경계는 HbyA_b&Sf_b, 아래 pEqn 경계 배열에서 처리).
        // psisCapRatio/psisIsentropic은 호스트 계산본 override 업로드.
        tmp<volScalarField> tpsisOv;
        if (psisCapRatio < GREAT || psisIsentropic)
        {
            tpsisOv = volScalarField::New
            (
                "psis",
                psisIsentropic
              ? volScalarField(thermo.psi()/(rho*thermo.gamma()))
              : volScalarField(thermo.psi()/rho)
            );
            if (psisCapRatio < GREAT)
            {
                volScalarField& psisOv = tpsisOv.ref();
                const volScalarField psisSm
                (
                    fvc::average(fvc::interpolate(psisOv))
                );
                psisOv = max(psisOv, psisSm/psisCapRatio);
                psisOv.correctBoundaryConditions();
            }
        }

        const label nc = mesh.nCells();
        // U.oldTime SoA는 momentumPredictor의 gather 재사용
        const double* U3old = gpuUBuf_.begin() + 3*nc;
        if (rgpPCorrPrep
            (
                rho.primitiveField().begin(),
                rho.oldTime().primitiveField().begin(),
                U3old,
                phi.oldTime().primitiveField().begin(),
                psi.primitiveField().begin(),
                tpsisOv.valid()
              ? tpsisOv().primitiveField().begin() : nullptr,
                LTS ? 1.0 : 1.0/runTime.deltaTValue(),
                LTS
              ? fv::localEulerDdt::localRDeltaT(mesh)
                   .primitiveField().begin()
              : nullptr,
                rcDdtScale
            ))
        {
            FatalErrorInFunction
                << "rgpPCorrPrep: " << rgpPEqnLastError()
                << exit(FatalError);
        }

        // constrainPressure형 p BC(fixedFluxPressure류) 지원: 경계값만
        // 유효한 surface 필드를 구성해 표준 호출 (내부값은 미사용)
        {
            bool needConstrain = false;
            forAll(p.boundaryField(), patchi)
            {
                if
                (
                    isA<fixedFluxPressureFvPatchScalarField>
                    (
                        p.boundaryField()[patchi]
                    )
                )
                {
                    needConstrain = true;
                }
            }
            if (needConstrain)
            {
                tmp<surfaceScalarField> tphiB = surfaceScalarField::New
                (
                    "phiHbyAvB", mesh,
                    dimensionedScalar(dimVelocity*dimArea, 0)
                );
                tmp<surfaceScalarField> trAUfB = surfaceScalarField::New
                (
                    "rAUfB", mesh,
                    dimensionedScalar(rAU.dimensions(), 1)
                );
                forAll(p.boundaryField(), patchi)
                {
                    const label np = p.boundaryField()[patchi].size();
                    if (np == 0) continue;
                    tphiB.ref().boundaryFieldRef()[patchi] ==
                        (
                            HbyA.boundaryField()[patchi]
                          & mesh.Sf().boundaryField()[patchi]
                        );
                    trAUfB.ref().boundaryFieldRef()[patchi] ==
                        rAU.boundaryField()[patchi];
                }
                constrainPressure(p, U, tphiB(), trAUfB(), MRF);
            }
        }
    }

    // --- RANK 4: Artificial Mass Diffusivity (AMD) on DENSITY -----------------
    // Kawai, Terashima & Negishi, J. Comput. Phys. 300 (2015) 116: at a large
    // density ratio, diffusing the SCALARS/temperature (as the manifold LAD does
    // on Z,C,h) drives spurious p/u oscillations THROUGH the nonlinear EOS,
    // whereas diffusing MASS/DENSITY is the consistent choice. Smooth the steep
    // transcritical density interface at source by adding a density-gradient-
    // sensed diffusion of rho to the (volumetric) continuity, WITHOUT touching
    // the manifold scalars. Kinematic AMD coefficient [m^2/s], sized to the
    // local cell and active only where rho varies:
    //   Dr = LADrhoCoeff * V^(1/3) * |U| * s,  s = min(|grad rho| V^(1/3)/rho, 1)
    // where s is a DIMENSIONLESS density-gradient sensor CAPPED at 1: the raw
    // |grad rho| blows up at the spurious spike itself, so an uncapped Dr
    // violates the diffusive CFL (Dr ~ 164 m^2/s -> dt collapse). The cap ties
    // Dr_max = LADrhoCoeff*V^(1/3)*|U| to the convective scale, keeping the
    // diffusive step >= the convective one (Olson & Lele, JCP 246 (2013) 207).
    // The continuity then gains -(1/rho) div(Dr grad rho) [1/s], matching the
    // volumetric pressure-evolution form. LADrhoCoeff read each step (default 0).
    const scalar LADrhoCoeff
    (
        pimple.dict().lookupOrDefault<scalar>("LADrhoCoeff", scalar(0))
    );
    volScalarField Dr
    (
        IOobject
        (
            "Dr_amd",
            mesh.time().name(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar(dimArea/dimTime, 0),
        zeroGradientFvPatchScalarField::typeName
    );
    if (LADrhoCoeff > 0)
    {
        const scalarField V13(pow(scalarField(mesh.V()), 1.0/3.0));
        // Dimensionless density-gradient sensor in [0,1]: the relative rho
        // change across a cell, CAPPED at 1 so the spurious spike itself (huge
        // |grad rho|) cannot drive the coefficient past the diffusive-CFL limit.
        const scalarField sensor
        (
            min
            (
                mag(fvc::grad(rho))().primitiveField()*V13/rho.primitiveField(),
                scalar(1)
            )
        );
        Dr.primitiveFieldRef() =
            LADrhoCoeff*V13*mag(U)().primitiveField()*sensor;
        Dr.correctBoundaryConditions();
        Info<< "LAD-rho: Dr_amd max = " << gMax(Dr.primitiveField())
            << " m^2/s" << endl;
    }

    if (gpuPEqn_)
    {
        // --- pEqnGPU: fvMatrix 우회 — 디바이스 조립 + Jacobi-PCG + 플럭스.
        // 'Gauss linear orthogonal' laplacian + Euler ddt + 비커플드 패치
        // 전용(v1). 경계 기여만 호스트가 fvPatchField API로 정확 계산.

        // 행렬-레벨 fvConstraint 차단 (CPU 경로의 constrain(pEqn)은 GPU
        // 조립에 미반영; limitPressure류 필드-레벨은 솔브 후 적용됨)
        checkGpuConstraints(p.name(), "gpuPEqn");

        const label nc = mesh.nCells();
        const label nif = mesh.owner().size();

        label nbf = 0;
        forAll(p.boundaryField(), patchi)
        {
            if
            (
                p.boundaryField()[patchi].coupled()
             && !isA<processorFvPatchScalarField>
                (
                    p.boundaryField()[patchi]
                )
            )
            {
                FatalErrorInFunction
                    << "gpuPEqn (v1) supports processor coupling only"
                    << exit(FatalError);
            }
            nbf += p.boundaryField()[patchi].size();
        }

        if (Pstream::parRun())
        {
            if (gpuPEqnSolver_ != "pcg")
            {
                FatalErrorInFunction
                    << "parallel gpuPEqn supports the pcg solver only"
                    << exit(FatalError);
            }
            if (gpuPEqnCheck_)
            {
                FatalErrorInFunction
                    << "gpuPEqnCheck is serial-only"
                    << exit(FatalError);
            }
        }

        // ── 1회 아밍: 정적 메시 (owner/neigh, magSf*deltaCoeffs, V) ──
        armGpuPEqnMesh();

        // ── 경계 기여: pEqn = -laplacian(rAUf, p)의 경계 계수 ─────────
        //    diag += -pGamma*gic, source += +pGamma*gbc (fvm 부호 규약)
        gpuPEqnBuf_.setSize(3*nbf);
        double* bDiagA = gpuPEqnBuf_.begin();
        double* bSrcA  = bDiagA + nbf;
        double* phiBA  = bSrcA + nbf;
        {
            label off = 0;
            label offPar = 0;
            forAll(p.boundaryField(), patchi)
            {
                const fvPatchScalarField& pp = p.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                // devChain: rAUf_b = rAU_b (조화평균 경계 = 패치값),
                // phiHbyAv_b = HbyA_b & Sf_b (경계 ddtCorr = 0)
                const scalarField pGamma
                (
                    (
                        devChain
                      ? scalarField(rAU.boundaryField()[patchi])
                      : scalarField(trAUf().boundaryField()[patchi])
                    )
                   *mesh.magSf().boundaryField()[patchi]
                );
                // coupled 패치는 무인자 gradient*Coeffs()가 미구현 —
                // 패치 deltaCoeffs 오버로드 사용 (직교 스킴 1:1)
                const scalarField pdc
                (
                    mesh.deltaCoeffs().boundaryField()[patchi]
                );
                const scalarField gic
                (
                    pp.coupled()
                  ? pp.gradientInternalCoeffs(pdc)
                  : pp.gradientInternalCoeffs()
                );
                const scalarField gbc
                (
                    pp.coupled()
                  ? pp.gradientBoundaryCoeffs(pdc)
                  : pp.gradientBoundaryCoeffs()
                );
                const scalarField phb
                (
                    devChain
                  ? scalarField
                    (
                        HbyA.boundaryField()[patchi]
                      & mesh.Sf().boundaryField()[patchi]
                    )
                  : scalarField(tphiHbyAv().boundaryField()[patchi])
                );

                if (pp.coupled())
                {
                    // processor 패치: iC는 diag로(동일 공식), 커플링
                    // 계수 bC(=+pGamma*gbc)는 인터페이스 SpMV로 —
                    // 소스에 접지 않는다 (OF boundaryCoeffs 규약).
                    // devChain: 비-devChain의 fvc 체인이 coupled 보간을
                    // 이미 담던 것을 호스트가 필드(cBC 후)로 재현 —
                    // rAUf=조화, phiHbyAv=flux(HbyA)+rho-일관 ddtCorr
                    // (fvcDdtPhiCoeff는 coupled 경계를 0으로 만들지
                    // 않으므로 내부면 공식 그대로), rhof=선형.
                    scalarField pGammaP(pGamma);
                    scalarField phbP(phb);
                    scalarField rhofP(np, 0.0);
                    if (devChain)
                    {
                        const scalarField cdw
                        (
                            mesh.surfaceInterpolation::weights()
                                .boundaryField()[patchi]
                        );
                        const scalarField rAUo
                        (
                            rAU.boundaryField()[patchi]
                                .patchInternalField()
                        );
                        const scalarField& rAUn =
                            rAU.boundaryField()[patchi];
                        const vectorField Ho
                        (
                            HbyA.boundaryField()[patchi]
                                .patchInternalField()
                        );
                        const vectorField& Hn =
                            HbyA.boundaryField()[patchi];
                        const scalarField rhoo
                        (
                            rho.boundaryField()[patchi]
                                .patchInternalField()
                        );
                        const scalarField& rhon =
                            rho.boundaryField()[patchi];
                        const scalarField rhoOldo
                        (
                            rho.oldTime().boundaryField()[patchi]
                                .patchInternalField()
                        );
                        const scalarField& rhoOldn =
                            rho.oldTime().boundaryField()[patchi];
                        const vectorField UOldo
                        (
                            U.oldTime().boundaryField()[patchi]
                                .patchInternalField()
                        );
                        const vectorField& UOldn =
                            U.oldTime().boundaryField()[patchi];
                        const scalarField& phiOldb =
                            phi.oldTime().boundaryField()[patchi];
                        const vectorField& Sfb =
                            mesh.Sf().boundaryField()[patchi];
                        const scalarField& msf =
                            mesh.magSf().boundaryField()[patchi];

                        tmp<scalarField> trdto, trdtn;
                        if (LTS)
                        {
                            const volScalarField& rdt =
                                fv::localEulerDdt::localRDeltaT(mesh);
                            trdto = rdt.boundaryField()[patchi]
                                .patchInternalField();
                            trdtn = tmp<scalarField>
                            (
                                rdt.boundaryField()[patchi]
                            );
                        }

                        for (label k = 0; k < np; k++)
                        {
                            const scalar w = cdw[k];
                            pGammaP[k] =
                                msf[k]
                               /(w/rAUo[k] + (1.0 - w)/rAUn[k]);
                            rhofP[k] =
                                w*rhoo[k] + (1.0 - w)*rhon[k];
                            const scalar rhorAUf =
                                w*rhoo[k]*rAUo[k]
                              + (1.0 - w)*rhon[k]*rAUn[k];
                            const vector rhoUoldf =
                                w*rhoOldo[k]*UOldo[k]
                              + (1.0 - w)*rhoOldn[k]*UOldn[k];
                            const scalar phiCorr =
                                phiOldb[k] - (rhoUoldf & Sfb[k]);
                            const scalar coeff =
                                1.0
                              - min
                                (
                                    mag(phiCorr)
                                   /(mag(phiOldb[k]) + SMALL),
                                    scalar(1)
                                );
                            const scalar rdtf =
                                LTS
                              ? (
                                    w*trdto()[k]
                                  + (1.0 - w)*trdtn()[k]
                                )
                              : 1.0/runTime.deltaTValue();
                            const vector Hf =
                                w*Ho[k] + (1.0 - w)*Hn[k];
                            phbP[k] =
                                (Hf & Sfb[k])
                              + rcDdtScale*rhorAUf*coeff*rdtf*phiCorr
                               /rhofP[k];
                        }
                    }

                    for (label k = 0; k < np; k++)
                    {
                        bDiagA[off + k] = -pGammaP[k]*gic[k];
                        bSrcA[off + k]  = 0;
                        phiBA[off + k]  = phbP[k];
                        gpuPEqnParB_[offPar + k] = pGammaP[k]*gbc[k];
                        gpuPEqnParPhb_[offPar + k] = phbP[k];
                        gpuPEqnParRhof_[offPar + k] = rhofP[k];
                    }
                    offPar += np;
                }
                else
                {
                    for (label k = 0; k < np; k++)
                    {
                        bDiagA[off + k] = -pGamma[k]*gic[k];
                        bSrcA[off + k]  =  pGamma[k]*gbc[k];
                        phiBA[off + k]  =  phb[k];
                    }
                }
                off += np;
            }

            if (offPar > 0 && rgpPEqnParCoeffs(gpuPEqnParB_.begin()))
            {
                FatalErrorInFunction
                    << "rgpPEqnParCoeffs: " << rgpPEqnLastError()
                    << exit(FatalError);
            }
        }

        // 병렬: refCell은 소유 랭크에서만 유효(타 랭크 -1) — OF
        // setReference와 동일하게 소유 랭크만 diag/b를 수정한다
        const bool needRef =
            p.needReference() && pressureReference.refCell() >= 0;
        const label refCell = needRef ? pressureReference.refCell() : 0;
        const scalar refValue = needRef ? pressureReference.refValue() : 0;
        const scalar dtInv = LTS ? 1.0 : 1.0/runTime.deltaTValue();

        // AMD(-fvc::laplacian(Dr,rho)/rho): 저장 소스로 주입 (b += X*V)
        tmp<volScalarField> tamdSrc;
        if (LADrhoCoeff > 0)
        {
            tamdSrc = fvc::laplacian(Dr, rho)/rho;
        }
        const double* amdSrc =
            tamdSrc.valid()
          ? tamdSrc().primitiveField().begin() : nullptr;
        const double* rdtCell =
            LTS
          ? fv::localEulerDdt::localRDeltaT(mesh).primitiveField().begin()
          : nullptr;

        // ── 검증 모드: CPU fvMatrix 계수와 대조 (devChain 미지원) ────
        if (gpuPEqnCheck_ && devChain)
        {
            FatalErrorInFunction
                << "gpuPEqnCheck requires the host fvc chain "
                << "(disable gpuUEqn or gpuPEqnCheck)"
                << exit(FatalError);
        }
        if (gpuPEqnCheck_)
        {
            fvScalarMatrix pDDtEqnChk
            (
                tpsis()*fvm::ddt(p)
              + fvc::div(tphiHbyAv())
            );
            if (LADrhoCoeff > 0)
            {
                pDDtEqnChk -= fvc::laplacian(Dr, rho)/rho;
            }
            fvScalarMatrix pEqnChk
            (
                pDDtEqnChk - fvm::laplacian(trAUf(), p)
            );
            pEqnChk.setReference
            (
                pressureReference.refCell(), pressureReference.refValue()
            );

            scalarField diagC(pEqnChk.diag());
            scalarField srcC(pEqnChk.source());
            forAll(p.boundaryField(), patchi)
            {
                const labelUList& fc = mesh.boundary()[patchi].faceCells();
                const scalarField& iC = pEqnChk.internalCoeffs()[patchi];
                const scalarField& bC = pEqnChk.boundaryCoeffs()[patchi];
                forAll(fc, k)
                {
                    diagC[fc[k]] += iC[k];
                    srcC[fc[k]] += bC[k];
                }
            }

            List<double> dG(nc), uG(nif), bG(nc);
            const int rc = rgpPEqnAssembleDump
            (
                dtInv, rdtCell, amdSrc,
                trAUf().primitiveField().begin(),
                tpsis().primitiveField().begin(),
                p.oldTime().primitiveField().begin(),
                tphiHbyAv().primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                dG.begin(), uG.begin(), bG.begin()
            );
            if (rc)
            {
                FatalErrorInFunction
                    << "rgpPEqnAssembleDump: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            scalar dMax = 0, uMax = 0, bMax = 0;
            forAll(diagC, i)
            {
                dMax = max(dMax, mag(dG[i] - diagC[i])
                    /max(mag(diagC[i]), small));
                bMax = max(bMax, mag(bG[i] - srcC[i])
                    /max(mag(srcC[i]), small));
            }
            const scalarField& uC = pEqnChk.upper();
            forAll(uC, f)
            {
                uMax = max(uMax, mag(uG[f] - uC[f])
                    /max(mag(uC[f]), small));
            }
            Info<< "gpuPEqnCheck: maxRel diag = " << dMax
                << ", upper = " << uMax << ", source = " << bMax << endl;
        }

        // ── 솔버 컨트롤 (fvSolution p/pFinal 자동 선택) + 솔브 ────────
        // fvMatrix::solver()와 동일 규약: 최종 보정자에서 "pFinal" 선택
        const dictionary& sd = mesh.solution().solverDict
        (
            p.select
            (
                !mesh.schemes().steady()
             && solutionControl::finalIteration(mesh)
            )
        );
        const scalar tol = sd.lookup<scalar>("tolerance");
        const scalar rtol = sd.lookupOrDefault<scalar>("relTol", 0);
        const label maxIter = sd.lookupOrDefault<label>("maxIter", 1000);

        gpuPEqnFlux_.setSize(nif);
        double res0 = 0, resF = 0;
        int nIter = 0;

        std::chrono::steady_clock::time_point tS;
        if (thermoTimings_) { tS = std::chrono::steady_clock::now(); }

        if (gpuPEqnSolver_ == "amgx")
        {
            // ── AmgX 직결: 디바이스 CSR 조립 → AmgX PCG+AMG(계층 동결)
            //    → OF 규약 잔차로 수렴 확인(미달 시 톨 조여 재솔브) ──
            typedef int (*AmgxFn)
            (
                int, int, const int*, const int*,
                void*, void*, void*, int, int, int*
            );
            typedef const char* (*AmgxErrFn)(void);
            static AmgxFn amgxFn = nullptr;
            static AmgxErrFn amgxErrFn = nullptr;
            if (!amgxFn)
            {
                amgxFn = reinterpret_cast<AmgxFn>
                    (dlsym(RTLD_DEFAULT, "rgpPEqnAmgxSolve"));
                amgxErrFn = reinterpret_cast<AmgxErrFn>
                    (dlsym(RTLD_DEFAULT, "rgpPEqnAmgxErr"));
                if (!amgxFn || !amgxErrFn)
                {
                    FatalErrorInFunction
                        << "gpuPEqnSolver amgx requires libRGP13amgx.so "
                        << "in the controlDict libs list"
                        << exit(FatalError);
                }
            }

            double normFactor = 0;
            int rc = rgpPEqnAssembleCsr
            (
                dtInv, rdtCell, amdSrc,
                devChain ? nullptr : trAUf().primitiveField().begin(),
                devChain ? nullptr : tpsis().primitiveField().begin(),
                p.oldTime().primitiveField().begin(),
                p.primitiveField().begin(),
                devChain ? nullptr : tphiHbyAv().primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                &normFactor, &res0
            );
            if (rc)
            {
                FatalErrorInFunction
                    << "rgpPEqnAssembleCsr: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            const label setupInterval =
                fgmTable_.lookupOrDefault<label>
                (
                    "gpuPEqnSetupInterval", 25
                );

            const scalar target = max(tol, rtol*res0);
            resF = res0;
            // AmgX per-solve 톨 주입은 라이브 솔버에 반영되지 않으므로
            // (config_add_parameters 잠복 이슈) 고정 4-iter 블록으로
            // 돌리고 OF 규약 잔차로 외부에서 수렴 판정한다. x는 블록 간
            // 연속이라 정확히 이어서 수렴한다. 블록 수는 fvSolution
            // maxIter를 존중 (블록당 4 iter).
            const int maxRounds = max(label(1), (maxIter + 3)/4);
            for (int round = 0; round < maxRounds && resF >= target; round++)
            {
                int it = 0;
                rc = amgxFn
                (
                    nc, gpuPEqnNnz_,
                    rgpPEqnCsrRowPtr(), rgpPEqnCsrColInd(),
                    rgpPEqnDevValues(), rgpPEqnDevB(), rgpPEqnDevX(),
                    round == 0 ? 1 : 0,   // newSolve: 계층 동결 주기는
                    setupInterval,        // 압력 솔브 횟수 기준으로 센다
                    &it
                );
                if (rc)
                {
                    FatalErrorInFunction
                        << "rgpPEqnAmgxSolve: " << amgxErrFn()
                        << exit(FatalError);
                }
                nIter += it;
                if (rgpPEqnResidual(normFactor, &resF))
                {
                    FatalErrorInFunction
                        << "rgpPEqnResidual: " << rgpPEqnLastError()
                        << exit(FatalError);
                }
                // NaN은 모든 비교에서 false → 루프가 수렴처럼 조용히
                // 끝나므로 명시적으로 잡는다 (동결 계층 발산 감지)
                if (resF != resF)
                {
                    FatalErrorInFunction
                        << "AmgX p-solve diverged (NaN residual) -- "
                        << "frozen AMG hierarchy may be stale; reduce "
                        << "gpuPEqnSetupInterval"
                        << exit(FatalError);
                }
            }

            // devChain은 Finish2가 p/플럭스를 회수한다 — 이중 D2H 회피
            if
            (
                !devChain
             && rgpPEqnFinish
                (
                    p.primitiveFieldRef().begin(), gpuPEqnFlux_.begin()
                )
            )
            {
                FatalErrorInFunction
                    << "rgpPEqnFinish: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            Info<< "rgpAmgX: Solving for p, Initial residual = " << res0
                << ", Final residual = " << resF
                << ", No Iterations " << nIter << endl;
        }
        else
        {
            rgpPEqnSetPrecon(gpuPEqnPrecon_ == "dic" ? 1 : 0);

            const int rc = rgpPEqnSolve
            (
                dtInv, rdtCell, amdSrc,
                devChain ? nullptr : trAUf().primitiveField().begin(),
                devChain ? nullptr : tpsis().primitiveField().begin(),
                p.oldTime().primitiveField().begin(),
                p.primitiveField().begin(),
                devChain ? nullptr : tphiHbyAv().primitiveField().begin(),
                phiBA, bDiagA, bSrcA,
                needRef, refCell, refValue,
                tol, rtol, maxIter,
                devChain ? nullptr : p.primitiveFieldRef().begin(),
                devChain ? nullptr : gpuPEqnFlux_.begin(),
                &res0, &resF, &nIter
            );
            if (rc)
            {
                FatalErrorInFunction
                    << "rgpPEqnSolve: " << rgpPEqnLastError()
                    << exit(FatalError);
            }

            Info<< (gpuPEqnPrecon_ == "dic" ? "rgpDICPCG" : "rgpPCG")
                << ":  Solving for p, Initial residual = " << res0
                << ", Final residual = " << resF
                << ", No Iterations " << nIter << endl;
        }

        if (thermoTimings_)
        {
            Info<< "pCorr pEqn.solve = "
                << std::chrono::duration<double>
                   (std::chrono::steady_clock::now() - tS).count()
                << " s" << endl;
        }

        // devChain: 솔브된 p를 먼저 회수해야 correctBoundaryConditions가
        // 새 내부장 기준으로 경계를 갱신한다 (pOut=null 계약 — 리뷰 F1:
        // cBC가 솔브-전 p로 돌면 경계 p가 한 솔브 지연되어 ∇p 오염)
        if (devChain)
        {
            if (rgpPEqnFinish2
                (
                    p.primitiveFieldRef().begin(),
                    gpuPEqnFlux_.begin()
                ))
            {
                FatalErrorInFunction
                    << "rgpPEqnFinish2: " << rgpPEqnLastError()
                    << exit(FatalError);
            }
        }

        p.correctBoundaryConditions();

        // ── 질량 플럭스 재구성: phi = rhof*(phiHbyAv + pEqn.flux()) ──
        {
            if (devChain)
            {
                scalarField& phic = phi.primitiveFieldRef();
                forAll(phic, f) { phic[f] = gpuPEqnFlux_[f]; }
            }
            else
            {
                scalarField& phic = phi.primitiveFieldRef();
                const scalarField& rf = trhof().primitiveField();
                const scalarField& ph = tphiHbyAv().primitiveField();
                forAll(phic, f)
                {
                    phic[f] = rf[f]*(ph[f] + gpuPEqnFlux_[f]);
                }
            }

            // 경계: fvMatrix::flux() 규약 — iC*p_internal - bC
            // (devChain: rhof_b = rho_b, phiHbyAv_b = HbyA_b & Sf_b)
            // processor 패치: bC 기여는 인터페이스 계수 × 이웃 p —
            // cBC 직후라 pp가 이웃 랭크의 p를 담고 있다
            label off = 0;
            label offPar = 0;
            forAll(p.boundaryField(), patchi)
            {
                const fvPatchScalarField& pp = p.boundaryField()[patchi];
                const label np = pp.size();
                if (np == 0) continue;

                const scalarField pi(pp.patchInternalField());
                const scalarField rfb
                (
                    devChain
                  ? scalarField(rho.boundaryField()[patchi])
                  : scalarField(trhof().boundaryField()[patchi])
                );
                const scalarField phb
                (
                    devChain
                  ? scalarField
                    (
                        HbyA.boundaryField()[patchi]
                      & mesh.Sf().boundaryField()[patchi]
                    )
                  : scalarField(tphiHbyAv().boundaryField()[patchi])
                );
                scalarField& phibf = phi.boundaryFieldRef()[patchi];

                if (pp.coupled())
                {
                    for (label k = 0; k < np; k++)
                    {
                        const scalar fluxB =
                            bDiagA[off + k]*pi[k]
                          - gpuPEqnParB_[offPar + k]*pp[k];
                        // devChain: concat 때 저장한 proc-면 rhof/phb
                        // (조화 rAUf·ddtCorr 포함) — 비-devChain은
                        // fvc 체인 boundaryField가 이미 정확
                        if (devChain)
                        {
                            phibf[k] =
                                gpuPEqnParRhof_[offPar + k]
                               *(gpuPEqnParPhb_[offPar + k] + fluxB);
                        }
                        else
                        {
                            phibf[k] = rfb[k]*(phb[k] + fluxB);
                        }
                    }
                    offPar += np;
                }
                else
                {
                    for (label k = 0; k < np; k++)
                    {
                        const scalar fluxB =
                            bDiagA[off + k]*pi[k] - bSrcA[off + k];
                        phibf[k] = rfb[k]*(phb[k] + fluxB);
                    }
                }
                off += np;
            }
        }
    }
    else
    {
        fvScalarMatrix pDDtEqn
        (
            tpsis()*fvm::ddt(p)
          + fvc::div(tphiHbyAv())
        );
        if (LADrhoCoeff > 0)
        {
            // AMD 항도 계수가 켜졌을 때만 조립 (0 기여, 생략은 비트-동일)
            pDDtEqn -= fvc::laplacian(Dr, rho)/rho;
        }

        while (pimple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn
            (
                pDDtEqn - fvm::laplacian(trAUf(), p)
            );

            pEqn.setReference
            (
                pressureReference.refCell(),
                pressureReference.refValue()
            );

            fvConstraints().constrain(pEqn);

            {
                std::chrono::steady_clock::time_point tS;
                if (thermoTimings_)
                {
                    tS = std::chrono::steady_clock::now();
                }
                pEqn.solve();
                if (thermoTimings_)
                {
                    Info<< "pCorr pEqn.solve = "
                        << std::chrono::duration<double>
                           (std::chrono::steady_clock::now() - tS).count()
                        << " s" << endl;
                }
            }

            if (pimple.finalNonOrthogonalIter())
            {
                // Reconstruct the mass flux from the corrected volumetric
                // flux
                phi = trhof()*(tphiHbyAv() + pEqn.flux());
            }
        }
    }

    // --- POSITIVITY GUARD: floor the SOLVED pressure to a physical minimum ---
    // ROOT ISSUE (2026-07-02): the swirl low-pressure core + stiff cold-LOX SRK
    // drive the pEqn to undershoot the SOLVED pressure NEGATIVE (min p ~ -1e7
    // observed even after the EOS-internal density floor). Absolute p<0 is
    // thermodynamically impossible; via SRK rho = p/(Z R T) it makes rho<0 and,
    // through -grad(p), pumps |U| overshoots that collapse dt. The EOS-internal
    // floor only protects rho -- the solved p field itself must also be bounded.
    // Floor p here (right after the solve, before correctRho/relax/U) so every
    // downstream step sees a positive pressure. pMinPa read each step (default 0
    // = off). Positivity guard confined to sub-physical cells, paired with the
    // EOS floor; a physically-exact version would use p_sat(T) instead.
    const scalar pMinPa
    (
        pimple.dict().lookupOrDefault<scalar>("pMinPa", scalar(0))
    );
    if (pMinPa > 0)
    {
        p = max(p, dimensionedScalar("pMinPa", p.dimensions(), pMinPa));
        p.correctBoundaryConditions();
    }

    // Upper positivity-guard twin of pMinPa: cap the solved pressure. Used as
    // a TEMPORARY surgical bound while the ignition-phase flame-zone spike
    // dissolves (fvConstraints limitPressure proved inert here even after
    // constrain(p) -- root cause not chased; this knob is on the proven
    // pMinPa path). pMaxPa read each step (default 0 = off).
    const scalar pMaxPa
    (
        pimple.dict().lookupOrDefault<scalar>("pMaxPa", scalar(0))
    );
    if (pMaxPa > 0)
    {
        p = min(p, dimensionedScalar("pMaxPa", p.dimensions(), pMaxPa));
        p.correctBoundaryConditions();
    }

    // Thermodynamic density update: the stock SIMPLErho increment
    // correctRho(psi*dp) is DISABLED in the PEP path. At a flame-zone
    // pressure spike the increment is huge (psi_gas ~1e-5 x dp ~ -4e8 ->
    // drho ~ -4000) and drove the THERMO density to -4869 kg/m3 in written
    // states -- and the per-corrector updateManifold()+thermo_.correct()+
    // rho_ re-sync at the top of this function already provides the full
    // EOS-consistent density update (one-corrector lag, nOuter >= 3).

    // Continuity diagnostics (base isothermalFluid::continuityErrors wraps
    // this grandparent call)
    fluidSolver::continuityErrors(rho, thermo.rho(), phi);

    // Explicitly relax pressure for momentum corrector
    p.relax();

    // Apply the field-level fvConstraints (limitPressure min/max). The stock
    // correctPressure applies them after the solve; the PEP override had
    // omitted this call, leaving constant/fvConstraints limitP INERT (both
    // its min and the surgical max) -- only the pMinPa floor was active.
    fvConstraints().constrain(p);

    if (gpuUEqn_ && gpuPEqn_)
    {
        // U = HbyA − rAU∇p — grad p·연산을 디바이스에서 (HbyA/rAU 상주)
        const label nc = mesh.nCells();
        label nbf = 0;
        forAll(p.boundaryField(), patchi)
        {
            nbf += p.boundaryField()[patchi].size();
        }
        double* pB = gpuUBuf_.begin() + 15*nc + 3*nbf;   // pB 슬롯 재사용
        double* U3out = gpuUBuf_.begin() + 6*nc;
        {
            label off = 0;
            forAll(p.boundaryField(), patchi)
            {
                const fvPatchScalarField& pbf =
                    p.boundaryField()[patchi];
                if (pbf.coupled())
                {
                    // gaussGrad coupled: 면 값 (cBC 직후 이웃 p 보유)
                    const scalarField w
                    (
                        mesh.surfaceInterpolation::weights()
                            .boundaryField()[patchi]
                    );
                    const scalarField po(pbf.patchInternalField());
                    forAll(pbf, k)
                    {
                        pB[off + k] =
                            w[k]*po[k] + (1.0 - w[k])*pbf[k];
                    }
                }
                else
                {
                    forAll(pbf, k) { pB[off + k] = pbf[k]; }
                }
                off += pbf.size();
            }
        }
        if (rgpPCorrU(p.primitiveField().begin(), pB, U3out))
        {
            FatalErrorInFunction
                << "rgpPCorrU: " << rgpPEqnLastError()
                << exit(FatalError);
        }
        vectorField& Uc = U.primitiveFieldRef();
        for (label i = 0; i < nc; i++)
        {
            for (label k = 0; k < 3; k++)
            {
                if (mesh.solutionD()[k] == 1)
                {
                    Uc[i][k] = U3out[k*nc + i];
                }
            }
        }
    }
    else
    {
        U = HbyA - rAAtU*fvc::grad(p);
    }
    U.correctBoundaryConditions();
    fvConstraints().constrain(U);
    K = 0.5*magSqr(U);

    // SIMPLErho: density from the equation of state
    if (pimple.simpleRho())
    {
        rho = thermo.rho();
        rho.relax();
    }

    // Correct rhoUf if the mesh is moving
    fvc::correctRhoUf(rhoUf, rho, U, phi, MRF);

    if (thermo.dpdt())
    {
        dpdt = fvc::ddt(p);
    }

    if (thermoTimings_)
    {
        Info<< "pCorr total = "
            << std::chrono::duration<double>
               (std::chrono::steady_clock::now() - tTot).count()
            << " s" << endl;
    }
}


void Foam::solvers::fgmFluid::pressureCorrector()
{
    if (buoyancy.valid())
    {
        FatalErrorInFunction
            << "fgmFluid PEP pressure corrector does not support buoyant "
            << "(p_rgh) cases." << exit(FatalError);
    }

    while (pimple.correct())
    {
        correctPressurePEP();
    }

    tUEqn.clear();
}


// ************************************************************************* //
