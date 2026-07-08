/*---------------------------------------------------------------------------*\
Application
    Test-rgpChemGPU

Description
    chemGPU 검증 유틸 — Burke 2012 H2/O2 (9종 19단계).

    1) 0-D 등압 단열 점화 (p=1atm, T0=1100K, 양론 H2-공기):
       CPU 레퍼런스 = 동일 RHS(rgpChemRHS.H) + OpenFOAM Rosenbrock34
       (독립 구현) vs GPU 배치 적분기. T(t) 프로파일과 점화지연
       tau_ign(max dT/dt) 비교.
    2) 처리율: N셀 배치를 CFD 스텝(dt=1e-7 s)으로 적분, cells/s 측정
       → GPU 화학 오프로드의 격자 규모 산정 근거.

Usage
    Test-rgpChemGPU [nBatch]
\*---------------------------------------------------------------------------*/

#include "scalarField.H"
#include "scalarMatrices.H"
#include "ODESystem.H"
#include "ODESolver.H"
#include "Rosenbrock34.H"
#include "dictionary.H"

#include "gpu/rgpChemTypes.H"
#include "gpu/rgpChemRHS.H"
#include "gpu/rgpH2O2Burke.H"

#include <chrono>

using namespace Foam;

// OF ODESolver용 래퍼: RHS는 GPU와 동일 소스, Jacobian은 FD
class burkeODE : public ODESystem
{
    const rgpChemMech& m_;
    const scalar p_;

public:

    mutable label nRHS = 0;

    burkeODE(const rgpChemMech& m, const scalar p) : m_(m), p_(p) {}

    label nEqns() const { return m_.nSpecies + 1; }

    void derivatives
    (
        const scalar, const scalarField& y, const label, scalarField& dydx
    ) const
    {
        nRHS++;
        rgpchem::chemRHS(m_, p_, y.begin(), dydx.begin());
    }

    void jacobian
    (
        const scalar, const scalarField& y, const label,
        scalarField& dfdx, scalarSquareMatrix& dfdy
    ) const
    {
        const label n = nEqns();
        dfdx = 0.0;
        scalarField y1(y), f0(n), f1(n);
        rgpchem::chemRHS(m_, p_, y.begin(), f0.begin());
        for (label j = 0; j < n; j++)
        {
            const scalar d = max(mag(y[j])*1e-7, 1e-14);
            y1[j] = y[j] + d;
            rgpchem::chemRHS(m_, p_, y1.begin(), f1.begin());
            y1[j] = y[j];
            for (label i = 0; i < n; i++)
            {
                dfdy(i, j) = (f1[i] - f0[i])/d;
            }
        }
    }
};


int main(int argc, char *argv[])
{
    const label nBatch = (argc > 1) ? atoi(argv[1]) : 200000;

    rgpChemMech mech;
    rgpBuildBurke2012(mech);
    const int n = mech.nSpecies;
    Info<< "Burke 2012 H2/O2: " << n << " species, " << mech.nReactions
        << " reversible reactions" << nl;

    // 초기 상태: p=1atm, T0=1100K, 양론 H2-공기 (X: 2/1/3.76)
    const scalar p0 = 101325.0, T0 = 1100.0;
    const scalar Xtot = 2.0 + 1.0 + 3.76;
    scalar X[nSp9] = {0};
    X[iH2] = 2.0/Xtot; X[iO2] = 1.0/Xtot; X[iN2] = 3.76/Xtot;
    const scalar cTot0 = p0/(mech.RR*T0);

    // ── 1) CPU 레퍼런스: OF Rosenbrock34로 T(t) & tau_ign ──────────────
    burkeODE sys(mech, p0);
    dictionary odeDict;
    odeDict.add("solver", word("Rosenbrock34"));
    odeDict.add("absTol", 1e-14);
    odeDict.add("relTol", 1e-6);
    autoPtr<ODESolver> ode = ODESolver::New(sys, odeDict);

    const scalar tEnd = 5e-4, dtSample = 1e-6;
    scalarField y(n + 1, 0.0);
    for (int s = 0; s < n; s++) { y[s] = X[s]*cTot0; }
    y[n] = T0;

    scalar tauCPU = -1, TmaxRateCPU = 0, dxTry = 1e-9, tPrev = 0;
    scalar Tprev = T0;
    auto cpu0 = std::chrono::steady_clock::now();
    // 궤적 스냅샷 (GPU 스텝-일관성 검증용): snap[k] = y(t_k), t_k = k*dtSample
    const label nSnap = label(tEnd/dtSample);
    List<scalarField> snap(nSnap + 1);
    snap[0] = y;
    for (scalar t = 0; t < tEnd; t += dtSample)
    {
        ode->solve(t, t + dtSample, y, 0, dxTry);
        const scalar rate = (y[n] - Tprev)/dtSample;
        if (rate > TmaxRateCPU) { TmaxRateCPU = rate; tauCPU = t + dtSample; }
        Tprev = y[n];
        tPrev = t;
        const label k = label((t + dtSample)/dtSample + 0.5);
        if (k <= nSnap) { snap[k] = y; }
    }
    const scalar cpuSec =
        std::chrono::duration<double>
        (std::chrono::steady_clock::now() - cpu0).count();
    const scalar TfinalCPU = y[n];
    Info<< "CPU(OF Rosenbrock34): tau_ign = " << tauCPU*1e6 << " us, "
        << "T_final = " << TfinalCPU << " K  (" << cpuSec << " s, "
        << sys.nRHS << " RHS evals -> ~" << sys.nRHS/16 << " substeps)" << nl;

    // ── 2) GPU: 동일 문제 1셀 + 프로파일 대조 ──────────────────────────
    if (rgpChemInit(-1))
    {
        Serr<< "FAIL: " << rgpChemLastError() << endl;
        return 1;
    }
    if (rgpChemUpload(&mech))
    {
        Serr<< "FAIL: " << rgpChemLastError() << endl;
        return 1;
    }

    // 진단: 동일 상태에서 device J vs host J 대조 (신선/점화중/평형)
    {
        const label ks[3] = {0, label(tauCPU/dtSample), nSnap - 1};
        const label neq = n + 1;
        for (int d = 0; d < 3; d++)
        {
            // host: 동일 FD 공식
            scalarField yh(snap[ks[d]]);
            scalarField f0(neq), f1(neq);
            List<double> Jh(neq*neq), Jd(neq*neq), dy0d(neq);
            rgpchem::chemRHS(mech, p0, yh.begin(), f0.begin());
            for (label j = 0; j < neq; j++)
            {
                const scalar yj = yh[j];
                const scalar dd = max(mag(yj)*1e-7, 1e-14);
                yh[j] = yj + dd;
                rgpchem::chemRHS(mech, p0, yh.begin(), f1.begin());
                yh[j] = yj;
                for (label i = 0; i < neq; i++)
                {
                    Jh[i*neq + j] = (f1[i] - f0[i])/dd;
                }
            }
            // device
            if (rgpChemDebugJac(p0, snap[ks[d]].begin(),
                                dy0d.begin(), Jd.begin()))
            {
                Serr<< "FAIL: " << rgpChemLastError() << endl;
                return 1;
            }
            scalar worstJ = 0, worstF = 0;
            label wi = -1, wj = -1;
            for (label i = 0; i < neq; i++)
            {
                worstF = max(worstF,
                    mag(dy0d[i] - f0[i])/max(mag(f0[i]), scalar(1e-30)));
                for (label j = 0; j < neq; j++)
                {
                    const scalar dJ = mag(Jd[i*neq + j] - Jh[i*neq + j])
                        /max(mag(Jh[i*neq + j]), scalar(1e3));
                    if (dJ > worstJ) { worstJ = dJ; wi = i; wj = j; }
                }
            }
            Info<< "Jdiag k=" << ks[d] << " T=" << snap[ks[d]][n]
                << ": max reldiff RHS=" << worstF << " J=" << worstJ
                << " @(" << wi << "," << wj << ") host="
                << Jh[wi*neq + wj] << " dev=" << Jd[wi*neq + wj] << endl;
        }
    }

    // 대조: OF Rosenbrock34::solve(고정 dx) vs 공유 스텝 — 같은 신선 상태
    {
        const label neq = n + 1;
        Rosenbrock34 ros(sys, odeDict);

        scalarField y0f(neq, 0.0);
        for (int s = 0; s < n; s++) { y0f[s] = X[s]*cTot0; }
        y0f[n] = T0;
        scalarField dydx0(neq), yOF(neq);
        sys.derivatives(0, y0f, 0, dydx0);

        List<double> yMine(neq), wk(neq*(neq + 8)), rows(neq);

        for (scalar h = 1e-6; h > 1e-12; h /= 10)
        {
            const scalar errOF = ros.solve(0, y0f, 0, dydx0, h, yOF);
            const double errMine = rgpchem::rosenbrock34Step
            (
                mech, p0, h, 1e-6, 1e-14,
                y0f.begin(), yMine.begin(), wk.begin(), rows.begin()
            );
            scalar dy = 0;
            for (label i = 0; i < neq; i++)
            {
                dy = max(dy, mag(yOF[i] - yMine[i])
                    /max(mag(yOF[i]), scalar(1e-30)));
            }
            label iw = 0;
            for (label i = 1; i < neq; i++)
            {
                if (rows[i] > rows[iw]) { iw = i; }
            }
            Info<< "h=" << h << "  errOF=" << errOF
                << "  errMine=" << errMine
                << "  maxRelDy(OF vs mine)=" << dy
                << "  worstRow=" << iw << " (|err|/sc=" << rows[iw]
                << ")" << endl;
        }

        // LU 단독 대조: 같은 A=(I/(γh)−J)에 대해 내 LU vs OF LU로 k1 비교
        {
            const scalar h = 1e-8, gam = 0.5;
            scalarField dfdx(neq);
            scalarSquareMatrix dfdy(neq);
            sys.jacobian(0, y0f, 0, dfdx, dfdy);

            // OF 경로
            scalarSquareMatrix A(neq), A0(neq);
            for (label i = 0; i < neq; i++)
            {
                for (label j = 0; j < neq; j++) { A(i, j) = -dfdy(i, j); }
                A(i, i) += 1.0/(gam*h);
                for (label j = 0; j < neq; j++) { A0(i, j) = A(i, j); }
            }
            labelList pivOF(neq);
            LUDecompose(A, pivOF);
            scalarField k1OF(dydx0);
            LUBacksubstitute(A, pivOF, k1OF);

            // 내 경로 (동일 J 사용)
            List<double> LU(neq*neq), k1M(neq);
            List<int> pivM(neq);
            for (label i = 0; i < neq; i++)
            {
                for (label j = 0; j < neq; j++)
                {
                    LU[i*neq + j] = -dfdy(i, j);
                }
                LU[i*neq + i] += 1.0/(gam*h);
                k1M[i] = dydx0[i];
            }
            rgpchem::luDecomposeHD(LU.begin(), pivM.begin(), neq);
            rgpchem::luSolveHD(LU.begin(), pivM.begin(), k1M.begin(), neq);

            Info<< "LU k1 대조 (h=1e-8):" << endl;
            scalar resOF = 0, resM = 0, fmax0 = 0;
            for (label i = 0; i < neq; i++)
            {
                scalar sOF = 0, sM = 0;
                for (label j = 0; j < neq; j++)
                {
                    sOF += A0(i, j)*k1OF[j];
                    sM += A0(i, j)*k1M[j];
                }
                resOF = max(resOF, mag(sOF - dydx0[i]));
                resM = max(resM, mag(sM - dydx0[i]));
                fmax0 = max(fmax0, mag(dydx0[i]));
            }
            Info<< "  잔차 |A·k1-f|max: OF=" << resOF << "  mine=" << resM
                << "  (|f|max=" << fmax0 << ")" << endl;
        }
    }

    if (nBatch == 0) { Info<< "(J-diag only)" << nl; return 0; }

    // 스텝-일관성 검증: CPU 궤적 스냅샷 nSnap개를 '셀'로 묶어 단일
    // launch — 셀 k는 snap[k]에서 dtSample만큼 적분, snap[k+1]과 대조.
    // (GPU 1스레드 순차 적분은 ~100× 느려서 금지 — 병렬로 궤적 전체 검증)
    {
        const label nb = nSnap;
        List<double> pB(nb, double(p0)), TB(nb), cB(nb*n);
        for (label k = 0; k < nb; k++)
        {
            for (int s = 0; s < n; s++) { cB[k*n + s] = snap[k][s]; }
            TB[k] = snap[k][n];
        }
        List<double> dtB(nb, double(dtSample));
        if (rgpChemIntegrate(nb, dtB.begin(), 1e-6, 1e-14,
                             pB.begin(), TB.begin(), cB.begin(), nullptr))
        {
            Serr<< "FAIL: " << rgpChemLastError() << endl;
            return 1;
        }
        scalar worstT = 0;
        label kw = -1;
        for (label k = 0; k < nb; k++)
        {
            const scalar dT = mag(TB[k] - snap[k + 1][n])/snap[k + 1][n];
            if (dT > worstT) { worstT = dT; kw = k; }
        }
        Info<< "GPU step-consistency over " << nb << " trajectory segments: "
            << "max dT = " << worstT*100 << " % @ t=" << kw*dtSample*1e6
            << " us (T_cpu=" << snap[kw + 1][n] << " T_gpu=" << TB[kw]
            << ")" << nl;
        if (worstT > 0.01)
        {
            Info<< "FAIL (step-consistency mismatch)" << nl;
            return 1;
        }
    }

    // ── 3) 처리율: N셀, CFD 스텝 dt=1e-7, T0 900-1300K 분포 ────────────
    {
        Info<< nl << "Throughput: " << nBatch
            << " cells, one CFD step dt = 1e-7 s" << nl;
        List<double> pB(nBatch, double(p0)), TB(nBatch);
        List<double> cB(nBatch*n, 0.0);
        for (label i = 0; i < nBatch; i++)
        {
            const double Ti = 900.0 + 400.0*double(i)/max(nBatch - 1, 1);
            TB[i] = Ti;
            const double cT = p0/(mech.RR*Ti);
            for (int s = 0; s < n; s++) { cB[i*n + s] = X[s]*cT; }
        }

        List<double> dtB(nBatch, 1e-7);
        long long stats[2] = {0, 0};
        // 워밍업(JIT/할당) 후 본측정
        rgpChemIntegrate(nBatch, dtB.begin(), 1e-5, 1e-12,
                         pB.begin(), TB.begin(), cB.begin(), stats);
        auto g0 = std::chrono::steady_clock::now();
        const int err = rgpChemIntegrate(nBatch, dtB.begin(), 1e-5, 1e-12,
                                         pB.begin(), TB.begin(),
                                         cB.begin(), stats);
        const double gSec =
            std::chrono::duration<double>
            (std::chrono::steady_clock::now() - g0).count();
        if (err)
        {
            Serr<< "FAIL: " << rgpChemLastError() << endl;
            return 1;
        }
        Info<< "  wall = " << gSec << " s -> "
            << scalar(nBatch)/gSec/1e6 << " Mcells/s"
            << "  (substeps: total " << scalar(stats[0])
            << ", max/cell " << scalar(stats[1]) << ")" << nl;
    }

    // ── 4) 반응-중 처리율: 궤적 스냅샷(신선→점화→연소)을 순환 복제 ────
    {
        Info<< nl << "Reacting-mix throughput: " << nBatch
            << " cells from trajectory snapshots, dt = 1e-7 s" << nl;
        List<double> pB(nBatch, double(p0)), TB(nBatch);
        List<double> cB(nBatch*n, 0.0);
        for (label i = 0; i < nBatch; i++)
        {
            const scalarField& sk = snap[i % nSnap];
            for (int s = 0; s < n; s++) { cB[i*n + s] = sk[s]; }
            TB[i] = sk[n];
        }

        List<double> dtB(nBatch, 1e-7);
        long long stats[2] = {0, 0};
        rgpChemIntegrate(nBatch, dtB.begin(), 1e-5, 1e-12,
                         pB.begin(), TB.begin(), cB.begin(), stats);
        // (위 호출로 상태가 전진했지만 처리율 측정 목적이라 무방 — 재시드)
        for (label i = 0; i < nBatch; i++)
        {
            const scalarField& sk = snap[i % nSnap];
            for (int s = 0; s < n; s++) { cB[i*n + s] = sk[s]; }
            TB[i] = sk[n];
        }
        auto g0 = std::chrono::steady_clock::now();
        const int err = rgpChemIntegrate(nBatch, dtB.begin(), 1e-5, 1e-12,
                                         pB.begin(), TB.begin(),
                                         cB.begin(), stats);
        const double gSec =
            std::chrono::duration<double>
            (std::chrono::steady_clock::now() - g0).count();
        if (err)
        {
            Serr<< "FAIL: " << rgpChemLastError() << endl;
            return 1;
        }
        Info<< "  wall = " << gSec << " s -> "
            << scalar(nBatch)/gSec/1e6 << " Mcells/s"
            << "  (substeps: total " << scalar(stats[0])
            << ", mean " << scalar(stats[0])/nBatch
            << ", max/cell " << scalar(stats[1]) << ")" << nl;
    }

    rgpChemFree();
    Info<< nl << "PASS" << nl << endl;
    return 0;
}

// ************************************************************************* //
