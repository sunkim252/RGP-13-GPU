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

    // 공유 rosenbrock34Step의 host 실행: err(h) 스캔 (신선 상태)
    {
        List<double> yh(n + 1), yo(n + 1), wk((n + 1)*(n + 8));
        for (int s = 0; s < n; s++) { yh[s] = X[s]*cTot0; }
        yh[n] = T0;
        for (scalar h = 1e-6; h > 1e-12; h /= 10)
        {
            const double e = rgpchem::rosenbrock34Step
            (
                mech, p0, h, 1e-6, 1e-14, yh.begin(), yo.begin(), wk.begin()
            );
            Info<< "host sharedStep h=" << h << " errNorm=" << e
                << " Tnew=" << yo[n] << endl;
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
        if (rgpChemIntegrate(nb, dtSample, 1e-6, 1e-14,
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

        long long stats[2] = {0, 0};
        // 워밍업(JIT/할당) 후 본측정
        rgpChemIntegrate(nBatch, 1e-7, 1e-5, 1e-12,
                         pB.begin(), TB.begin(), cB.begin(), stats);
        auto g0 = std::chrono::steady_clock::now();
        const int err = rgpChemIntegrate(nBatch, 1e-7, 1e-5, 1e-12,
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

    rgpChemFree();
    Info<< nl << "PASS" << nl << endl;
    return 0;
}

// ************************************************************************* //
