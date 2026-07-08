/*---------------------------------------------------------------------------*\
Application
    Test-rgpThermoGPU

Description
    thermoGPU 커널 검증 유틸리티.

    종 데이터 딕셔너리(예: testCases/counterflow_fgmFluid/constant/
    thermo.compressibleGas4S)로 SRKchungTakaMixture를 구성한 뒤,
    (p, T, Y) 상태 그리드에서

      CPU 레퍼런스 : calcMixture와 동일한 공개 API 경로
                     (realGasCoeffs → thermo 블렌드 → updateEoS →
                      X 보정 → updateTRANS_noDiffusion → rho/mu/kappa)
      GPU          : rgpGpuBridge 업로드 + rgpGpuEvaluate

    의 rho/mu/kappa를 대조하고 최대 상대오차를 보고한다.

Usage
    Test-rgpThermoGPU <thermoDictFile> [maxRelErr]

    기본 합격 기준 1e-10 (FP 재배열/초월함수 ULP 차이 허용).
\*---------------------------------------------------------------------------*/

#include "IFstream.H"
#include "dictionary.H"
#include "scalarList.H"

#include "rfSpecie.H"
#include "SRKGas.H"
#include "janafThermo.H"
#include "robustSensibleEnthalpy.H"
#include "chungTransport.H"
#include "thermo.H"
#include "SRKchungTakaMixture.H"

#include "rgpGpuBridge.H"

using namespace Foam;

typedef chungTransport
<
    species::thermo
    <
        janafThermo<SRKGas<rfSpecie>>,
        robustSensibleEnthalpy
    >
> RGPThermo;

typedef SRKchungTakaMixture<RGPThermo> RGPMixture;


// CPU 레퍼런스: SRKchungTakaMixture::calcMixture의 라이브 경로 재현
// (compositionToX + realGasCoeffs + 블렌드 + X 보정 + updateTRANS_noDiffusion)
void cpuReference
(
    const RGPMixture& mix,
    const List<scalar>& Y,
    const scalar p,
    const scalar T,
    scalar& rho,
    scalar& mu,
    scalar& kappa,
    scalar& Cp,
    scalar& Cv,
    scalar& psi
)
{
    const label n = Y.size();

    // 베이스 thermo 질량 블렌드
    RGPThermo t(Y[0]*mix.specieThermos()[0]);
    for (label i = 1; i < n; i++)
    {
        t += Y[i]*mix.specieThermos()[i];
    }

    // 13개 실기체 혼합 계수 (라이브 믹싱 규칙 그대로)
    List<scalar> coeffs;
    mix.realGasCoeffs(Y, coeffs);
    t.updateEoS(coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);

    // compositionToX + X 보정 (calcMixture와 동일 산식)
    List<scalar> X(n), Yl(n);
    scalar sumXb = 0;
    forAll(Y, i) { sumXb += Y[i]/mix.gpuListW()[i]; }
    if (sumXb == 0) { sumXb = 1e-30; }
    forAll(Y, i)
    {
        X[i] = (Y[i]/mix.gpuListW()[i])/sumXb;
        if (X[i] <= 0) { X[i] = 0; }
    }

    scalar WmixCorrect = 0, sumXcorrected = 0;
    forAll(X, i) { X[i] += 1e-40; sumXcorrected += X[i]; }
    forAll(X, i)
    {
        X[i] /= sumXcorrected;
        WmixCorrect += X[i]*mix.gpuListW()[i];
    }
    forAll(Yl, i) { Yl[i] = X[i]*mix.gpuListW()[i]/WmixCorrect; }

    t.updateTRANS_noDiffusion
    (
        coeffs[5], coeffs[6], coeffs[7], coeffs[8], coeffs[9],
        coeffs[10], coeffs[11], coeffs[12], Yl, X
    );

    rho = t.rho(p, T);
    mu = t.mu(p, T);
    kappa = t.kappa(p, T);
    Cp = t.Cp(p, T);
    Cv = t.Cv(p, T);
    psi = t.psi(p, T);
}


int main(int argc, char *argv[])
{
    // mesh/case 없는 단독 유틸: argList 없이 인자만 직접 사용
    if (argc < 2)
    {
        Serr<< "Usage: Test-rgpThermoGPU <thermoDictFile> [maxRelErr]" << endl;
        return 1;
    }
    const fileName dictFile(argv[1]);
    const scalar tol = (argc > 2) ? atof(argv[2]) : 1e-10;

    Info<< "Reading species dictionary: " << dictFile << nl;
    IFstream is(dictFile);
    if (!is.good())
    {
        Serr<< "Cannot open " << dictFile << endl;
        return 1;
    }
    dictionary dict(is);

    RGPMixture mix(dict);
    const label n = mix.specieThermos().size();
    Info<< "Species (" << n << "): ";
    forAll(mix.specieThermos(), i)
    {
        Info<< mix.specieThermos()[i].name() << " ";
    }
    Info<< nl << "stableRoot = " << RGPThermo::stableRoot() << nl << endl;

    // ── 상태 그리드 ──────────────────────────────────────────────────
    const scalarList pList({1.0e5, 5.25e6, 1.0e7});
    const scalarList TList({100, 150, 300, 800, 1500, 3000});

    // 조성: 각 종 순수 + 대표 혼합 3종
    List<List<scalar>> comps;
    for (label i = 0; i < n; i++)
    {
        List<scalar> Y(n, scalar(0));
        Y[i] = 1.0;
        comps.append(Y);
    }
    {
        List<scalar> Y(n, scalar(0));        // 산화제/연료 전단층
        forAll(mix.specieThermos(), i)
        {
            const word& nm = mix.specieThermos()[i].name();
            if (nm == "O2") Y[i] = 0.7;
            if (nm == "KERO") Y[i] = 0.3;
        }
        comps.append(Y);
    }
    {
        List<scalar> Y(n, scalar(0));        // 연소 생성물
        forAll(mix.specieThermos(), i)
        {
            const word& nm = mix.specieThermos()[i].name();
            if (nm == "N2") Y[i] = 0.5;
            if (nm == "CO2") Y[i] = 0.2;
            if (nm == "H2O") Y[i] = 0.2;
            if (nm == "CO") Y[i] = 0.1;
        }
        comps.append(Y);
    }
    {
        List<scalar> Y(n, 1.0/n);            // 균등 혼합
        comps.append(Y);
    }

    const label nStates = pList.size()*TList.size()*comps.size();
    Info<< "State grid: " << pList.size() << " p x " << TList.size()
        << " T x " << comps.size() << " comps = " << nStates
        << " states" << nl << endl;

    // ── CPU 레퍼런스 + GPU 입력 패킹 ─────────────────────────────────
    scalarList pF(nStates), TF(nStates), YF(nStates*n);
    scalarList rhoCPU(nStates), muCPU(nStates), kappaCPU(nStates);
    scalarList CpCPU(nStates), CvCPU(nStates), psiCPU(nStates);

    label s = 0;
    forAll(comps, c)
    {
        forAll(pList, ip)
        {
            forAll(TList, it)
            {
                pF[s] = pList[ip];
                TF[s] = TList[it];
                for (label i = 0; i < n; i++)
                {
                    YF[s*n + i] = comps[c][i];
                }
                cpuReference
                (
                    mix, comps[c], pF[s], TF[s],
                    rhoCPU[s], muCPU[s], kappaCPU[s],
                    CpCPU[s], CvCPU[s], psiCPU[s]
                );
                s++;
            }
        }
    }

    // ── GPU 평가 ─────────────────────────────────────────────────────
    if (rgpGpuDeviceCount() == 0)
    {
        Serr<< "FAIL: no CUDA device visible "
            << "(run inside the container with --nv --bind /usr/lib/wsl)"
            << endl;
        return 1;
    }

    int err = rgpGpuInit(-1);
    if (err)
    {
        Serr<< "FAIL: rgpGpuInit: " << rgpGpuLastError() << endl;
        return 1;
    }

    if (!rgpGpuUploadFromMixture(mix, err))
    {
        Serr<< "FAIL: rgpGpuUpload: " << rgpGpuLastError() << endl;
        return 1;
    }
    Info<< "GPU tables uploaded (" << n << " species)" << nl;

    scalarList rhoGPU(nStates), muGPU(nStates), kappaGPU(nStates);
    scalarList CpGPU(nStates), CvGPU(nStates), psiGPU(nStates);
    err = rgpGpuEvaluate
    (
        nStates, pF.begin(), TF.begin(), YF.begin(),
        rhoGPU.begin(), muGPU.begin(), kappaGPU.begin(),
        CpGPU.begin(), CvGPU.begin(), psiGPU.begin()
    );
    if (err)
    {
        Serr<< "FAIL: rgpGpuEvaluate: " << rgpGpuLastError() << endl;
        return 1;
    }

    // ── 대조 ─────────────────────────────────────────────────────────
    auto relErr = [](scalar a, scalar b) -> scalar
    {
        const scalar denom = max(mag(a), scalar(1e-300));
        return mag(a - b)/denom;
    };

    scalar maxRho = 0, maxMu = 0, maxKappa = 0;
    scalar maxCp = 0, maxCv = 0, maxPsi = 0;
    label iRho = -1, iMu = -1, iKappa = -1, iCp = -1, iCv = -1, iPsi = -1;

    for (label i = 0; i < nStates; i++)
    {
        const scalar er = relErr(rhoCPU[i], rhoGPU[i]);
        const scalar em = relErr(muCPU[i], muGPU[i]);
        const scalar ek = relErr(kappaCPU[i], kappaGPU[i]);
        const scalar ecp = relErr(CpCPU[i], CpGPU[i]);
        const scalar ecv = relErr(CvCPU[i], CvGPU[i]);
        const scalar eps = relErr(psiCPU[i], psiGPU[i]);
        if (er > maxRho) { maxRho = er; iRho = i; }
        if (em > maxMu) { maxMu = em; iMu = i; }
        if (ek > maxKappa) { maxKappa = ek; iKappa = i; }
        if (ecp > maxCp) { maxCp = ecp; iCp = i; }
        if (ecv > maxCv) { maxCv = ecv; iCv = i; }
        if (eps > maxPsi) { maxPsi = eps; iPsi = i; }
    }

    auto report = [&](const char* name, scalar e, label i,
                      const scalarList& c, const scalarList& g)
    {
        Info<< name << ": max relErr = " << e;
        if (i >= 0)
        {
            Info<< "  @ state " << i
                << " (p=" << pF[i] << " T=" << TF[i] << ")"
                << "  CPU=" << c[i] << " GPU=" << g[i];
        }
        Info<< nl;
    };

    Info<< nl << "=== CPU vs GPU comparison ===" << nl;
    report("rho  ", maxRho, iRho, rhoCPU, rhoGPU);
    report("mu   ", maxMu, iMu, muCPU, muGPU);
    report("kappa", maxKappa, iKappa, kappaCPU, kappaGPU);
    report("Cp   ", maxCp, iCp, CpCPU, CpGPU);
    report("Cv   ", maxCv, iCv, CvCPU, CvGPU);
    report("psi  ", maxPsi, iPsi, psiCPU, psiGPU);

    rgpGpuFree();

    // psi는 전방 FD (rho1-rho0)/dp -- rho의 ULP급 차이가 차분 소거로
    // ~1e5배 증폭되므로 해석적 5개 물성과 별도 허용오차(1e4*tol)를 쓴다.
    const scalar psiTol = 1e4*tol;
    const scalar worst =
        max(maxRho, max(maxMu, max(maxKappa, max(maxCp, maxCv))));
    if (worst <= tol && maxPsi <= psiTol)
    {
        Info<< nl << "PASS (analytic worst " << worst << " <= tol " << tol
            << "; psi " << maxPsi << " <= " << psiTol << " [FD])"
            << nl << endl;
        return 0;
    }

    Info<< nl << "FAIL (analytic worst " << worst << " vs tol " << tol
        << "; psi " << maxPsi << " vs " << psiTol << ")"
        << nl << endl;
    return 1;
}


// ************************************************************************* //
