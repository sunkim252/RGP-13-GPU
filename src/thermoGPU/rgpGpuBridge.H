/*---------------------------------------------------------------------------*\
  RGP-13 thermoGPU — Foam 호스트 브리지 (header-only)

  SRKchungTakaMixture의 종-상수/쌍-혼합 테이블을 플랫 배열로 풀어
  rgpGpuUpload()로 디바이스에 올린다. 이 헤더는 g++로 컴파일되는
  번역 단위에서만 include할 것 (Foam 헤더 사용). 디바이스 계층과의
  경계는 rgpKernelTypes.H의 C ABI뿐이다.
\*---------------------------------------------------------------------------*/

#ifndef rgpGpuBridge_H
#define rgpGpuBridge_H

#include "gpu/rgpKernelTypes.H"

#include "scalarList.H"
#include "thermodynamicConstants.H"

namespace Foam
{

//- MixtureType = SRKchungTakaMixture<ThermoType>. 업로드 성공 시 true.
//  플랫 버퍼는 업로드 동안만 유효하면 되므로 로컬 스코프로 충분하다.
template<class MixtureType>
bool rgpGpuUploadFromMixture(const MixtureType& mix, int& errCode)
{
    const label n = mix.specieThermos().size();
    if (n > RGP_GPU_MAX_SPECIES)
    {
        errCode = -1;
        return false;
    }

    // 종-상수 벡터
    scalarList W(n), BM(n), CM(n);
    scalarList jH(7*n), jL(7*n);

    forAll(mix.specieThermos(), i)
    {
        const auto& th = mix.specieThermos()[i];
        W[i]  = th.W();
        BM[i] = mix.gpuBM()[i];
        CM[i] = mix.gpuCM()[i];

        // janafThermo 계수는 생성 시 질량기준(J/kg/K)으로 변환돼 저장됨
        const auto& hc = th.highCpCoeffs();
        const auto& lc = th.lowCpCoeffs();
        for (label c = 0; c < 7; c++)
        {
            jH[7*i + c] = hc[c];
            jL[7*i + c] = lc[c];
        }
    }

    // 쌍 행렬 평탄화 [n*n]
    scalarList C1(n*n), C2(n*n), C3(n*n), S3(n*n), EK0(n*n),
               OM0(n*n), M0(n*n), MI0(n*n), KA(n*n);
    for (label i = 0; i < n; i++)
    {
        for (label j = 0; j < n; j++)
        {
            const label ij = i*n + j;
            C1[ij]  = mix.gpuCOEF1()[i][j];
            C2[ij]  = mix.gpuCOEF2()[i][j];
            C3[ij]  = mix.gpuCOEF3()[i][j];
            S3[ij]  = mix.gpuSIGMA3M()[i][j];
            EK0[ij] = mix.gpuEPSILONKM0()[i][j];
            OM0[ij] = mix.gpuOMEGAM0()[i][j];
            M0[ij]  = mix.gpuMM0()[i][j];
            MI0[ij] = mix.gpuMIUIM0()[i][j];
            KA[ij]  = mix.gpuKAPPAIM()[i][j];
        }
    }

    rgpGpuTables t;
    t.nSpecies   = n;
    t.stableRoot = MixtureType::thermoType::stableRoot() ? 1 : 0;
    t.RR         = constant::thermodynamic::RR;

    // OF 관례: 믹스처 janaf 온도범위는 종-0에서 승계
    t.TlowJ   = mix.specieThermos()[0].Tlow();
    t.ThighJ  = mix.specieThermos()[0].Thigh();
    t.Tcommon = mix.specieThermos()[0].Tcommon();

    t.W = W.begin();  t.BM = BM.begin();  t.CM = CM.begin();
    t.janafHigh = jH.begin();  t.janafLow = jL.begin();
    t.COEF1 = C1.begin();  t.COEF2 = C2.begin();  t.COEF3 = C3.begin();
    t.SIGMA3M = S3.begin();  t.EPSILONKM0 = EK0.begin();
    t.OMEGAM0 = OM0.begin();  t.MM0 = M0.begin();
    t.MIUIM0 = MI0.begin();  t.KAPPAIM = KA.begin();

    errCode = rgpGpuUpload(&t);
    return errCode == 0;
}

} // End namespace Foam

#endif

// ************************************************************************* //
