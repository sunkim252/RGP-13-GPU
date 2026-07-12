/*---------------------------------------------------------------------------*\
  RGP-13 fgmGPU — updateManifold의 셀 루프 오프로드

  FGMTable::bracket/makeStencil/interpolate와 비트-동일한 산술로,
  셀별 (gZ 대수 클로저 + chi_st Pitsch-Steiner + 4번째 좌표) 계산과
  전체 필드(sourcePV, T, Y_k, RG_*, Le_*)의 16-코너 보간을 수행한다.
  규칙: OpenFOAM 헤더 include 금지.
\*---------------------------------------------------------------------------*/

#include "rgpFgmTypes.H"
#include "rgpStage.H"

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

namespace
{
    char gFgmErr[256] = "no error";

    int ffail(cudaError_t e, const char* where)
    {
        snprintf(gFgmErr, sizeof(gFgmErr), "%s: %s",
                 where, cudaGetErrorString(e));
        return int(e);
    }

    struct FgmDev
    {
        int nZ = 0, nG = 0, nC = 0, nK = 0, nFields = 0;
        double *Zax = nullptr, *Gax = nullptr, *Cax = nullptr,
               *Kax = nullptr, *tables = nullptr;
    } gT;

    struct FgmBuf
    {
        int cap = 0;
        double *Z = nullptr, *C = nullptr, *rho = nullptr, *Lsqr = nullptr,
               *msg = nullptr, *Deff = nullptr, *hw = nullptr,
               *gZ = nullptr, *chi = nullptr, *out = nullptr;
        size_t outCap = 0;
        int lastN = 0;   // 마지막 evaluate의 nCells (디바이스 체인 검증용)
        double *outActive = nullptr;   // 체인이 읽을 실제 출력 포인터
    } gB;

    int gSkipFirst = 0;
    int gSkipN = 0;

    void fgmFreeAll()
    {
        double** ps[] = {&gT.Zax,&gT.Gax,&gT.Cax,&gT.Kax,&gT.tables,
                         &gB.Z,&gB.C,&gB.rho,&gB.Lsqr,&gB.msg,&gB.Deff,
                         &gB.hw,&gB.gZ,&gB.chi,&gB.out};
        for (auto p : ps) { if (*p) { cudaFree(*p); *p = nullptr; } }
        gT = FgmDev(); gB = FgmBuf();
    }
}

namespace rgpfgm
{

constexpr double SMALL = 1e-15, VSMALL = 1e-300;

//- Foam::max/min 의미론 (NaN 전파 — fmax/fmin은 NaN을 버려 상류
//  필드 오염을 가려버림; 비-NaN 값에서는 결과 동일)
__device__ inline double fMax(double a, double b) { return (a > b) ? a : b; }
__device__ inline double fMin(double a, double b) { return (a < b) ? a : b; }

//- FGMTable::bracket 1:1
__device__ void bracket
(
    const double* axis, const int n, double v, int& i, double& w
)
{
    if (n <= 1) { i = 0; w = 0; return; }
    if (v <= axis[0]) { i = 0; w = 0; return; }
    if (v >= axis[n - 1]) { i = n - 2; w = 1; return; }

    const double span = axis[n - 1] - axis[0];
    int j = 1;
    if (span > VSMALL)
    {
        j = int((v - axis[0])/span*(n - 1)) + 1;
        j = max(1, min(j, n - 1));
    }
    while (j < n - 1 && axis[j] < v) { j++; }
    while (j > 1 && axis[j - 1] >= v) { j--; }

    i = j - 1;
    const double d = axis[j] - axis[i];
    w = (d > VSMALL) ? (v - axis[i])/d : 0.0;
}


__global__ void fgmKernel
(
    const int nCells, const int mode4,
    const double Cv, const double shapeZst,
    const double chiMin, const double chiMax,
    const double srcScale, const double chi0,
    const double hOx, const double hFuel,
    const double Wlo, const double Whi,
    const int nZ, const int nG, const int nC, const int nK,
    const double* __restrict__ Zax, const double* __restrict__ Gax,
    const double* __restrict__ Cax, const double* __restrict__ Kax,
    const int nFields, const double* __restrict__ tables,
    const double* __restrict__ Zf, const double* __restrict__ Cf,
    const double* __restrict__ rhof, const double* __restrict__ Lsqrf,
    const double* __restrict__ msgf, const double* __restrict__ Defff,
    const double* __restrict__ hwf,
    double* __restrict__ gZf, double* __restrict__ chif,
    double* __restrict__ outf
)
{
    const int celli = blockIdx.x*blockDim.x + threadIdx.x;
    if (celli >= nCells) return;

    const double Zcl = fMax(fMin(Zf[celli], 1.0), 0.0);
    const double Ccl = fMax(Cf[celli], 0.0);
    const double rho_l = fMax(rhof[celli], SMALL);

    const double Zvar = Cv*Lsqrf[celli]*msgf[celli];
    const double gz =
        fMin(fMax(Zvar/fMax(Zcl*(1.0 - Zcl), SMALL), 0.0), 1.0);

    const double D = Defff[celli]/rho_l;
    const double chiTilde = 2.0*D*msgf[celli];
    const double shapeCell = fMax(Zcl*(1.0 - Zcl), SMALL);
    const double chi_st =
        fMax(chiMin, fMin(chiMax, chiTilde*shapeZst/shapeCell));

    double coord4 = chi0;
    if (mode4 == 1) { coord4 = chi_st; }
    else if (mode4 == 2)
    {
        coord4 = hwf[celli] - ((1.0 - Zcl)*hOx + Zcl*hFuel);
    }
    else if (mode4 == 3)
    {
        coord4 = fMax(Wlo, fMin(Whi, hwf[celli]));
    }

    gZf[celli] = gz;
    chif[celli] = chi_st;

    // 스텐실 (FGMTable::makeStencil의 코너 순서와 동일)
    int iZ, iG, iC, iK;
    double wZ, wG, wC, wK;
    bracket(Zax, nZ, Zcl, iZ, wZ);
    bracket(Gax, nG, gz, iG, wG);
    bracket(Cax, nC, Ccl, iC, wC);
    bracket(Kax, nK, coord4, iK, wK);
    // 퇴화 축(n==1) 폴딩: 상단 코너를 자기 자신으로 접어 OOB 차단
    // (w=0이라 값 기여는 없지만, 미폴딩 시 이웃 슬라이스/할당 밖을 읽음)
    const int iZp = (nZ >= 2) ? (iZ + 1) : iZ;
    const int iGp = (nG >= 2) ? (iG + 1) : iG;
    const int iCp = (nC >= 2) ? (iC + 1) : iC;
    const int iKp = (nK >= 2) ? (iK + 1) : iK;

    size_t idx[16];
    int m = 0;
    for (int dK = 0; dK < 2; dK++)
    {
        const int k = dK ? iKp : iK;
        for (int dC = 0; dC < 2; dC++)
        {
            const int c = dC ? iCp : iC;
            for (int dG = 0; dG < 2; dG++)
            {
                const int g = dG ? iGp : iG;
                for (int dZ = 0; dZ < 2; dZ++)
                {
                    const int z = dZ ? iZp : iZ;
                    idx[m++] =
                        (((size_t)z*nG + g)*nC + c)*nK + k;
                }
            }
        }
    }

    // 테이블은 노드-우선 [node*nFields + f] (업로드 시 전치) — 코너당
    // 전 필드가 연속이라 필드 루프가 16개의 국소-연속 스트림을 읽는다
    // (필드-우선 산발 8B 게더의 캐시라인 낭비 제거; 값·산술 순서 동일
    // → 비트-동일)
    size_t base[16];
    for (int m2 = 0; m2 < 16; m2++)
    {
        base[m2] = idx[m2]*(size_t)nFields;
    }
    for (int f = 0; f < nFields; f++)
    {
        const double c0000 = tables[base[0]  + f];
        const double c1000 = tables[base[1]  + f];
        const double c0100 = tables[base[2]  + f];
        const double c1100 = tables[base[3]  + f];
        const double c0010 = tables[base[4]  + f];
        const double c1010 = tables[base[5]  + f];
        const double c0110 = tables[base[6]  + f];
        const double c1110 = tables[base[7]  + f];
        const double c0001 = tables[base[8]  + f];
        const double c1001 = tables[base[9]  + f];
        const double c0101 = tables[base[10] + f];
        const double c1101 = tables[base[11] + f];
        const double c0011 = tables[base[12] + f];
        const double c1011 = tables[base[13] + f];
        const double c0111 = tables[base[14] + f];
        const double c1111 = tables[base[15] + f];

        const double a00 = c0000*(1 - wZ) + c1000*wZ;
        const double a10 = c0100*(1 - wZ) + c1100*wZ;
        const double a01 = c0010*(1 - wZ) + c1010*wZ;
        const double a11 = c0110*(1 - wZ) + c1110*wZ;
        const double a0  = a00*(1 - wG) + a10*wG;
        const double a1  = a01*(1 - wG) + a11*wG;
        const double A   = a0*(1 - wC) + a1*wC;

        const double b00 = c0001*(1 - wZ) + c1001*wZ;
        const double b10 = c0101*(1 - wZ) + c1101*wZ;
        const double b01 = c0011*(1 - wZ) + c1011*wZ;
        const double b11 = c0111*(1 - wZ) + c1111*wZ;
        const double b0  = b00*(1 - wG) + b10*wG;
        const double b1  = b01*(1 - wG) + b11*wG;
        const double B   = b0*(1 - wC) + b1*wC;

        double v = A*(1 - wK) + B*wK;
        if (f == 0) { v *= srcScale*rho_l; }     // sourcePV [1/s] → 체적량

        outf[(size_t)f*nCells + celli] = v;
    }
}

} // namespace rgpfgm


// * * * * * * * * * * * * * * * C ABI * * * * * * * * * * * * * * * * * * * //

extern "C"
{

int rgpFgmUpload
(
    int nZ, int nG, int nC, int nK,
    const double* Zax, const double* Gax,
    const double* Cax, const double* Kax,
    int nFields,
    const double* tables
)
{
    fgmFreeAll();
    gT.nZ = nZ; gT.nG = nG; gT.nC = nC; gT.nK = nK; gT.nFields = nFields;
    const size_t nTot = (size_t)nZ*nG*nC*nK;

    // 노드-우선 전치 [node*nFields + f] — 커널의 16-코너 게더가
    // 코너당 전 필드를 연속으로 읽도록 (호스트 1회, 값 불변)
    std::vector<double> tN(nTot*nFields);
    for (size_t f = 0; f < (size_t)nFields; f++)
    {
        const double* src = tables + f*nTot;
        for (size_t n = 0; n < nTot; n++)
        {
            tN[n*nFields + f] = src[n];
        }
    }

    struct { double** d; const double* s; size_t n; } it[] =
    {
        {&gT.Zax, Zax, (size_t)nZ}, {&gT.Gax, Gax, (size_t)nG},
        {&gT.Cax, Cax, (size_t)nC}, {&gT.Kax, Kax, (size_t)nK},
        {&gT.tables, tN.data(), nTot*nFields}
    };
    for (auto& e : it)
    {
        cudaError_t rc = cudaMalloc(e.d, e.n*sizeof(double));
        if (rc != cudaSuccess) { fgmFreeAll(); return ffail(rc, "fgm/malloc"); }
        rc = cudaMemcpy(*e.d, e.s, e.n*sizeof(double),
                        cudaMemcpyHostToDevice);
        if (rc != cudaSuccess) { fgmFreeAll(); return ffail(rc, "fgm/memcpy"); }
    }
    return 0;
}


int rgpFgmEvaluate
(
    int nCells, int mode4,
    double Cv, double shapeZst, double chiMin, double chiMax,
    double srcScale, double chi0, double hOx, double hFuel,
    double Wlo, double Whi,
    const double* Z, const double* C, const double* rho,
    const double* Lsqr, const double* magSqrGradZ, const double* DeffZ,
    const double* hOrW,
    double* gZ, double* chiSt,
    double* fieldsOut
)
{
    if (nCells <= 0) return 0;
    if (!gT.tables)
    {
        snprintf(gFgmErr, sizeof(gFgmErr), "fgm tables not uploaded");
        return -1;
    }

    // 부분 실패 시 이전 스텝의 SoA를 체인이 소비하지 않도록 선-무효화
    gB.lastN = 0;
    gB.outActive = nullptr;

    cudaError_t rc;
    // native(2)는 입력·출력 모두 호스트 포인터 직행(zero-copy) —
    // 스테이징 버퍼가 전혀 쓰이지 않으므로 할당 자체를 생략
    // ((9+nFields)*nCells*8B, 2.6M셀 ~2.7GB 절약). copy/mapped 폴백만 할당.
    if (gRgpUnified != 2 && nCells > gB.cap)
    {
        double** ps[] = {&gB.Z,&gB.C,&gB.rho,&gB.Lsqr,&gB.msg,&gB.Deff,
                         &gB.hw,&gB.gZ,&gB.chi};
        for (auto p : ps) { if (*p) cudaFree(*p); *p = nullptr; }
        for (auto p : ps)
        {
            if ((rc = cudaMalloc(p, nCells*sizeof(double))) != cudaSuccess)
                return ffail(rc, "fgm/buf");
        }
        gB.cap = nCells;
    }
    const size_t need = (size_t)gT.nFields*nCells;
    if (gRgpUnified != 2 && gB.outCap < need)
    {
        if (gB.out) cudaFree(gB.out);
        if ((rc = cudaMalloc(&gB.out, need*sizeof(double))) != cudaSuccess)
            return ffail(rc, "fgm/out");
        gB.outCap = need;
    }

    const size_t n1 = (size_t)nCells;
    const double* dZ = rgpInPtr(Z, gB.Z, n1, &rc);
    if (rc != cudaSuccess) return ffail(rc, "fgm/in Z");
    const double* dC = rgpInPtr(C, gB.C, n1, &rc);
    if (rc != cudaSuccess) return ffail(rc, "fgm/in C");
    const double* dRho = rgpInPtr(rho, gB.rho, n1, &rc);
    if (rc != cudaSuccess) return ffail(rc, "fgm/in rho");
    const double* dLsqr = rgpInPtr(Lsqr, gB.Lsqr, n1, &rc);
    if (rc != cudaSuccess) return ffail(rc, "fgm/in Lsqr");
    const double* dMsg = rgpInPtr(magSqrGradZ, gB.msg, n1, &rc);
    if (rc != cudaSuccess) return ffail(rc, "fgm/in msg");
    const double* dDeff = rgpInPtr(DeffZ, gB.Deff, n1, &rc);
    if (rc != cudaSuccess) return ffail(rc, "fgm/in Deff");
    const double* dHw = gB.hw;
    if (hOrW)
    {
        dHw = rgpInPtr(hOrW, gB.hw, n1, &rc);
        if (rc != cudaSuccess) return ffail(rc, "fgm/in hw");
    }

    double* oGZ = rgpOutPtr(gZ, gB.gZ);
    double* oChi = rgpOutPtr(chiSt, gB.chi);
    double* oOut = rgpOutPtr(fieldsOut, gB.out);

    constexpr int bs = 128;
    rgpfgm::fgmKernel<<<(nCells + bs - 1)/bs, bs>>>
    (
        nCells, mode4, Cv, shapeZst, chiMin, chiMax, srcScale, chi0,
        hOx, hFuel, Wlo, Whi,
        gT.nZ, gT.nG, gT.nC, gT.nK, gT.Zax, gT.Gax, gT.Cax, gT.Kax,
        gT.nFields, gT.tables,
        dZ, dC, dRho, dLsqr, dMsg, dDeff, dHw,
        oGZ, oChi, oOut
    );
    if ((rc = cudaGetLastError()) != cudaSuccess)
        return ffail(rc, "fgm/launch");

    if ((rc = rgpOutFinish(gZ, oGZ, gB.gZ, n1)) != cudaSuccess)
        return ffail(rc, "fgm/out gZ");
    if ((rc = rgpOutFinish(chiSt, oChi, gB.chi, n1)) != cudaSuccess)
        return ffail(rc, "fgm/out chi");
    // fieldsOut 회수: copy 경유(oOut==gB.out)일 때만 D2H — skip 범위
    // [gSkipFirst, gSkipFirst+gSkipN)는 내리지 않는다 (호스트 미소비
    // 필드의 대역폭 절약; 디바이스 상주 SoA는 완전하므로 체인 무영향).
    if (oOut == gB.out && fieldsOut != nullptr)
    {
        const int nF = gT.nFields;
        int s0 = gSkipFirst < nF ? gSkipFirst : nF;
        int s1 = gSkipFirst + gSkipN < nF ? gSkipFirst + gSkipN : nF;
        if (gSkipN <= 0) { s0 = nF; s1 = nF; }
        const size_t cb = (size_t)nCells*sizeof(double);
        if (s0 > 0)
        {
            if ((rc = cudaMemcpy(fieldsOut, gB.out,
                                 (size_t)s0*cb, cudaMemcpyDeviceToHost))
                != cudaSuccess) return ffail(rc, "fgm/out fields lo");
        }
        if (s1 < nF)
        {
            if ((rc = cudaMemcpy(fieldsOut + (size_t)s1*nCells,
                                 gB.out + (size_t)s1*nCells,
                                 (size_t)(nF - s1)*cb,
                                 cudaMemcpyDeviceToHost))
                != cudaSuccess) return ffail(rc, "fgm/out fields hi");
        }
    }
    else
    {
        if ((rc = cudaDeviceSynchronize()) != cudaSuccess)
            return ffail(rc, "fgm/out fields sync");
    }

    gB.lastN = nCells;
    gB.outActive = oOut;   // 체인(he 재시드/refresh)이 읽을 위치
    gSkipFirst = 0; gSkipN = 0;   // one-shot: 다음 호출자에 안 새게
    return 0;
}


/*  디바이스 체인용 접근자: 마지막 rgpFgmEvaluate의 SoA 출력(디바이스
    포인터)과 그 레이아웃. rgpGpuEvaluate*FromFgm(rgpKernels.cu)이 같은
    .so 안에서 호출해 (T, Y_k)를 호스트 왕복 없이 재사용한다.          */
const double* rgpFgmDevOutPtr(void)
{
    return gB.outActive ? gB.outActive : gB.out;
}
int rgpFgmDevNFields(void) { return gT.nFields; }
void rgpFgmHostCopySkip(int first, int n)
{
    gSkipFirst = first > 0 ? first : 0;
    gSkipN = n > 0 ? n : 0;
}

int rgpFgmDevLastN(void) { return gB.lastN; }

void rgpFgmFree(void) { fgmFreeAll(); }

const char* rgpFgmLastError(void) { return gFgmErr; }

} // extern "C"

// ************************************************************************* //
