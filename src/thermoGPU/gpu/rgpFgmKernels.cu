/*---------------------------------------------------------------------------*\
  RGP-13 fgmGPU — updateManifold의 셀 루프 오프로드

  FGMTable::bracket/makeStencil/interpolate와 비트-동일한 산술로,
  셀별 (gZ 대수 클로저 + chi_st Pitsch-Steiner + 4번째 좌표) 계산과
  전체 필드(sourcePV, T, Y_k, RG_*, Le_*)의 16-코너 보간을 수행한다.
  규칙: OpenFOAM 헤더 include 금지.
\*---------------------------------------------------------------------------*/

#include "rgpFgmTypes.H"

#include <cuda_runtime.h>
#include <stdio.h>

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
        int outCap = 0;
    } gB;

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

    const double Zcl = fmax(fmin(Zf[celli], 1.0), 0.0);
    const double Ccl = fmax(Cf[celli], 0.0);
    const double rho_l = fmax(rhof[celli], SMALL);

    const double Zvar = Cv*Lsqrf[celli]*msgf[celli];
    const double gz =
        fmin(fmax(Zvar/fmax(Zcl*(1.0 - Zcl), SMALL), 0.0), 1.0);

    const double D = Defff[celli]/rho_l;
    const double chiTilde = 2.0*D*msgf[celli];
    const double shapeCell = fmax(Zcl*(1.0 - Zcl), SMALL);
    const double chi_st =
        fmax(chiMin, fmin(chiMax, chiTilde*shapeZst/shapeCell));

    double coord4 = chi0;
    if (mode4 == 1) { coord4 = chi_st; }
    else if (mode4 == 2)
    {
        coord4 = hwf[celli] - ((1.0 - Zcl)*hOx + Zcl*hFuel);
    }
    else if (mode4 == 3)
    {
        coord4 = fmax(Wlo, fmin(Whi, hwf[celli]));
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
    const int iKp = (nK >= 2) ? (iK + 1) : iK;

    int idx[16];
    int m = 0;
    for (int dK = 0; dK < 2; dK++)
    {
        const int k = dK ? iKp : iK;
        for (int dC = 0; dC < 2; dC++)
        {
            for (int dG = 0; dG < 2; dG++)
            {
                for (int dZ = 0; dZ < 2; dZ++)
                {
                    idx[m++] =
                        (((iZ + dZ)*nG + (iG + dG))*nC + (iC + dC))*nK + k;
                }
            }
        }
    }

    const size_t nTot = (size_t)nZ*nG*nC*nK;
    for (int f = 0; f < nFields; f++)
    {
        const double* tb = tables + (size_t)f*nTot;

        const double c0000 = tb[idx[0]],  c1000 = tb[idx[1]];
        const double c0100 = tb[idx[2]],  c1100 = tb[idx[3]];
        const double c0010 = tb[idx[4]],  c1010 = tb[idx[5]];
        const double c0110 = tb[idx[6]],  c1110 = tb[idx[7]];
        const double c0001 = tb[idx[8]],  c1001 = tb[idx[9]];
        const double c0101 = tb[idx[10]], c1101 = tb[idx[11]];
        const double c0011 = tb[idx[12]], c1011 = tb[idx[13]];
        const double c0111 = tb[idx[14]], c1111 = tb[idx[15]];

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

    struct { double** d; const double* s; size_t n; } it[] =
    {
        {&gT.Zax, Zax, (size_t)nZ}, {&gT.Gax, Gax, (size_t)nG},
        {&gT.Cax, Cax, (size_t)nC}, {&gT.Kax, Kax, (size_t)nK},
        {&gT.tables, tables, nTot*nFields}
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

    cudaError_t rc;
    if (nCells > gB.cap)
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
    if ((size_t)gB.outCap < need)
    {
        if (gB.out) cudaFree(gB.out);
        if ((rc = cudaMalloc(&gB.out, need*sizeof(double))) != cudaSuccess)
            return ffail(rc, "fgm/out");
        gB.outCap = (int)need;
    }

    const size_t b1 = nCells*sizeof(double);
    struct { double* d; const double* s; } up[] =
    {
        {gB.Z, Z}, {gB.C, C}, {gB.rho, rho}, {gB.Lsqr, Lsqr},
        {gB.msg, magSqrGradZ}, {gB.Deff, DeffZ}
    };
    for (auto& e : up)
    {
        if ((rc = cudaMemcpy(e.d, e.s, b1, cudaMemcpyHostToDevice))
            != cudaSuccess) return ffail(rc, "fgm/H2D");
    }
    if (hOrW)
    {
        if ((rc = cudaMemcpy(gB.hw, hOrW, b1, cudaMemcpyHostToDevice))
            != cudaSuccess) return ffail(rc, "fgm/H2D hw");
    }

    constexpr int bs = 128;
    rgpfgm::fgmKernel<<<(nCells + bs - 1)/bs, bs>>>
    (
        nCells, mode4, Cv, shapeZst, chiMin, chiMax, srcScale, chi0,
        hOx, hFuel, Wlo, Whi,
        gT.nZ, gT.nG, gT.nC, gT.nK, gT.Zax, gT.Gax, gT.Cax, gT.Kax,
        gT.nFields, gT.tables,
        gB.Z, gB.C, gB.rho, gB.Lsqr, gB.msg, gB.Deff, gB.hw,
        gB.gZ, gB.chi, gB.out
    );
    if ((rc = cudaGetLastError()) != cudaSuccess)
        return ffail(rc, "fgm/launch");

    if ((rc = cudaMemcpy(gZ, gB.gZ, b1, cudaMemcpyDeviceToHost))
        != cudaSuccess) return ffail(rc, "fgm/D2H gZ");
    if ((rc = cudaMemcpy(chiSt, gB.chi, b1, cudaMemcpyDeviceToHost))
        != cudaSuccess) return ffail(rc, "fgm/D2H chi");
    if ((rc = cudaMemcpy(fieldsOut, gB.out, need*sizeof(double),
                         cudaMemcpyDeviceToHost)) != cudaSuccess)
        return ffail(rc, "fgm/D2H out");

    return 0;
}


void rgpFgmFree(void) { fgmFreeAll(); }

const char* rgpFgmLastError(void) { return gFgmErr; }

} // extern "C"

// ************************************************************************* //
