/*---------------------------------------------------------------------------*\
  RGP-13 GPU — unified/copy 스테이징 선택 (nvcc TU 전용)

  모드 (rgpGpuInit이 기기 감지로 설정, env RGP_GPU_UNIFIED로 override):
    0 = copy    : 기존 명시적 cudaMemcpy 스테이징 (디스크리트 GPU + PCIe)
    1 = mapped  : cudaHostRegister(Mapped)된 버퍼만 디바이스 별칭으로
                  zero-copy, 미등록 포인터는 copy 폴백 — 디스크리트에서
                  unified 경로의 '정합' 검증용 (PCIe라 느릴 수 있음)
    2 = native  : GH200류 일관 메모리(HMM/ATS) — 호스트 포인터 그대로
                  커널에 전달 (cudaDevAttrPageableMemoryAccess)

  사용 규약:
    in  = rgpInPtr(h, devBuf, n, &e)  → 커널 입력 포인터 (e 확인)
    out = rgpOutPtr(h, devBuf)        → 커널이 직접 쓰는 출력 포인터
          커널 후 out == devBuf 일 때만 D2H copy 필요.
  내부 스크래치(반복 솔버 벡터, 누적 버퍼)는 항상 cudaMalloc 유지.
\*---------------------------------------------------------------------------*/

#ifndef rgpStage_H
#define rgpStage_H

#include <cuda_runtime.h>

extern int gRgpUnified;   // rgpKernels.cu에서 정의

static inline const double* rgpInPtr
(
    const double* h, double* devBuf, size_t n, cudaError_t* err
)
{
    *err = cudaSuccess;
    if (gRgpUnified == 2)
    {
        return h;
    }
    if (gRgpUnified == 1)
    {
        cudaPointerAttributes a;
        if
        (
            cudaPointerGetAttributes(&a, h) == cudaSuccess
         && a.devicePointer != nullptr
        )
        {
            return static_cast<const double*>(a.devicePointer);
        }
        cudaGetLastError();   // 미등록 포인터 오류 클리어 → copy 폴백
    }
    *err = cudaMemcpy(devBuf, h, n*sizeof(double), cudaMemcpyHostToDevice);
    return devBuf;
}

static inline double* rgpOutPtr(double* h, double* devBuf)
{
    if (gRgpUnified == 2)
    {
        return h;
    }
    if (gRgpUnified == 1)
    {
        cudaPointerAttributes a;
        if
        (
            cudaPointerGetAttributes(&a, h) == cudaSuccess
         && a.devicePointer != nullptr
        )
        {
            return static_cast<double*>(a.devicePointer);
        }
        cudaGetLastError();
    }
    return devBuf;
}

//- out == devBuf(스테이징 경유)일 때만 D2H
static inline cudaError_t rgpOutFinish
(
    double* h, double* outPtr, double* devBuf, size_t n
)
{
    if (outPtr == devBuf && h != nullptr)
    {
        return cudaMemcpy(h, devBuf, n*sizeof(double),
                          cudaMemcpyDeviceToHost);
    }
    // zero-copy 경로: 커널 쓰기가 호스트에서 보이도록 동기화
    return cudaDeviceSynchronize();
}

#endif

/* ************************************************************************* */
