// ROCm/HIP comparison kernels — element-wise a op b → u8 mask

#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>

struct __rocm_bf16 {
    uint16_t __x;
    __device__ __rocm_bf16() : __x(0) {}
    __device__ __rocm_bf16(float f) { union { float f; uint16_t u[2]; } c = {f}; __x = c.u[1]; }
    __device__ operator float() const { union { uint16_t u[2]; float f; } c = {{0, __x}}; return c.f; }
};

#define CMP_OP(TYPENAME, FN_NAME, OP) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    uint8_t *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        out[i] = (lhs[i] OP rhs[i]) ? 1 : 0; \
    } \
}

// flt = float/int comparisons
#define CMP_OPS(TYPENAME, NAME) \
    CMP_OP(TYPENAME, NAME##_eq_##NAME, ==) \
    CMP_OP(TYPENAME, NAME##_ne_##NAME, !=) \
    CMP_OP(TYPENAME, NAME##_lt_##NAME, <)  \
    CMP_OP(TYPENAME, NAME##_le_##NAME, <=) \
    CMP_OP(TYPENAME, NAME##_gt_##NAME, >)  \
    CMP_OP(TYPENAME, NAME##_ge_##NAME, >=)

CMP_OPS(float, f32)
CMP_OPS(double, f64)
CMP_OPS(int64_t, i64)
CMP_OPS(uint8_t, u8)
CMP_OPS(uint32_t, u32)
CMP_OPS(__rocm_bf16, bf16)
