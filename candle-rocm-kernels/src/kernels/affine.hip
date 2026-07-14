#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// bf16 storage type matching cast.hip — __hip_bfloat16 has inconsistent
// operator support across ROCm versions, so we use our own struct.
struct __rocm_bf16 {
    uint16_t __x;

    __device__ __rocm_bf16() : __x(0) {}
    __device__ __rocm_bf16(float f) {
        unsigned int u;
        memcpy(&u, &f, sizeof(u));
        __x = (uint16_t)(u >> 16);
    }
    __device__ operator float() const {
        float f = 0.0f;
        unsigned int u = (unsigned int)__x << 16;
        memcpy(&f, &u, sizeof(f));
        return f;
    }
};

__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    unsigned int tmp_i = idx;
    for (int d = num_dims - 1; d >= 0; d--) {
        unsigned int i_dim = tmp_i % dims[d];
        strided_i += i_dim * strides[d];
        tmp_i /= dims[d];
    }
    return strided_i;
}

#define AFFINE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) { \
    if (info == nullptr || is_contiguous(num_dims, info, info + num_dims)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = inp[i] * mul + add; \
        } \
    } else { \
        const size_t *strides = info + num_dims; \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int src_i = get_strided_index(i, num_dims, info, strides); \
            out[i] = inp[src_i] * mul + add; \
        } \
    } \
}

AFFINE_OP(float, affine_f32)
AFFINE_OP(double, affine_f64)
AFFINE_OP(uint8_t, affine_u8)
AFFINE_OP(uint32_t, affine_u32)
AFFINE_OP(int64_t, affine_i64)

// BF16 uses explicit float conversion via __rocm_bf16's operator float()
// and __rocm_bf16(float) constructor.
extern "C" __global__ void affine_bf16(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const __rocm_bf16 *inp,
    __rocm_bf16 *out,
    const __rocm_bf16 mul,
    const __rocm_bf16 add
) {
    const float mul_f = float(mul);
    const float add_f = float(add);
    if (info == nullptr || is_contiguous(num_dims, info, info + num_dims)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = __rocm_bf16(float(inp[i]) * mul_f + add_f);
        }
    } else {
        const size_t *strides = info + num_dims;
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned int src_i = get_strided_index(i, num_dims, info, strides);
            out[i] = __rocm_bf16(float(inp[src_i]) * mul_f + add_f);
        }
    }
}
