#ifndef POPCORN_H
#define POPCORN_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    POPCORN_SUCCESS = 0,
    POPCORN_ERROR_INVALID_VALUE = 1,
    POPCORN_ERROR_CUDA = 2,
} popcornStatus_t;

// Get error string for status code
const char* popcornGetErrorString(popcornStatus_t status);

// -----------------------------------------------------------------------------
// Unary Elementwise Operations
// All operations: out[i] = f(in[i]) for i in [0, n)
// -----------------------------------------------------------------------------

popcornStatus_t popcornNeg_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornAbs_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornExp_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLog_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSqrt_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSquare_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSign_f32(float* out, const float* in, int64_t n, cudaStream_t stream);

// Activations not covered by cuDNN
popcornStatus_t popcornGelu_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLeakyRelu_f32(float* out, const float* in, float alpha, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Binary Elementwise Operations
// All operations: out[i] = f(a[i], b[i]) for i in [0, n)
// -----------------------------------------------------------------------------

popcornStatus_t popcornAdd_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSub_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornMul_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornDiv_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornPow_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Binary Scalar Operations
// All operations: out[i] = f(in[i], scalar) for i in [0, n)
// -----------------------------------------------------------------------------

popcornStatus_t popcornAddScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSubScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornMulScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornDivScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornPowScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Comparison / Selection Operations
// -----------------------------------------------------------------------------

popcornStatus_t popcornClamp_f32(float* out, const float* in, float minVal, float maxVal, int64_t n, cudaStream_t stream);
popcornStatus_t popcornWhere_f32(float* out, const float* cond, const float* a, const float* b, int64_t n, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // POPCORN_H
