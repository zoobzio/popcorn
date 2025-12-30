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

// -----------------------------------------------------------------------------
// Gather / Index Operations
// -----------------------------------------------------------------------------

// Gathers values from input at indices specified by idx
// out[i] = in[i * stride + idx[i]]
// Used for selecting class probabilities at target indices (NLLLoss/CrossEntropyLoss)
popcornStatus_t popcornGather_f32(
    float* out,           // [n] output values
    const float* in,      // [n, classes] input tensor (row-major)
    const int64_t* idx,   // [n] indices (0 to stride-1)
    int64_t n,            // batch size
    int64_t stride,       // number of classes (inner dimension)
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Reduction Operations
// -----------------------------------------------------------------------------

// Returns indices of max values along last dimension
// out[i] = argmax(in[i*stride : i*stride+stride])
popcornStatus_t popcornArgMax_f32(
    int64_t* out,         // [n] output indices
    const float* in,      // [n, dim] input tensor (row-major)
    int64_t n,            // number of rows
    int64_t stride,       // size of dimension to reduce
    cudaStream_t stream
);

// Returns indices of min values along last dimension
// out[i] = argmin(in[i*stride : i*stride+stride])
popcornStatus_t popcornArgMin_f32(
    int64_t* out,         // [n] output indices
    const float* in,      // [n, dim] input tensor (row-major)
    int64_t n,            // number of rows
    int64_t stride,       // size of dimension to reduce
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Normalization Operations
// -----------------------------------------------------------------------------

// Applies layer normalization: out = (in - mean) / sqrt(var + eps) * weight + bias
// Normalizes over last `norm_size` elements
popcornStatus_t popcornLayerNorm_f32(
    float* out,           // [n, norm_size] output
    const float* in,      // [n, norm_size] input
    const float* weight,  // [norm_size] scale (gamma), nullable for no scaling
    const float* bias,    // [norm_size] shift (beta), nullable for no shift
    int64_t n,            // batch size (product of all dims except normalized)
    int64_t norm_size,    // size of normalized dimension(s)
    float eps,            // epsilon for numerical stability (typically 1e-5)
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // POPCORN_H
