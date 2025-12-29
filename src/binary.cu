#include "common.cuh"

// -----------------------------------------------------------------------------
// Binary Kernel Templates
// -----------------------------------------------------------------------------

template <typename Op>
__global__ void binaryKernel(float* out, const float* a, const float* b, int64_t n, Op op) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = op(a[i], b[i]);
    }
}

template <typename Op>
__global__ void binaryScalarKernel(float* out, const float* in, float scalar, int64_t n, Op op) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = op(in[i], scalar);
    }
}

// -----------------------------------------------------------------------------
// Operation Functors
// -----------------------------------------------------------------------------

struct AddOp {
    __device__ float operator()(float a, float b) const { return a + b; }
};

struct SubOp {
    __device__ float operator()(float a, float b) const { return a - b; }
};

struct MulOp {
    __device__ float operator()(float a, float b) const { return a * b; }
};

struct DivOp {
    __device__ float operator()(float a, float b) const { return a / b; }
};

struct PowOp {
    __device__ float operator()(float a, float b) const { return powf(a, b); }
};

// -----------------------------------------------------------------------------
// Comparison / Selection Kernels
// -----------------------------------------------------------------------------

__global__ void clampKernel(float* out, const float* in, float minVal, float maxVal, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = fminf(fmaxf(x, minVal), maxVal);
    }
}

__global__ void whereKernel(float* out, const float* cond, const float* a, const float* b, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // cond > 0 selects a, otherwise b
        out[i] = (cond[i] > 0.0f) ? a[i] : b[i];
    }
}

// -----------------------------------------------------------------------------
// C API Implementation - Binary Operations
// -----------------------------------------------------------------------------

extern "C" {

popcornStatus_t popcornAdd_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, a, b); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, a, b, n, AddOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSub_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, a, b); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, a, b, n, SubOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornMul_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, a, b); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, a, b, n, MulOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornDiv_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, a, b); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, a, b, n, DivOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornPow_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, a, b); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, a, b, n, PowOp{});
    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// C API Implementation - Scalar Operations
// -----------------------------------------------------------------------------

popcornStatus_t popcornAddScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryScalarKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, scalar, n, AddOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSubScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryScalarKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, scalar, n, SubOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornMulScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryScalarKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, scalar, n, MulOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornDivScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryScalarKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, scalar, n, DivOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornPowScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    binaryScalarKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, scalar, n, PowOp{});
    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// C API Implementation - Comparison / Selection
// -----------------------------------------------------------------------------

popcornStatus_t popcornClamp_f32(float* out, const float* in, float minVal, float maxVal, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    clampKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, minVal, maxVal, n);
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornWhere_f32(float* out, const float* cond, const float* a, const float* b, int64_t n, cudaStream_t stream) {
    if (out == nullptr || cond == nullptr || a == nullptr || b == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    whereKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, cond, a, b, n);
    return checkCuda(cudaGetLastError());
}

} // extern "C"
