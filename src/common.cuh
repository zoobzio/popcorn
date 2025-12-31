#ifndef POPCORN_COMMON_CUH
#define POPCORN_COMMON_CUH

#include <cuda_runtime.h>
#include <cmath>
#include "popcorn.h"

// Thread block size for elementwise operations
constexpr int BLOCK_SIZE = 256;

// Calculate grid size for n elements
inline int gridSize(int64_t n) {
    return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// Thread-local storage for last CUDA error (preserves error details)
inline thread_local cudaError_t __popcorn_last_cuda_error = cudaSuccess;

// Check CUDA error and convert to popcorn status
// Stores the actual CUDA error for later retrieval via popcornGetLastCudaError()
inline popcornStatus_t checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        __popcorn_last_cuda_error = err;
        return POPCORN_ERROR_CUDA;
    }
    return POPCORN_SUCCESS;
}

// Validate pointers are non-null
inline popcornStatus_t validatePtrs(const void* a, const void* b) {
    if (a == nullptr || b == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    return POPCORN_SUCCESS;
}

inline popcornStatus_t validatePtrs(const void* a, const void* b, const void* c) {
    if (a == nullptr || b == nullptr || c == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    return POPCORN_SUCCESS;
}

// Mathematical constants
constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
constexpr float GELU_COEF = 0.044715f;

// Debug-mode index bounds checking
// When POPCORN_DEBUG is defined, validates indices before use
// Zero overhead in release builds
#ifdef POPCORN_DEBUG
#define POPCORN_VALIDATE_INDEX(idx, max) \
    if ((idx) < 0 || (idx) >= (max)) return
#else
#define POPCORN_VALIDATE_INDEX(idx, max) ((void)0)
#endif

#endif // POPCORN_COMMON_CUH
