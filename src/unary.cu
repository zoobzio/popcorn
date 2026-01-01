#include "common.cuh"

// -----------------------------------------------------------------------------
// Unary Kernel Template
// -----------------------------------------------------------------------------

template <typename Op>
__global__ void unaryKernel(float* out, const float* in, int64_t n, Op op) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = op(in[i]);
    }
}

// -----------------------------------------------------------------------------
// Operation Functors
// -----------------------------------------------------------------------------

struct NegOp {
    __device__ float operator()(float x) const { return -x; }
};

struct AbsOp {
    __device__ float operator()(float x) const { return fabsf(x); }
};

struct ExpOp {
    __device__ float operator()(float x) const { return expf(x); }
};

struct LogOp {
    __device__ float operator()(float x) const { return logf(x); }
};

struct SqrtOp {
    __device__ float operator()(float x) const { return sqrtf(x); }
};

struct SquareOp {
    __device__ float operator()(float x) const { return x * x; }
};

struct SignOp {
    __device__ float operator()(float x) const {
        if (x > 0.0f) return 1.0f;
        if (x < 0.0f) return -1.0f;
        return 0.0f;
    }
};

struct GeluOp {
    __device__ float operator()(float x) const {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
};

struct LeakyReluOp {
    float alpha;
    __device__ explicit LeakyReluOp(float a) : alpha(a) {}
    __device__ float operator()(float x) const {
        return x > 0.0f ? x : alpha * x;
    }
};

struct SinOp {
    __device__ float operator()(float x) const { return sinf(x); }
};

struct CosOp {
    __device__ float operator()(float x) const { return cosf(x); }
};

struct SiluOp {
    __device__ float operator()(float x) const {
        // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        return x / (1.0f + expf(-x));
    }
};

// -----------------------------------------------------------------------------
// C API Implementation
// -----------------------------------------------------------------------------

extern "C" {

const char* popcornGetErrorString(popcornStatus_t status) {
    switch (status) {
        case POPCORN_SUCCESS:
            return "success";
        case POPCORN_ERROR_INVALID_VALUE:
            return "invalid value";
        case POPCORN_ERROR_CUDA:
            return "CUDA error";
        default:
            return "unknown error";
    }
}

cudaError_t popcornGetLastCudaError(void) {
    return __popcorn_last_cuda_error;
}

const char* popcornGetLastCudaErrorString(void) {
    return cudaGetErrorString(__popcorn_last_cuda_error);
}

popcornStatus_t popcornNeg_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, NegOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornAbs_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, AbsOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornExp_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, ExpOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornLog_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, LogOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSqrt_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, SqrtOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSquare_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, SquareOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSign_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, SignOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornGelu_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, GeluOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornLeakyRelu_f32(float* out, const float* in, float alpha, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, LeakyReluOp{alpha});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSin_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, SinOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornCos_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, CosOp{});
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSilu_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, SiluOp{});
    return checkCuda(cudaGetLastError());
}

} // extern "C"
