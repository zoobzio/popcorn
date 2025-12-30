#include "common.cuh"

// -----------------------------------------------------------------------------
// Gather Kernel
// out[i] = in[i * stride + idx[i]]
// -----------------------------------------------------------------------------

__global__ void gatherKernel(
    float* out,
    const float* in,
    const int64_t* idx,
    int64_t n,
    int64_t stride
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int64_t index = idx[i];
        out[i] = in[i * stride + index];
    }
}

// -----------------------------------------------------------------------------
// ArgMax / ArgMin Kernels
// One thread per row, iterates over stride dimension
// -----------------------------------------------------------------------------

__global__ void argMaxKernel(
    int64_t* out,
    const float* in,
    int64_t n,
    int64_t stride
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const float* row_start = in + row * stride;
        float max_val = row_start[0];
        int64_t max_idx = 0;

        for (int64_t j = 1; j < stride; j++) {
            float val = row_start[j];
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        out[row] = max_idx;
    }
}

__global__ void argMinKernel(
    int64_t* out,
    const float* in,
    int64_t n,
    int64_t stride
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const float* row_start = in + row * stride;
        float min_val = row_start[0];
        int64_t min_idx = 0;

        for (int64_t j = 1; j < stride; j++) {
            float val = row_start[j];
            if (val < min_val) {
                min_val = val;
                min_idx = j;
            }
        }
        out[row] = min_idx;
    }
}

// -----------------------------------------------------------------------------
// LayerNorm Kernel
// Two-pass: compute mean, then normalize with variance
// One thread per row (batch element)
// -----------------------------------------------------------------------------

__global__ void layerNormKernel(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    int64_t n,
    int64_t norm_size,
    float eps
) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        const float* row_in = in + row * norm_size;
        float* row_out = out + row * norm_size;

        // Pass 1: compute mean
        float sum = 0.0f;
        for (int64_t j = 0; j < norm_size; j++) {
            sum += row_in[j];
        }
        float mean = sum / static_cast<float>(norm_size);

        // Pass 2: compute variance
        float var_sum = 0.0f;
        for (int64_t j = 0; j < norm_size; j++) {
            float diff = row_in[j] - mean;
            var_sum += diff * diff;
        }
        float inv_std = rsqrtf(var_sum / static_cast<float>(norm_size) + eps);

        // Pass 3: normalize and apply affine transform
        for (int64_t j = 0; j < norm_size; j++) {
            float normalized = (row_in[j] - mean) * inv_std;
            if (weight != nullptr) {
                normalized *= weight[j];
            }
            if (bias != nullptr) {
                normalized += bias[j];
            }
            row_out[j] = normalized;
        }
    }
}

// -----------------------------------------------------------------------------
// C API Implementation
// -----------------------------------------------------------------------------

extern "C" {

popcornStatus_t popcornGather_f32(
    float* out,
    const float* in,
    const int64_t* idx,
    int64_t n,
    int64_t stride,
    cudaStream_t stream
) {
    if (auto err = validatePtrs(out, in, idx); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;
    if (stride <= 0) return POPCORN_ERROR_INVALID_VALUE;

    gatherKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, idx, n, stride);
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornArgMax_f32(
    int64_t* out,
    const float* in,
    int64_t n,
    int64_t stride,
    cudaStream_t stream
) {
    if (out == nullptr || in == nullptr) return POPCORN_ERROR_INVALID_VALUE;
    if (n <= 0) return POPCORN_SUCCESS;
    if (stride <= 0) return POPCORN_ERROR_INVALID_VALUE;

    argMaxKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, stride);
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornArgMin_f32(
    int64_t* out,
    const float* in,
    int64_t n,
    int64_t stride,
    cudaStream_t stream
) {
    if (out == nullptr || in == nullptr) return POPCORN_ERROR_INVALID_VALUE;
    if (n <= 0) return POPCORN_SUCCESS;
    if (stride <= 0) return POPCORN_ERROR_INVALID_VALUE;

    argMinKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, stride);
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornLayerNorm_f32(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    int64_t n,
    int64_t norm_size,
    float eps,
    cudaStream_t stream
) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    // weight and bias are nullable, no validation needed
    if (n <= 0) return POPCORN_SUCCESS;
    if (norm_size <= 0) return POPCORN_ERROR_INVALID_VALUE;

    layerNormKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        out, in, weight, bias, n, norm_size, eps
    );
    return checkCuda(cudaGetLastError());
}

} // extern "C"
