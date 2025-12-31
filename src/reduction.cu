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
        POPCORN_VALIDATE_INDEX(index, stride);
        out[i] = in[i * stride + index];
    }
}

// -----------------------------------------------------------------------------
// ArgMax / ArgMin Kernels
// Sequential version: one thread per row, iterates over stride dimension
// Parallel version: one block per row, uses shared memory reduction
// -----------------------------------------------------------------------------

// Sequential kernel for small strides (stride <= BLOCK_SIZE)
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

// Parallel reduction kernel for large strides
// One block per row, threads cooperate to find max/min
template <int THREADS>
__global__ void argMaxParallelKernel(
    int64_t* out,
    const float* in,
    int64_t n,
    int64_t stride
) {
    __shared__ float s_vals[THREADS];
    __shared__ int64_t s_idxs[THREADS];

    int64_t row = blockIdx.x;
    if (row >= n) return;

    const float* row_data = in + row * stride;

    // Phase 1: Each thread finds local max across its assigned elements
    float local_max = -INFINITY;
    int64_t local_idx = 0;

    for (int64_t j = threadIdx.x; j < stride; j += THREADS) {
        float val = row_data[j];
        if (val > local_max) {
            local_max = val;
            local_idx = j;
        }
    }

    s_vals[threadIdx.x] = local_max;
    s_idxs[threadIdx.x] = local_idx;
    __syncthreads();

    // Phase 2: Block reduction
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_vals[threadIdx.x + s] > s_vals[threadIdx.x]) {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + s];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (threadIdx.x == 0) {
        out[row] = s_idxs[0];
    }
}

template <int THREADS>
__global__ void argMinParallelKernel(
    int64_t* out,
    const float* in,
    int64_t n,
    int64_t stride
) {
    __shared__ float s_vals[THREADS];
    __shared__ int64_t s_idxs[THREADS];

    int64_t row = blockIdx.x;
    if (row >= n) return;

    const float* row_data = in + row * stride;

    // Phase 1: Each thread finds local min across its assigned elements
    float local_min = INFINITY;
    int64_t local_idx = 0;

    for (int64_t j = threadIdx.x; j < stride; j += THREADS) {
        float val = row_data[j];
        if (val < local_min) {
            local_min = val;
            local_idx = j;
        }
    }

    s_vals[threadIdx.x] = local_min;
    s_idxs[threadIdx.x] = local_idx;
    __syncthreads();

    // Phase 2: Block reduction
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_vals[threadIdx.x + s] < s_vals[threadIdx.x]) {
                s_vals[threadIdx.x] = s_vals[threadIdx.x + s];
                s_idxs[threadIdx.x] = s_idxs[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (threadIdx.x == 0) {
        out[row] = s_idxs[0];
    }
}

// Threshold for switching to parallel reduction
constexpr int64_t PARALLEL_REDUCTION_THRESHOLD = 256;

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
// LayerNorm Kernel with Statistics Output
// Same as layerNormKernel but optionally outputs mean and invstd for backward
// -----------------------------------------------------------------------------

__global__ void layerNormWithStatsKernel(
    float* out,
    float* out_mean,
    float* out_invstd,
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

        // Save statistics for backward pass
        if (out_mean != nullptr) {
            out_mean[row] = mean;
        }
        if (out_invstd != nullptr) {
            out_invstd[row] = inv_std;
        }

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

    if (stride <= PARALLEL_REDUCTION_THRESHOLD) {
        // Sequential: one thread per row
        argMaxKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, stride);
    } else {
        // Parallel: one block per row with shared memory reduction
        argMaxParallelKernel<BLOCK_SIZE><<<n, BLOCK_SIZE, 0, stream>>>(out, in, n, stride);
    }
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

    if (stride <= PARALLEL_REDUCTION_THRESHOLD) {
        // Sequential: one thread per row
        argMinKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, stride);
    } else {
        // Parallel: one block per row with shared memory reduction
        argMinParallelKernel<BLOCK_SIZE><<<n, BLOCK_SIZE, 0, stream>>>(out, in, n, stride);
    }
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

popcornStatus_t popcornLayerNormWithStats_f32(
    float* out,
    float* out_mean,
    float* out_invstd,
    const float* in,
    const float* weight,
    const float* bias,
    int64_t n,
    int64_t norm_size,
    float eps,
    cudaStream_t stream
) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    // weight, bias, out_mean, out_invstd are nullable
    if (n <= 0) return POPCORN_SUCCESS;
    if (norm_size <= 0) return POPCORN_ERROR_INVALID_VALUE;

    layerNormWithStatsKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        out, out_mean, out_invstd, in, weight, bias, n, norm_size, eps
    );
    return checkCuda(cudaGetLastError());
}

} // extern "C"
