#include "common.cuh"

// -----------------------------------------------------------------------------
// Quantized MatMul Kernels
// Fused dequantize + matrix multiply for weight-only quantized inference
// -----------------------------------------------------------------------------

// Tile sizes for tiled matrix multiply
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 32;

// -----------------------------------------------------------------------------
// INT8 Per-Channel Dequantize + MatMul
// out[m,n] = sum_k(x[m,k] * qweight[n,k] * scale[n])
// -----------------------------------------------------------------------------

__global__ void dequantizeMatmulKernel_i8f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K
) {
    __shared__ float s_x[TILE_M][TILE_K];
    __shared__ int8_t s_w[TILE_N][TILE_K];
    __shared__ float s_scale[TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc = 0.0f;

    // Load scales for this tile's output columns into shared memory
    if (ty == 0 && col < N) {
        s_scale[tx] = scale[col];
    }

    // Iterate over K dimension in tiles
    for (int64_t k_base = 0; k_base < K; k_base += TILE_K) {
        // Load x tile into shared memory
        int64_t k_idx = k_base + tx;
        if (row < M && k_idx < K) {
            s_x[ty][tx] = x[row * K + k_idx];
        } else {
            s_x[ty][tx] = 0.0f;
        }

        // Load weight tile into shared memory
        if (col < N && k_idx < K) {
            s_w[tx][ty] = qweight[col * K + k_base + ty];
        } else {
            s_w[tx][ty] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        if (row < M && col < N) {
            float sc = s_scale[tx];
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                if (k_base + k < K) {
                    float w_fp = static_cast<float>(s_w[tx][k]) * sc;
                    acc += s_x[ty][k] * w_fp;
                }
            }
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

extern "C" popcornStatus_t popcornDequantizeMatmul_i8f32(
    float* out,
    const float* x,
    const int8_t* qweight,
    const float* scale,
    int64_t M,
    int64_t N,
    int64_t K,
    cudaStream_t stream
) {
    if (out == nullptr || x == nullptr || qweight == nullptr || scale == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    dequantizeMatmulKernel_i8f32<<<grid, block, 0, stream>>>(
        out, x, qweight, scale, M, N, K
    );

    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// INT8 Per-Group Dequantize + MatMul
// out[m,n] = sum_k(x[m,k] * qweight[n,k] * scale[n, k/group_size])
// -----------------------------------------------------------------------------

__global__ void dequantizeMatmulGroupedKernel_i8f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size,
    int64_t num_groups
) {
    __shared__ float s_x[TILE_M][TILE_K];
    __shared__ int8_t s_w[TILE_N][TILE_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc = 0.0f;

    // Iterate over K dimension in tiles
    for (int64_t k_base = 0; k_base < K; k_base += TILE_K) {
        // Load x tile into shared memory
        int64_t k_idx = k_base + tx;
        if (row < M && k_idx < K) {
            s_x[ty][tx] = x[row * K + k_idx];
        } else {
            s_x[ty][tx] = 0.0f;
        }

        // Load weight tile into shared memory
        if (col < N && k_idx < K) {
            s_w[tx][ty] = qweight[col * K + k_base + ty];
        } else {
            s_w[tx][ty] = 0;
        }

        __syncthreads();

        // Compute partial dot product for this tile
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                int64_t k_global = k_base + k;
                if (k_global < K) {
                    int64_t group_idx = k_global / group_size;
                    float sc = scale[col * num_groups + group_idx];
                    float w_fp = static_cast<float>(s_w[tx][k]) * sc;
                    acc += s_x[ty][k] * w_fp;
                }
            }
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

extern "C" popcornStatus_t popcornDequantizeMatmulGrouped_i8f32(
    float* out,
    const float* x,
    const int8_t* qweight,
    const float* scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size,
    cudaStream_t stream
) {
    if (out == nullptr || x == nullptr || qweight == nullptr || scale == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    if (group_size <= 0 || K % group_size != 0) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    int64_t num_groups = K / group_size;

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    dequantizeMatmulGroupedKernel_i8f32<<<grid, block, 0, stream>>>(
        out, x, qweight, scale, M, N, K, group_size, num_groups
    );

    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// INT8 Batched Dequantize + MatMul
// For each batch b: out[b,m,n] = sum_k(x[b,m,k] * qweight[n,k] * scale[n])
// Weights are shared across batches
// -----------------------------------------------------------------------------

__global__ void dequantizeMatmulBatchedKernel_i8f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const int8_t* __restrict__ qweight,
    const float* __restrict__ scale,
    int64_t B,
    int64_t M,
    int64_t N,
    int64_t K
) {
    __shared__ float s_x[TILE_M][TILE_K];
    __shared__ int8_t s_w[TILE_N][TILE_K];
    __shared__ float s_scale[TILE_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;  // batch index

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc = 0.0f;

    // Load scales for this tile's output columns
    if (ty == 0 && col < N) {
        s_scale[tx] = scale[col];
    }

    // Offset for this batch
    int64_t x_offset = bz * M * K;
    int64_t out_offset = bz * M * N;

    // Iterate over K dimension in tiles
    for (int64_t k_base = 0; k_base < K; k_base += TILE_K) {
        // Load x tile
        int64_t k_idx = k_base + tx;
        if (row < M && k_idx < K) {
            s_x[ty][tx] = x[x_offset + row * K + k_idx];
        } else {
            s_x[ty][tx] = 0.0f;
        }

        // Load weight tile (same for all batches)
        if (col < N && k_idx < K) {
            s_w[tx][ty] = qweight[col * K + k_base + ty];
        } else {
            s_w[tx][ty] = 0;
        }

        __syncthreads();

        // Compute partial dot product
        if (row < M && col < N) {
            float sc = s_scale[tx];
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                if (k_base + k < K) {
                    float w_fp = static_cast<float>(s_w[tx][k]) * sc;
                    acc += s_x[ty][k] * w_fp;
                }
            }
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        out[out_offset + row * N + col] = acc;
    }
}

extern "C" popcornStatus_t popcornDequantizeMatmulBatched_i8f32(
    float* out,
    const float* x,
    const int8_t* qweight,
    const float* scale,
    int64_t B,
    int64_t M,
    int64_t N,
    int64_t K,
    cudaStream_t stream
) {
    if (out == nullptr || x == nullptr || qweight == nullptr || scale == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (B <= 0 || M <= 0 || N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, B);

    dequantizeMatmulBatchedKernel_i8f32<<<grid, block, 0, stream>>>(
        out, x, qweight, scale, B, M, N, K
    );

    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// INT4 Per-Group Dequantize + MatMul
// Packed format: 2 int4 values per byte (low nibble first)
// Asymmetric quantization: w_fp = (w_int4 - zero) * scale
// -----------------------------------------------------------------------------

__global__ void dequantizeMatmulGroupedKernel_i4f32(
    float* __restrict__ out,
    const float* __restrict__ x,
    const uint8_t* __restrict__ qweight,
    const float* __restrict__ scale,
    const float* __restrict__ zero,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size,
    int64_t num_groups
) {
    __shared__ float s_x[TILE_M][TILE_K];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float acc = 0.0f;

    // Iterate over K dimension in tiles
    for (int64_t k_base = 0; k_base < K; k_base += TILE_K) {
        // Load x tile into shared memory
        int64_t k_idx = k_base + tx;
        if (row < M && k_idx < K) {
            s_x[ty][tx] = x[row * K + k_idx];
        } else {
            s_x[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_K; k++) {
                int64_t k_global = k_base + k;
                if (k_global < K) {
                    // Unpack INT4 from packed byte
                    int64_t byte_idx = k_global / 2;
                    uint8_t packed = qweight[col * (K / 2) + byte_idx];
                    int w_int4;
                    if (k_global % 2 == 0) {
                        w_int4 = packed & 0x0F;  // Low nibble
                    } else {
                        w_int4 = packed >> 4;    // High nibble
                    }

                    // Dequantize with zero point
                    int64_t group_idx = k_global / group_size;
                    float sc = scale[col * num_groups + group_idx];
                    float zp = zero[col * num_groups + group_idx];
                    float w_fp = (static_cast<float>(w_int4) - zp) * sc;

                    acc += s_x[ty][k] * w_fp;
                }
            }
        }

        __syncthreads();
    }

    // Store result
    if (row < M && col < N) {
        out[row * N + col] = acc;
    }
}

extern "C" popcornStatus_t popcornDequantizeMatmulGrouped_i4f32(
    float* out,
    const float* x,
    const uint8_t* qweight,
    const float* scale,
    const float* zero,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size,
    cudaStream_t stream
) {
    if (out == nullptr || x == nullptr || qweight == nullptr ||
        scale == nullptr || zero == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (M <= 0 || N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    if (group_size <= 0 || K % group_size != 0) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (K % 2 != 0) {
        return POPCORN_ERROR_INVALID_VALUE;  // K must be even for INT4 packing
    }

    int64_t num_groups = K / group_size;

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    dequantizeMatmulGroupedKernel_i4f32<<<grid, block, 0, stream>>>(
        out, x, qweight, scale, zero, M, N, K, group_size, num_groups
    );

    return checkCuda(cudaGetLastError());
}

// =============================================================================
// Quantization Kernels (float32 -> int8/int4)
// =============================================================================

// -----------------------------------------------------------------------------
// INT8 Per-Channel Quantization
// Symmetric quantization: scale = max_abs / 127, q = round(val / scale)
// -----------------------------------------------------------------------------

// Phase 1: Find max absolute value per row
__global__ void quantizeMaxAbsKernel_f32(
    float* __restrict__ max_abs,
    const float* __restrict__ input,
    int64_t N,
    int64_t K
) {
    int64_t row = blockIdx.x;
    if (row >= N) return;

    const float* row_data = input + row * K;

    // Each thread finds local max across its assigned elements
    float local_max = 0.0f;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        float v = row_data[k];
        float abs_v = v < 0 ? -v : v;
        if (abs_v > local_max) {
            local_max = abs_v;
        }
    }

    // Block reduction to find row max
    __shared__ float s_max[256];
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max[threadIdx.x + s] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_abs[row] = s_max[0];
    }
}

// Phase 2: Quantize using computed scales
__global__ void quantizeApplyKernel_i8(
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    const float* __restrict__ input,
    const float* __restrict__ max_abs,
    int64_t N,
    int64_t K
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int64_t row = idx / K;

    // Compute scale from max_abs
    float ma = max_abs[row];
    float sc = ma / 127.0f;
    if (sc == 0.0f) sc = 1.0f;

    // Store scale (once per row, first thread in row)
    if (idx % K == 0) {
        scale[row] = sc;
    }

    // Quantize
    float val = input[idx];
    float q = rintf(val / sc);
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    out[idx] = static_cast<int8_t>(q);
}

extern "C" popcornStatus_t popcornQuantize_f32i8(
    int8_t* out,
    float* scale,
    const float* input,
    int64_t N,
    int64_t K,
    cudaStream_t stream
) {
    if (out == nullptr || scale == nullptr || input == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    // Allocate temporary for max_abs
    float* d_max_abs;
    cudaError_t err = cudaMalloc(&d_max_abs, N * sizeof(float));
    if (err != cudaSuccess) {
        return checkCuda(err);
    }

    // Phase 1: Find max abs per row
    int threads = 256;
    quantizeMaxAbsKernel_f32<<<N, threads, 0, stream>>>(d_max_abs, input, N, K);

    // Phase 2: Quantize
    int64_t total = N * K;
    int blocks = (total + threads - 1) / threads;
    quantizeApplyKernel_i8<<<blocks, threads, 0, stream>>>(out, scale, input, d_max_abs, N, K);

    cudaFree(d_max_abs);

    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// INT8 Per-Group Quantization
// -----------------------------------------------------------------------------

// Phase 1: Find max abs per group
__global__ void quantizeGroupedMaxAbsKernel_f32(
    float* __restrict__ max_abs,
    const float* __restrict__ input,
    int64_t N,
    int64_t K,
    int64_t group_size,
    int64_t num_groups
) {
    // One block per (row, group) pair
    int64_t row = blockIdx.x / num_groups;
    int64_t group = blockIdx.x % num_groups;
    if (row >= N) return;

    const float* group_data = input + row * K + group * group_size;

    float local_max = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = group_data[i];
        float abs_v = v < 0 ? -v : v;
        if (abs_v > local_max) {
            local_max = abs_v;
        }
    }

    __shared__ float s_max[256];
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_max[threadIdx.x + s] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_abs[row * num_groups + group] = s_max[0];
    }
}

// Phase 2: Quantize using computed scales
__global__ void quantizeGroupedApplyKernel_i8(
    int8_t* __restrict__ out,
    float* __restrict__ scale,
    const float* __restrict__ input,
    const float* __restrict__ max_abs,
    int64_t N,
    int64_t K,
    int64_t group_size,
    int64_t num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    int64_t row = idx / K;
    int64_t k = idx % K;
    int64_t group = k / group_size;

    // Compute scale from max_abs
    float ma = max_abs[row * num_groups + group];
    float sc = ma / 127.0f;
    if (sc == 0.0f) sc = 1.0f;

    // Store scale (once per group)
    if (k % group_size == 0) {
        scale[row * num_groups + group] = sc;
    }

    // Quantize
    float val = input[idx];
    float q = rintf(val / sc);
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    out[idx] = static_cast<int8_t>(q);
}

extern "C" popcornStatus_t popcornQuantizeGrouped_f32i8(
    int8_t* out,
    float* scale,
    const float* input,
    int64_t N,
    int64_t K,
    int64_t group_size,
    cudaStream_t stream
) {
    if (out == nullptr || scale == nullptr || input == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    if (group_size <= 0 || K % group_size != 0) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    int64_t num_groups = K / group_size;

    // Allocate temporary for max_abs
    float* d_max_abs;
    cudaError_t err = cudaMalloc(&d_max_abs, N * num_groups * sizeof(float));
    if (err != cudaSuccess) {
        return checkCuda(err);
    }

    int threads = 256;

    // Phase 1: Find max abs per group
    quantizeGroupedMaxAbsKernel_f32<<<N * num_groups, threads, 0, stream>>>(
        d_max_abs, input, N, K, group_size, num_groups
    );

    // Phase 2: Quantize
    int64_t total = N * K;
    int blocks = (total + threads - 1) / threads;
    quantizeGroupedApplyKernel_i8<<<blocks, threads, 0, stream>>>(
        out, scale, input, d_max_abs, N, K, group_size, num_groups
    );

    cudaFree(d_max_abs);

    return checkCuda(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// INT4 Per-Group Quantization (Asymmetric)
// Packed format: 2 int4 values per byte (low nibble first)
// zero point allows asymmetric quantization for better range utilization
// -----------------------------------------------------------------------------

// Phase 1: Find min and max per group
__global__ void quantizeGroupedMinMaxKernel_f32(
    float* __restrict__ min_vals,
    float* __restrict__ max_vals,
    const float* __restrict__ input,
    int64_t N,
    int64_t K,
    int64_t group_size,
    int64_t num_groups
) {
    int64_t row = blockIdx.x / num_groups;
    int64_t group = blockIdx.x % num_groups;
    if (row >= N) return;

    const float* group_data = input + row * K + group * group_size;

    float local_min = INFINITY;
    float local_max = -INFINITY;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        float v = group_data[i];
        if (v < local_min) local_min = v;
        if (v > local_max) local_max = v;
    }

    __shared__ float s_min[256];
    __shared__ float s_max[256];
    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_min[threadIdx.x + s] < s_min[threadIdx.x]) {
                s_min[threadIdx.x] = s_min[threadIdx.x + s];
            }
            if (s_max[threadIdx.x + s] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int64_t gidx = row * num_groups + group;
        min_vals[gidx] = s_min[0];
        max_vals[gidx] = s_max[0];
    }
}

// Phase 2: Compute scales and zero points, then quantize and pack
__global__ void quantizeGroupedApplyKernel_i4(
    uint8_t* __restrict__ out,
    float* __restrict__ scale,
    float* __restrict__ zero,
    const float* __restrict__ input,
    const float* __restrict__ min_vals,
    const float* __restrict__ max_vals,
    int64_t N,
    int64_t K,
    int64_t group_size,
    int64_t num_groups
) {
    // Each thread handles one packed byte (2 int4 values)
    int64_t byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_bytes = N * (K / 2);
    if (byte_idx >= total_bytes) return;

    int64_t row = byte_idx / (K / 2);
    int64_t byte_in_row = byte_idx % (K / 2);
    int64_t k_lo = byte_in_row * 2;
    int64_t k_hi = k_lo + 1;

    int64_t group_lo = k_lo / group_size;
    int64_t group_hi = k_hi / group_size;

    // Get min/max for groups
    int64_t gidx_lo = row * num_groups + group_lo;
    int64_t gidx_hi = row * num_groups + group_hi;

    float min_lo = min_vals[gidx_lo];
    float max_lo = max_vals[gidx_lo];
    float min_hi = min_vals[gidx_hi];
    float max_hi = max_vals[gidx_hi];

    // Compute scale and zero for each group
    // scale = (max - min) / 15, zero = -min / scale
    float range_lo = max_lo - min_lo;
    float scale_lo = range_lo / 15.0f;
    if (scale_lo == 0.0f) scale_lo = 1.0f;
    float zero_lo = -min_lo / scale_lo;

    float range_hi = max_hi - min_hi;
    float scale_hi = range_hi / 15.0f;
    if (scale_hi == 0.0f) scale_hi = 1.0f;
    float zero_hi = -min_hi / scale_hi;

    // Store scale and zero (once per group)
    if (k_lo % group_size == 0) {
        scale[gidx_lo] = scale_lo;
        zero[gidx_lo] = zero_lo;
    }
    if (k_hi % group_size == 0) {
        scale[gidx_hi] = scale_hi;
        zero[gidx_hi] = zero_hi;
    }

    // Quantize both values
    float val_lo = input[row * K + k_lo];
    float val_hi = input[row * K + k_hi];

    float q_lo = rintf(val_lo / scale_lo + zero_lo);
    float q_hi = rintf(val_hi / scale_hi + zero_hi);

    // Clamp to [0, 15]
    if (q_lo < 0.0f) q_lo = 0.0f;
    if (q_lo > 15.0f) q_lo = 15.0f;
    if (q_hi < 0.0f) q_hi = 0.0f;
    if (q_hi > 15.0f) q_hi = 15.0f;

    // Pack: low nibble first
    uint8_t packed = (static_cast<uint8_t>(q_hi) << 4) | static_cast<uint8_t>(q_lo);
    out[byte_idx] = packed;
}

extern "C" popcornStatus_t popcornQuantizeGrouped_f32i4(
    uint8_t* out,
    float* scale,
    float* zero,
    const float* input,
    int64_t N,
    int64_t K,
    int64_t group_size,
    cudaStream_t stream
) {
    if (out == nullptr || scale == nullptr || zero == nullptr || input == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (N <= 0 || K <= 0) {
        return POPCORN_SUCCESS;
    }

    if (group_size <= 0 || K % group_size != 0) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    if (K % 2 != 0) {
        return POPCORN_ERROR_INVALID_VALUE;  // K must be even for packing
    }

    int64_t num_groups = K / group_size;

    // Allocate temporaries for min/max
    float* d_min;
    float* d_max;
    cudaError_t err = cudaMalloc(&d_min, N * num_groups * sizeof(float));
    if (err != cudaSuccess) {
        return checkCuda(err);
    }
    err = cudaMalloc(&d_max, N * num_groups * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_min);
        return checkCuda(err);
    }

    int threads = 256;

    // Phase 1: Find min/max per group
    quantizeGroupedMinMaxKernel_f32<<<N * num_groups, threads, 0, stream>>>(
        d_min, d_max, input, N, K, group_size, num_groups
    );

    // Phase 2: Quantize and pack
    int64_t total_bytes = N * (K / 2);
    int blocks = (total_bytes + threads - 1) / threads;
    quantizeGroupedApplyKernel_i4<<<blocks, threads, 0, stream>>>(
        out, scale, zero, input, d_min, d_max, N, K, group_size, num_groups
    );

    cudaFree(d_min);
    cudaFree(d_max);

    return checkCuda(cudaGetLastError());
}
