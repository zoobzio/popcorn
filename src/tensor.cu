#include "common.cuh"

// -----------------------------------------------------------------------------
// Embedding Kernel
// -----------------------------------------------------------------------------

// Each thread handles one element of the output
// out[i * embed_dim + j] = weight[indices[i] * embed_dim + j]
__global__ void embeddingKernel(
    float* out,
    const float* weight,
    const int64_t* indices,
    int64_t n,
    int64_t embed_dim,
    int64_t vocab_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = n * embed_dim;

    if (idx < total) {
        int64_t i = idx / embed_dim;  // which token
        int64_t j = idx % embed_dim;  // which element in embedding
        int64_t token_id = indices[i];
        POPCORN_VALIDATE_INDEX(token_id, vocab_size);
        out[idx] = weight[token_id * embed_dim + j];
    }
}

// -----------------------------------------------------------------------------
// Cat Kernel
// -----------------------------------------------------------------------------

// Concatenate tensors along a dimension
// Layout: [outer_size, cat_dim, inner_size]
// Each thread copies one element from the appropriate input tensor
__global__ void catKernel(
    float* out,
    const float* const* inputs,
    const int64_t* offsets,      // cumulative offsets along cat dim
    int64_t num_inputs,
    int64_t outer_size,
    int64_t total_cat_size,      // sum of all sizes along cat dim
    int64_t inner_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * total_cat_size * inner_size;

    if (idx < total) {
        // Decompose linear index into [outer, cat, inner]
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t cat_idx = temp % total_cat_size;
        int64_t outer_idx = temp / total_cat_size;

        // Find which input tensor this cat_idx belongs to
        int64_t input_idx = 0;
        for (int64_t i = 0; i < num_inputs; i++) {
            if (cat_idx < offsets[i + 1]) {
                input_idx = i;
                break;
            }
        }

        // Calculate position within the input tensor
        int64_t local_cat_idx = cat_idx - offsets[input_idx];
        int64_t input_cat_size = offsets[input_idx + 1] - offsets[input_idx];
        int64_t input_offset = outer_idx * input_cat_size * inner_size +
                               local_cat_idx * inner_size +
                               inner_idx;

        out[idx] = inputs[input_idx][input_offset];
    }
}

// -----------------------------------------------------------------------------
// Stack Kernel
// -----------------------------------------------------------------------------

// Stack tensors along a new dimension (dimension 0)
// out[i, ...] = inputs[i][...]
// Each thread copies one element
__global__ void stackKernel(
    float* out,
    const float* const* inputs,
    int64_t num_inputs,
    int64_t tensor_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_inputs * tensor_size;

    if (idx < total) {
        int64_t input_idx = idx / tensor_size;
        int64_t elem_idx = idx % tensor_size;
        out[idx] = inputs[input_idx][elem_idx];
    }
}

// -----------------------------------------------------------------------------
// Tril Kernel (Lower Triangular)
// -----------------------------------------------------------------------------

// Zero out elements above the k-th diagonal
// out[row, col] = in[row, col] if col <= row + k, else 0
__global__ void trilKernel(
    float* out,
    const float* in,
    int64_t rows,
    int64_t cols,
    int64_t k
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = rows * cols;

    if (idx < total) {
        int64_t row = idx / cols;
        int64_t col = idx % cols;

        if (col <= row + k) {
            out[idx] = in[idx];
        } else {
            out[idx] = 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// C API Implementation
// -----------------------------------------------------------------------------

extern "C" {

popcornStatus_t popcornEmbedding_f32(
    float* out,
    const float* weight,
    const int64_t* indices,
    int64_t n,
    int64_t embed_dim,
    int64_t vocab_size,
    cudaStream_t stream
) {
    if (out == nullptr || weight == nullptr || indices == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0 || embed_dim <= 0) return POPCORN_SUCCESS;
    if (vocab_size <= 0) return POPCORN_ERROR_INVALID_VALUE;

    int64_t total = n * embed_dim;
    embeddingKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        out, weight, indices, n, embed_dim, vocab_size
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornCat_f32(
    float* out,
    const float* const* inputs,
    int64_t num_inputs,
    const int64_t* sizes,
    int64_t outer_size,
    int64_t inner_size,
    cudaStream_t stream
) {
    if (out == nullptr || inputs == nullptr || sizes == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (num_inputs <= 0 || outer_size <= 0 || inner_size <= 0) {
        return POPCORN_SUCCESS;
    }

    // Calculate total size along cat dimension and build offset array
    int64_t total_cat_size = 0;
    for (int64_t i = 0; i < num_inputs; i++) {
        total_cat_size += sizes[i];
    }

    // Allocate device memory for inputs array and offsets
    float** d_inputs;
    int64_t* d_offsets;

    cudaError_t err = cudaMallocAsync(&d_inputs, num_inputs * sizeof(float*), stream);
    if (err != cudaSuccess) return POPCORN_ERROR_CUDA;

    err = cudaMallocAsync(&d_offsets, (num_inputs + 1) * sizeof(int64_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_inputs, stream);
        return POPCORN_ERROR_CUDA;
    }

    // Build offsets array on host and copy
    int64_t* h_offsets = new int64_t[num_inputs + 1];
    h_offsets[0] = 0;
    for (int64_t i = 0; i < num_inputs; i++) {
        h_offsets[i + 1] = h_offsets[i] + sizes[i];
    }

    err = cudaMemcpyAsync(d_inputs, inputs, num_inputs * sizeof(float*),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        delete[] h_offsets;
        cudaFreeAsync(d_inputs, stream);
        cudaFreeAsync(d_offsets, stream);
        return POPCORN_ERROR_CUDA;
    }

    err = cudaMemcpyAsync(d_offsets, h_offsets, (num_inputs + 1) * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream);
    delete[] h_offsets;
    if (err != cudaSuccess) {
        cudaFreeAsync(d_inputs, stream);
        cudaFreeAsync(d_offsets, stream);
        return POPCORN_ERROR_CUDA;
    }

    int64_t total = outer_size * total_cat_size * inner_size;
    catKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        out, d_inputs, d_offsets, num_inputs, outer_size, total_cat_size, inner_size
    );

    err = cudaGetLastError();
    cudaFreeAsync(d_inputs, stream);
    cudaFreeAsync(d_offsets, stream);

    return checkCuda(err);
}

popcornStatus_t popcornStack_f32(
    float* out,
    const float* const* inputs,
    int64_t num_inputs,
    int64_t tensor_size,
    cudaStream_t stream
) {
    if (out == nullptr || inputs == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (num_inputs <= 0 || tensor_size <= 0) {
        return POPCORN_SUCCESS;
    }

    // Allocate device memory for inputs array
    float** d_inputs;
    cudaError_t err = cudaMallocAsync(&d_inputs, num_inputs * sizeof(float*), stream);
    if (err != cudaSuccess) return POPCORN_ERROR_CUDA;

    err = cudaMemcpyAsync(d_inputs, inputs, num_inputs * sizeof(float*),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_inputs, stream);
        return POPCORN_ERROR_CUDA;
    }

    int64_t total = num_inputs * tensor_size;
    stackKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        out, d_inputs, num_inputs, tensor_size
    );

    err = cudaGetLastError();
    cudaFreeAsync(d_inputs, stream);

    return checkCuda(err);
}

popcornStatus_t popcornTril_f32(
    float* out,
    const float* in,
    int64_t rows,
    int64_t cols,
    int64_t k,
    cudaStream_t stream
) {
    if (out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (rows <= 0 || cols <= 0) return POPCORN_SUCCESS;

    int64_t total = rows * cols;
    trilKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(out, in, rows, cols, k);
    return checkCuda(cudaGetLastError());
}

} // extern "C"
