#include "common.cuh"

// -----------------------------------------------------------------------------
// GELU Backward
// -----------------------------------------------------------------------------

// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// GELU'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * sqrt(2/pi) * (1 + 3 * 0.044715 * x²)
// where u = sqrt(2/pi) * (x + 0.044715 * x^3)

__global__ void geluBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float x2 = x * x;
        float x3 = x2 * x;

        // u = sqrt(2/pi) * (x + 0.044715 * x^3)
        float u = SQRT_2_OVER_PI * (x + GELU_COEF * x3);
        float tanh_u = tanhf(u);

        // sech²(u) = 1 - tanh²(u)
        float sech2_u = 1.0f - tanh_u * tanh_u;

        // derivative of u w.r.t. x: du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x²)
        float du_dx = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEF * x2);

        // GELU'(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * sech²(u) * du/dx
        float gelu_grad = 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;

        grad_in[i] = grad_out[i] * gelu_grad;
    }
}

// -----------------------------------------------------------------------------
// LeakyReLU Backward
// -----------------------------------------------------------------------------

__global__ void leakyReluBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    float alpha,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float grad = (x > 0.0f) ? 1.0f : alpha;
        grad_in[i] = grad_out[i] * grad;
    }
}

// -----------------------------------------------------------------------------
// LayerNorm Backward
// -----------------------------------------------------------------------------

// LayerNorm forward: y = (x - mean) * invstd * weight + bias
// This backward computes grad_input, grad_weight, grad_bias
//
// The math is complex - see https://arxiv.org/abs/1502.03167 and PyTorch implementation

__global__ void layerNormBackwardKernel(
    float* grad_in,           // [n, norm_size] output
    float* grad_weight,       // [norm_size] output (can be null)
    float* grad_bias,         // [norm_size] output (can be null)
    const float* grad_out,    // [n, norm_size] input
    const float* in,          // [n, norm_size] saved input
    const float* mean,        // [n] saved mean
    const float* invstd,      // [n] saved inverse std
    const float* weight,      // [norm_size] weight (can be null)
    int64_t n,
    int64_t norm_size
) {
    // Each block handles one row (one instance in the batch)
    int64_t row = blockIdx.x;
    if (row >= n) return;

    extern __shared__ float smem[];
    float* s_sum1 = smem;                    // For sum of grad_out * (x - mean) * invstd
    float* s_sum2 = smem + blockDim.x;       // For sum of grad_out * weight (if weight exists)

    float row_mean = mean[row];
    float row_invstd = invstd[row];

    // First pass: compute sums needed for gradient
    float local_sum1 = 0.0f;
    float local_sum2 = 0.0f;

    for (int64_t j = threadIdx.x; j < norm_size; j += blockDim.x) {
        int64_t idx = row * norm_size + j;
        float x_hat = (in[idx] - row_mean) * row_invstd;
        float dy = grad_out[idx];
        float w = (weight != nullptr) ? weight[j] : 1.0f;

        local_sum1 += dy * w * x_hat;
        local_sum2 += dy * w;
    }

    s_sum1[threadIdx.x] = local_sum1;
    s_sum2[threadIdx.x] = local_sum2;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum1[threadIdx.x] += s_sum1[threadIdx.x + s];
            s_sum2[threadIdx.x] += s_sum2[threadIdx.x + s];
        }
        __syncthreads();
    }

    float sum1 = s_sum1[0];  // sum of dy * w * x_hat
    float sum2 = s_sum2[0];  // sum of dy * w

    // Second pass: compute grad_input
    float inv_norm_size = 1.0f / norm_size;

    for (int64_t j = threadIdx.x; j < norm_size; j += blockDim.x) {
        int64_t idx = row * norm_size + j;
        float x_hat = (in[idx] - row_mean) * row_invstd;
        float dy = grad_out[idx];
        float w = (weight != nullptr) ? weight[j] : 1.0f;

        // grad_input = invstd * (dy * w - mean(dy * w) - x_hat * mean(dy * w * x_hat))
        float dx = row_invstd * (dy * w - sum2 * inv_norm_size - x_hat * sum1 * inv_norm_size);
        grad_in[idx] = dx;

        // Accumulate grad_weight and grad_bias (atomic since multiple rows contribute)
        if (grad_weight != nullptr) {
            atomicAdd(&grad_weight[j], dy * x_hat);
        }
        if (grad_bias != nullptr) {
            atomicAdd(&grad_bias[j], dy);
        }
    }
}

// -----------------------------------------------------------------------------
// Embedding Backward (Scatter-Add)
// -----------------------------------------------------------------------------

// Accumulates gradients back into the embedding table
// grad_weight[indices[i]] += grad_out[i]

__global__ void embeddingBackwardKernel(
    float* grad_weight,       // [vocab_size, embed_dim] output
    const float* grad_out,    // [n, embed_dim] input
    const int64_t* indices,   // [n] indices
    int64_t n,
    int64_t embed_dim,
    int64_t vocab_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = n * embed_dim;

    if (idx < total) {
        int64_t i = idx / embed_dim;  // which token
        int64_t j = idx % embed_dim;  // which element
        int64_t token_id = indices[i];
        POPCORN_VALIDATE_INDEX(token_id, vocab_size);

        atomicAdd(&grad_weight[token_id * embed_dim + j], grad_out[idx]);
    }
}

// -----------------------------------------------------------------------------
// ReLU Backward
// -----------------------------------------------------------------------------

__global__ void reluBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = (in[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Sigmoid Backward
// -----------------------------------------------------------------------------

__global__ void sigmoidBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* out,  // sigmoid output from forward
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = out[i];
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

// -----------------------------------------------------------------------------
// Tanh Backward
// -----------------------------------------------------------------------------

__global__ void tanhBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* out,  // tanh output from forward
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float t = out[i];
        grad_in[i] = grad_out[i] * (1.0f - t * t);
    }
}

// -----------------------------------------------------------------------------
// SiLU Backward
// -----------------------------------------------------------------------------

__global__ void siluBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        // d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        grad_in[i] = grad_out[i] * (sigmoid + x * sigmoid * (1.0f - sigmoid));
    }
}

// -----------------------------------------------------------------------------
// Softmax Backward
// -----------------------------------------------------------------------------

__global__ void softmaxBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* out,  // softmax output from forward
    int64_t batch,
    int64_t dim
) {
    int64_t row = blockIdx.x;
    if (row >= batch) return;

    extern __shared__ float smem[];

    const float* row_out = out + row * dim;
    const float* row_grad_out = grad_out + row * dim;
    float* row_grad_in = grad_in + row * dim;

    // Compute dot = sum(grad_out * out) for this row
    float local_dot = 0.0f;
    for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
        local_dot += row_grad_out[j] * row_out[j];
    }

    smem[threadIdx.x] = local_dot;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float dot = smem[0];

    // grad_in = out * (grad_out - dot)
    for (int64_t j = threadIdx.x; j < dim; j += blockDim.x) {
        row_grad_in[j] = row_out[j] * (row_grad_out[j] - dot);
    }
}

// -----------------------------------------------------------------------------
// Exp Backward
// -----------------------------------------------------------------------------

__global__ void expBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* out,  // exp(input) from forward
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = grad_out[i] * out[i];
    }
}

// -----------------------------------------------------------------------------
// Log Backward
// -----------------------------------------------------------------------------

__global__ void logBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = grad_out[i] / in[i];
    }
}

// -----------------------------------------------------------------------------
// Sqrt Backward
// -----------------------------------------------------------------------------

__global__ void sqrtBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* out,  // sqrt(input) from forward
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = grad_out[i] / (2.0f * out[i]);
    }
}

// -----------------------------------------------------------------------------
// Sin Backward
// -----------------------------------------------------------------------------

__global__ void sinBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = grad_out[i] * cosf(in[i]);
    }
}

// -----------------------------------------------------------------------------
// Cos Backward
// -----------------------------------------------------------------------------

__global__ void cosBackwardKernel(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_in[i] = grad_out[i] * -sinf(in[i]);
    }
}

// -----------------------------------------------------------------------------
// RMSNorm Backward
// -----------------------------------------------------------------------------

// RMSNorm: y = x * rrms * weight, where rrms = 1/sqrt(mean(x^2) + eps)
// grad_input = rrms * (weight * grad_out - rrms^2 * x * mean(grad_out * x * weight))
// grad_weight = sum over batch of (grad_out * x * rrms)

__global__ void rmsNormBackwardKernel(
    float* grad_in,           // [n, norm_size] output
    float* grad_weight,       // [norm_size] output (can be null)
    const float* grad_out,    // [n, norm_size] input
    const float* in,          // [n, norm_size] saved input
    const float* rrms,        // [n] saved 1/rms
    const float* weight,      // [norm_size] weight (can be null)
    int64_t n,
    int64_t norm_size
) {
    // Each block handles one row (one instance in the batch)
    int64_t row = blockIdx.x;
    if (row >= n) return;

    extern __shared__ float smem[];

    float row_rrms = rrms[row];
    float row_rrms2 = row_rrms * row_rrms;

    // First pass: compute c = mean(grad_out * x * weight)
    float local_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < norm_size; j += blockDim.x) {
        int64_t idx = row * norm_size + j;
        float w = (weight != nullptr) ? weight[j] : 1.0f;
        local_sum += grad_out[idx] * in[idx] * w;
    }

    smem[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float c = smem[0] / static_cast<float>(norm_size);

    // Second pass: compute grad_input and accumulate grad_weight
    for (int64_t j = threadIdx.x; j < norm_size; j += blockDim.x) {
        int64_t idx = row * norm_size + j;
        float x = in[idx];
        float dy = grad_out[idx];
        float w = (weight != nullptr) ? weight[j] : 1.0f;

        // grad_input = rrms * (w * dy - rrms^2 * x * c)
        float dx = row_rrms * (w * dy - row_rrms2 * x * c);
        grad_in[idx] = dx;

        // grad_weight[j] += dy * x * rrms (atomic since multiple rows contribute)
        if (grad_weight != nullptr) {
            atomicAdd(&grad_weight[j], dy * x * row_rrms);
        }
    }
}

// -----------------------------------------------------------------------------
// CrossEntropy Backward (fused softmax + NLL)
// -----------------------------------------------------------------------------

__global__ void crossEntropyBackwardKernel(
    float* grad_in,
    const float* softmax,
    const int64_t* targets,
    int64_t batch,
    int64_t classes,
    float scale
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * classes;

    if (idx < total) {
        int64_t b = idx / classes;
        int64_t c = idx % classes;
        int64_t target = targets[b];

        // grad = scale * (softmax - one_hot(target))
        float one_hot = (c == target) ? 1.0f : 0.0f;
        grad_in[idx] = scale * (softmax[idx] - one_hot);
    }
}

// -----------------------------------------------------------------------------
// Scatter
// -----------------------------------------------------------------------------

// Scatters values into output at specified indices
// out[idx[i]] = in[i]  (or with atomicAdd for accumulation)

__global__ void scatterKernel(
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
        int64_t out_idx = i * stride + index;
        out[out_idx] = in[i];
    }
}

__global__ void scatterAddKernel(
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
        int64_t out_idx = i * stride + index;
        atomicAdd(&out[out_idx], in[i]);
    }
}

// -----------------------------------------------------------------------------
// Split (Cat Backward)
// -----------------------------------------------------------------------------

// Splits input tensor into multiple outputs along a dimension
__global__ void splitKernel(
    float* const* outputs,
    const int64_t* offsets,
    const float* in,
    int64_t num_outputs,
    int64_t outer_size,
    int64_t total_split_size,
    int64_t inner_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * total_split_size * inner_size;

    if (idx < total) {
        // Decompose linear index
        int64_t inner_idx = idx % inner_size;
        int64_t temp = idx / inner_size;
        int64_t split_idx = temp % total_split_size;
        int64_t outer_idx = temp / total_split_size;

        // Find which output tensor this belongs to
        int64_t output_idx = 0;
        for (int64_t i = 0; i < num_outputs; i++) {
            if (split_idx < offsets[i + 1]) {
                output_idx = i;
                break;
            }
        }

        // Calculate position within output tensor
        int64_t local_split_idx = split_idx - offsets[output_idx];
        int64_t output_split_size = offsets[output_idx + 1] - offsets[output_idx];
        int64_t out_offset = outer_idx * output_split_size * inner_size +
                             local_split_idx * inner_size +
                             inner_idx;

        outputs[output_idx][out_offset] = in[idx];
    }
}

// -----------------------------------------------------------------------------
// Unstack (Stack Backward)
// -----------------------------------------------------------------------------

// Splits stacked tensor back into individual tensors
__global__ void unstackKernel(
    float* const* outputs,
    const float* in,
    int64_t num_outputs,
    int64_t tensor_size
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = num_outputs * tensor_size;

    if (idx < total) {
        int64_t output_idx = idx / tensor_size;
        int64_t elem_idx = idx % tensor_size;
        outputs[output_idx][elem_idx] = in[idx];
    }
}

// -----------------------------------------------------------------------------
// C API Implementation
// -----------------------------------------------------------------------------

extern "C" {

popcornStatus_t popcornGeluBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    geluBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornLeakyReluBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    float alpha,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    leakyReluBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, alpha, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornLayerNormBackward_f32(
    float* grad_in,
    float* grad_weight,
    float* grad_bias,
    const float* grad_out,
    const float* in,
    const float* mean,
    const float* invstd,
    const float* weight,
    int64_t n,
    int64_t norm_size,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr ||
        mean == nullptr || invstd == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0 || norm_size <= 0) return POPCORN_SUCCESS;

    // Zero out grad_weight and grad_bias if provided (they accumulate)
    if (grad_weight != nullptr) {
        cudaMemsetAsync(grad_weight, 0, norm_size * sizeof(float), stream);
    }
    if (grad_bias != nullptr) {
        cudaMemsetAsync(grad_bias, 0, norm_size * sizeof(float), stream);
    }

    // Use one block per row, shared memory for reductions
    int threads = min((int)norm_size, 256);
    // Round up to power of 2 for reduction
    threads = 1 << (32 - __builtin_clz(threads - 1));
    threads = max(32, min(threads, 256));

    size_t smem_size = 2 * threads * sizeof(float);

    layerNormBackwardKernel<<<n, threads, smem_size, stream>>>(
        grad_in, grad_weight, grad_bias,
        grad_out, in, mean, invstd, weight,
        n, norm_size
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornEmbeddingBackward_f32(
    float* grad_weight,
    const float* grad_out,
    const int64_t* indices,
    int64_t n,
    int64_t embed_dim,
    int64_t vocab_size,
    cudaStream_t stream
) {
    if (grad_weight == nullptr || grad_out == nullptr || indices == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0 || embed_dim <= 0) return POPCORN_SUCCESS;

    // Zero out grad_weight first (accumulates with atomicAdd)
    cudaMemsetAsync(grad_weight, 0, vocab_size * embed_dim * sizeof(float), stream);

    int64_t total = n * embed_dim;
    embeddingBackwardKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        grad_weight, grad_out, indices, n, embed_dim, vocab_size
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornReluBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    reluBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSigmoidBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* out,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || out == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    sigmoidBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, out, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornTanhBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* out,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || out == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    tanhBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, out, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSiluBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    siluBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSoftmaxBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* out,
    int64_t batch,
    int64_t dim,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || out == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (batch <= 0 || dim <= 0) return POPCORN_SUCCESS;

    // One block per row, shared memory for reduction
    int threads = min((int)dim, 256);
    threads = 1 << (32 - __builtin_clz(threads - 1));
    threads = max(32, min(threads, 256));

    size_t smem_size = threads * sizeof(float);

    softmaxBackwardKernel<<<batch, threads, smem_size, stream>>>(
        grad_in, grad_out, out, batch, dim
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornCrossEntropyBackward_f32(
    float* grad_in,
    const float* softmax,
    const int64_t* targets,
    int64_t batch,
    int64_t classes,
    float scale,
    cudaStream_t stream
) {
    if (grad_in == nullptr || softmax == nullptr || targets == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (batch <= 0 || classes <= 0) return POPCORN_SUCCESS;

    int64_t total = batch * classes;
    crossEntropyBackwardKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        grad_in, softmax, targets, batch, classes, scale
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornExpBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* out,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || out == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    expBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, out, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornLogBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    logBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSqrtBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* out,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || out == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    sqrtBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, out, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSinBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    sinBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornCosBackward_f32(
    float* grad_in,
    const float* grad_out,
    const float* in,
    int64_t n,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    cosBackwardKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        grad_in, grad_out, in, n
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornRMSNormBackward_f32(
    float* grad_in,
    float* grad_weight,
    const float* grad_out,
    const float* in,
    const float* rrms,
    const float* weight,
    int64_t n,
    int64_t norm_size,
    cudaStream_t stream
) {
    if (grad_in == nullptr || grad_out == nullptr || in == nullptr || rrms == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0 || norm_size <= 0) return POPCORN_SUCCESS;

    // Zero out grad_weight if provided (it accumulates)
    if (grad_weight != nullptr) {
        cudaMemsetAsync(grad_weight, 0, norm_size * sizeof(float), stream);
    }

    // Use one block per row, shared memory for reductions
    int threads = min((int)norm_size, 256);
    threads = 1 << (32 - __builtin_clz(threads - 1));
    threads = max(32, min(threads, 256));

    size_t smem_size = threads * sizeof(float);

    rmsNormBackwardKernel<<<n, threads, smem_size, stream>>>(
        grad_in, grad_weight, grad_out, in, rrms, weight, n, norm_size
    );
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornScatter_f32(
    float* out,
    const float* in,
    const int64_t* idx,
    int64_t n,
    int64_t stride,
    cudaStream_t stream
) {
    if (out == nullptr || in == nullptr || idx == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    scatterKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, idx, n, stride);
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornScatterAdd_f32(
    float* out,
    const float* in,
    const int64_t* idx,
    int64_t n,
    int64_t stride,
    cudaStream_t stream
) {
    if (out == nullptr || in == nullptr || idx == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) return POPCORN_SUCCESS;

    scatterAddKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, idx, n, stride);
    return checkCuda(cudaGetLastError());
}

popcornStatus_t popcornSplit_f32(
    float* const* outputs,
    int64_t num_outputs,
    const int64_t* sizes,
    const float* in,
    int64_t outer_size,
    int64_t inner_size,
    cudaStream_t stream
) {
    if (outputs == nullptr || sizes == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (num_outputs <= 0 || outer_size <= 0 || inner_size <= 0) {
        return POPCORN_SUCCESS;
    }

    // Calculate total split size and build offsets
    int64_t total_split_size = 0;
    for (int64_t i = 0; i < num_outputs; i++) {
        total_split_size += sizes[i];
    }

    // Allocate device memory
    float** d_outputs;
    int64_t* d_offsets;

    cudaError_t err = cudaMallocAsync(&d_outputs, num_outputs * sizeof(float*), stream);
    if (err != cudaSuccess) return POPCORN_ERROR_CUDA;

    err = cudaMallocAsync(&d_offsets, (num_outputs + 1) * sizeof(int64_t), stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_outputs, stream);
        return POPCORN_ERROR_CUDA;
    }

    // Build offsets
    int64_t* h_offsets = new int64_t[num_outputs + 1];
    h_offsets[0] = 0;
    for (int64_t i = 0; i < num_outputs; i++) {
        h_offsets[i + 1] = h_offsets[i] + sizes[i];
    }

    err = cudaMemcpyAsync(d_outputs, outputs, num_outputs * sizeof(float*),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        delete[] h_offsets;
        cudaFreeAsync(d_outputs, stream);
        cudaFreeAsync(d_offsets, stream);
        return POPCORN_ERROR_CUDA;
    }

    err = cudaMemcpyAsync(d_offsets, h_offsets, (num_outputs + 1) * sizeof(int64_t),
                          cudaMemcpyHostToDevice, stream);
    delete[] h_offsets;
    if (err != cudaSuccess) {
        cudaFreeAsync(d_outputs, stream);
        cudaFreeAsync(d_offsets, stream);
        return POPCORN_ERROR_CUDA;
    }

    int64_t total = outer_size * total_split_size * inner_size;
    splitKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        d_outputs, d_offsets, in, num_outputs, outer_size, total_split_size, inner_size
    );

    err = cudaGetLastError();
    cudaFreeAsync(d_outputs, stream);
    cudaFreeAsync(d_offsets, stream);

    return checkCuda(err);
}

popcornStatus_t popcornUnstack_f32(
    float* const* outputs,
    const float* in,
    int64_t num_outputs,
    int64_t tensor_size,
    cudaStream_t stream
) {
    if (outputs == nullptr || in == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (num_outputs <= 0 || tensor_size <= 0) {
        return POPCORN_SUCCESS;
    }

    // Allocate device memory for outputs array
    float** d_outputs;
    cudaError_t err = cudaMallocAsync(&d_outputs, num_outputs * sizeof(float*), stream);
    if (err != cudaSuccess) return POPCORN_ERROR_CUDA;

    err = cudaMemcpyAsync(d_outputs, outputs, num_outputs * sizeof(float*),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(d_outputs, stream);
        return POPCORN_ERROR_CUDA;
    }

    int64_t total = num_outputs * tensor_size;
    unstackKernel<<<gridSize(total), BLOCK_SIZE, 0, stream>>>(
        d_outputs, in, num_outputs, tensor_size
    );

    err = cudaGetLastError();
    cudaFreeAsync(d_outputs, stream);

    return checkCuda(err);
}

} // extern "C"
