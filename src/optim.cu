#include "common.cuh"

// -----------------------------------------------------------------------------
// AdamW Fused Kernel
// Updates param, m, v in-place in a single pass
// -----------------------------------------------------------------------------

__global__ void adamw_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float bc1,
    float bc2,
    int64_t n
) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];
    float p = param[i];

    // Update moments
    float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    // Bias-corrected estimates
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;

    // Update parameter with decoupled weight decay
    param[i] = p - lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p);
}

extern "C" popcornStatus_t popcornAdamW_f32(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    float bias_correction1,
    float bias_correction2,
    int64_t n,
    cudaStream_t stream
) {
    if (param == nullptr || grad == nullptr || m == nullptr || v == nullptr) {
        return POPCORN_ERROR_INVALID_VALUE;
    }
    if (n <= 0) {
        return POPCORN_SUCCESS;
    }
    if (bias_correction1 == 0.0f || bias_correction2 == 0.0f) {
        return POPCORN_ERROR_INVALID_VALUE;
    }

    adamw_kernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(
        param, grad, m, v,
        lr, beta1, beta2, epsilon, weight_decay,
        bias_correction1, bias_correction2,
        n
    );

    return checkCuda(cudaGetLastError());
}
