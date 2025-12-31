# AdamW Fused Kernel Specification

## Overview

Fused AdamW optimizer kernel that performs all update operations in a single pass, eliminating intermediate memory allocations and reducing kernel launch overhead from ~14 to 1 per parameter tensor.

## Function Signature

```c
// Fused AdamW update: updates param, m, v in-place
// param -= lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
popcornStatus_t popcornAdamW_f32(
    float* param,             // [n] parameter tensor (updated in-place)
    const float* grad,        // [n] gradient tensor
    float* m,                 // [n] first moment (updated in-place)
    float* v,                 // [n] second moment (updated in-place)
    float lr,                 // learning rate
    float beta1,              // first moment decay (typically 0.9)
    float beta2,              // second moment decay (typically 0.999)
    float epsilon,            // numerical stability (typically 1e-8)
    float weight_decay,       // L2 penalty (typically 0.01)
    float bias_correction1,   // 1 - beta1^t (precomputed by caller)
    float bias_correction2,   // 1 - beta2^t (precomputed by caller)
    int64_t n,                // number of elements
    cudaStream_t stream
);
```

## Algorithm

For each element i in [0, n):

```
m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i]

m_hat = m[i] / bias_correction1
v_hat = v[i] / bias_correction2

param[i] -= lr * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * param[i])
```

## Reference Implementation

```cuda
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
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float g = grad[i];

    // Update moments
    float m_new = beta1 * m[i] + (1.0f - beta1) * g;
    float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    // Bias-corrected estimates
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;

    // Update parameter with weight decay
    param[i] -= lr * (m_hat / (sqrtf(v_hat) + eps) + wd * param[i]);
}
```

## Notes

- Caller computes bias corrections: `bc1 = 1 - pow(beta1, step)`, `bc2 = 1 - pow(beta2, step)`
- All tensors must be on same device
- m and v should be zero-initialized before first step
- Weight decay is decoupled (AdamW) not L2 regularisation (Adam)

## Performance Expectation

| Metric | Composed Ops | Fused |
|--------|--------------|-------|
| Kernel launches | ~14 | 1 |
| Memory reads | ~14n | 4n |
| Memory writes | ~14n | 3n |
| Temp allocations | ~6 tensors | 0 |
