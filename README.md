# popcorn

[![CI](https://github.com/zoobzio/popcorn/actions/workflows/ci.yml/badge.svg)](https://github.com/zoobzio/popcorn/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/zoobzio/popcorn)](https://github.com/zoobzio/popcorn/releases)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

CUDA elementwise kernels for [tendo](https://github.com/zoobzio/tendo).

## Overview

popcorn provides high-performance CUDA kernels for elementwise tensor operations. It exists because cuBLAS handles matrix operations and cuDNN handles deep learning primitives, but neither exposes simple elementwise math - you need custom kernels for that.

This library fills that gap with a clean C API designed for integration via cgo.

## Operations

| Category          | Functions                                                       |
| ----------------- | --------------------------------------------------------------- |
| **Unary**         | `Neg`, `Abs`, `Exp`, `Log`, `Sqrt`, `Square`, `Sign`, `Sin`, `Cos` |
| **Activations**   | `GELU`, `LeakyReLU`                                             |
| **Binary**        | `Add`, `Sub`, `Mul`, `Div`, `Pow`                               |
| **Scalar**        | `AddScalar`, `SubScalar`, `MulScalar`, `DivScalar`, `PowScalar` |
| **Selection**     | `Clamp`, `Where`                                                |
| **Indexing**      | `Gather`, `Scatter`, `ScatterAdd`                               |
| **Reduction**     | `ArgMax`, `ArgMin`                                              |
| **Normalization** | `LayerNorm`                                                     |
| **Tensor**        | `Embedding`, `Cat`, `Stack`, `Tril`, `Split`, `Unstack`         |
| **Backward**      | `GeluBackward`, `LeakyReluBackward`, `LayerNormBackward`, `EmbeddingBackward` |
| **Optimizer**     | `AdamW`                                                         |

All operations support float32 and operate on contiguous memory. Broadcasting is handled at the consumer layer.

## Requirements

- CUDA Toolkit 11.0+
- GPU with compute capability 7.5+ (Turing or newer)
- C++17 compatible compiler

## Build

```bash
# Build the library
make

# Run tests
make test

# Clean build artifacts
make clean
```

The default architecture target is `sm_80` (Ampere). Override for your GPU:

```bash
make CUDA_ARCH=sm_75  # Turing
make CUDA_ARCH=sm_86  # GA102
make CUDA_ARCH=sm_89  # Ada
```

## Output

```
lib/libpopcorn.a    # Static library
include/popcorn.h   # C API header
```

## Integration

Link against `libpopcorn.a` and include `popcorn.h`:

```c
#include "popcorn.h"

// All functions follow this pattern:
popcornStatus_t popcornExp_f32(
    float* out,           // Output buffer (device memory)
    const float* in,      // Input buffer (device memory)
    int64_t n,            // Number of elements
    cudaStream_t stream   // CUDA stream (NULL for default)
);
```

### cgo Example

```go
/*
#cgo CFLAGS: -I/path/to/popcorn/include
#cgo LDFLAGS: -L/path/to/popcorn/lib -lpopcorn
#include "popcorn.h"
*/
import "C"

func exp(out, in uintptr, n int64) error {
    status := C.popcornExp_f32(
        (*C.float)(unsafe.Pointer(out)),
        (*C.float)(unsafe.Pointer(in)),
        C.int64_t(n),
        nil,
    )
    if status != C.POPCORN_SUCCESS {
        return errors.New(C.GoString(C.popcornGetErrorString(status)))
    }
    return nil
}
```

## API Reference

### Status Codes

```c
POPCORN_SUCCESS           // Operation completed successfully
POPCORN_ERROR_INVALID_VALUE  // NULL pointer or invalid parameter
POPCORN_ERROR_CUDA        // CUDA runtime error
```

### Unary Operations

```c
popcornStatus_t popcornNeg_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornAbs_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornExp_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLog_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSqrt_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSquare_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSign_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSin_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornCos_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornGelu_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLeakyRelu_f32(float* out, const float* in, float alpha, int64_t n, cudaStream_t stream);
```

### Binary Operations

```c
popcornStatus_t popcornAdd_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSub_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornMul_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornDiv_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornPow_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
```

### Scalar Operations

```c
popcornStatus_t popcornAddScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSubScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornMulScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornDivScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornPowScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
```

### Selection Operations

```c
popcornStatus_t popcornClamp_f32(float* out, const float* in, float minVal, float maxVal, int64_t n, cudaStream_t stream);
popcornStatus_t popcornWhere_f32(float* out, const float* cond, const float* a, const float* b, int64_t n, cudaStream_t stream);
```

### Indexing Operations

```c
// Gather: out[i] = in[i * stride + idx[i]]
popcornStatus_t popcornGather_f32(float* out, const float* in, const int64_t* idx, int64_t n, int64_t stride, cudaStream_t stream);
```

### Reduction Operations

```c
// ArgMax: out[i] = argmax(in[i*stride : i*stride+stride])
popcornStatus_t popcornArgMax_f32(int64_t* out, const float* in, int64_t n, int64_t stride, cudaStream_t stream);

// ArgMin: out[i] = argmin(in[i*stride : i*stride+stride])
popcornStatus_t popcornArgMin_f32(int64_t* out, const float* in, int64_t n, int64_t stride, cudaStream_t stream);
```

### Normalization Operations

```c
// LayerNorm: out = (in - mean) / sqrt(var + eps) * weight + bias
// weight and bias are optional (pass NULL to skip)
popcornStatus_t popcornLayerNorm_f32(float* out, const float* in, const float* weight, const float* bias, int64_t n, int64_t norm_size, float eps, cudaStream_t stream);
```

### Tensor Operations

```c
// Embedding lookup: out[i] = weight[indices[i]]
popcornStatus_t popcornEmbedding_f32(float* out, const float* weight, const int64_t* indices, int64_t n, int64_t embed_dim, cudaStream_t stream);

// Concatenate tensors along a dimension
popcornStatus_t popcornCat_f32(float* out, const float* const* inputs, int64_t num_inputs, const int64_t* sizes, int64_t outer_size, int64_t inner_size, cudaStream_t stream);

// Stack tensors along a new first dimension
popcornStatus_t popcornStack_f32(float* out, const float* const* inputs, int64_t num_inputs, int64_t tensor_size, cudaStream_t stream);

// Lower triangular mask (for causal attention)
popcornStatus_t popcornTril_f32(float* out, const float* in, int64_t rows, int64_t cols, int64_t k, cudaStream_t stream);

// Split tensor into multiple outputs (Cat backward)
popcornStatus_t popcornSplit_f32(float* const* outputs, int64_t num_outputs, const int64_t* sizes, const float* in, int64_t outer_size, int64_t inner_size, cudaStream_t stream);

// Unstack tensor into individual tensors (Stack backward)
popcornStatus_t popcornUnstack_f32(float* const* outputs, const float* in, int64_t num_outputs, int64_t tensor_size, cudaStream_t stream);

// Scatter: out[i * stride + idx[i]] = in[i]
popcornStatus_t popcornScatter_f32(float* out, const float* in, const int64_t* idx, int64_t n, int64_t stride, cudaStream_t stream);

// ScatterAdd: out[i * stride + idx[i]] += in[i] (for Gather backward)
popcornStatus_t popcornScatterAdd_f32(float* out, const float* in, const int64_t* idx, int64_t n, int64_t stride, cudaStream_t stream);
```

### Backward Operations

For autograd support:

```c
popcornStatus_t popcornGeluBackward_f32(float* grad_in, const float* grad_out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLeakyReluBackward_f32(float* grad_in, const float* grad_out, const float* in, float alpha, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLayerNormBackward_f32(float* grad_in, float* grad_weight, float* grad_bias, const float* grad_out, const float* in, const float* mean, const float* invstd, const float* weight, int64_t n, int64_t norm_size, cudaStream_t stream);
popcornStatus_t popcornEmbeddingBackward_f32(float* grad_weight, const float* grad_out, const int64_t* indices, int64_t n, int64_t embed_dim, int64_t vocab_size, cudaStream_t stream);
```

### Optimizer Operations

Fused optimizer kernels for efficient parameter updates:

```c
// AdamW: fused update of param, m, v in a single kernel
// Caller computes bias corrections: bc1 = 1 - beta1^t, bc2 = 1 - beta2^t
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
    float bias_correction1,   // 1 - beta1^t
    float bias_correction2,   // 1 - beta2^t
    int64_t n,                // number of elements
    cudaStream_t stream
);
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE)
