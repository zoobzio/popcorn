# popcorn

[![CI](https://github.com/zoobzio/popcorn/actions/workflows/ci.yml/badge.svg)](https://github.com/zoobzio/popcorn/actions/workflows/ci.yml)
[![CodeQL](https://github.com/zoobzio/popcorn/actions/workflows/codeql.yml/badge.svg)](https://github.com/zoobzio/popcorn/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/zoobzio/popcorn)](https://github.com/zoobzio/popcorn/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

CUDA elementwise kernels for [tendo](https://github.com/zoobzio/tendo).

## Overview

popcorn provides high-performance CUDA kernels for elementwise tensor operations. It exists because cuBLAS handles matrix operations and cuDNN handles deep learning primitives, but neither exposes simple elementwise math - you need custom kernels for that.

This library fills that gap with a clean C API designed for integration via cgo.

## Operations

| Category | Functions |
|----------|-----------|
| **Unary** | `Neg`, `Abs`, `Exp`, `Log`, `Sqrt`, `Square`, `Sign` |
| **Activations** | `GELU`, `LeakyReLU` |
| **Binary** | `Add`, `Sub`, `Mul`, `Div`, `Pow` |
| **Scalar** | `AddScalar`, `SubScalar`, `MulScalar`, `DivScalar`, `PowScalar` |
| **Selection** | `Clamp`, `Where` |
| **Indexing** | `Gather` |
| **Reduction** | `ArgMax`, `ArgMin` |
| **Normalization** | `LayerNorm` |

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

[MIT](LICENSE)
