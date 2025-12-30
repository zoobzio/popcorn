# Contributing to popcorn

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/popcorn.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `make test`
6. Commit with a descriptive message
7. Push to your fork
8. Open a pull request

## Development Setup

### Prerequisites

- CUDA Toolkit 11.0+
- nvcc (included with CUDA Toolkit)
- GNU Make
- A CUDA-capable GPU for testing

### Building

```bash
# Build the library
make

# Build with specific GPU architecture
make CUDA_ARCH=sm_75

# View configuration
make info
```

### Running Tests

| Command | Description |
|---------|-------------|
| `make test` | Build and run all tests |
| `make tests` | Build tests only |
| `make clean` | Remove build artifacts |

## Code Style

### C/CUDA Conventions

- Use `snake_case` for functions and variables
- Use `UPPER_CASE` for constants and macros
- Prefix all public API functions with `popcorn`
- Use `int64_t` for sizes to match tensor libraries
- All kernels should handle edge cases (n <= 0, null pointers)

### File Organization

```
src/
  common.cuh     # Shared utilities, constants
  unary.cu       # Unary operations
  binary.cu      # Binary operations
  reduction.cu   # Gather, ArgMax/ArgMin, LayerNorm
include/
  popcorn.h      # Public C API (extern "C")
test/
  test_runner.h  # Test framework
  test_*.cu      # Test files
```

### Kernel Pattern

Follow the established pattern for new kernels:

```cuda
// 1. Define a functor for the operation
struct MyOp {
    __device__ float operator()(float x) const {
        return /* operation */;
    }
};

// 2. Use the template kernel
template <typename Op>
__global__ void unaryKernel(float* out, const float* in, int64_t n, Op op) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = op(in[i]);
    }
}

// 3. Implement the C API function
extern "C" {
popcornStatus_t popcornMyOp_f32(float* out, const float* in, int64_t n, cudaStream_t stream) {
    if (auto err = validatePtrs(out, in); err != POPCORN_SUCCESS) return err;
    if (n <= 0) return POPCORN_SUCCESS;

    unaryKernel<<<gridSize(n), BLOCK_SIZE, 0, stream>>>(out, in, n, MyOp{});
    return checkCuda(cudaGetLastError());
}
}
```

## Pull Request Process

1. Ensure all tests pass (`make test`)
2. Add tests for new functionality
3. Update `popcorn.h` if adding new public API
4. Update README.md if adding new operations
5. Use a clear PR title describing the change

## Testing Guidelines

### Writing Tests

Tests use the minimal framework in `test/test_runner.h`:

```c
TEST(my_operation_basic) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected[] = {/* expected values */};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornMyOp_f32(d_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}
```

### Test Categories

Each operation should have:

- **Basic test**: Simple cases with known outputs
- **Edge cases**: Zero, negative values, special float values where relevant
- **Large array test**: Verify grid calculation works (100k+ elements)
- **In-place test**: Verify `out == in` works correctly

Register tests in `main()`:

```c
int main() {
    printf("My Operation Tests\n");
    RUN_TEST(my_operation_basic);
    RUN_TEST(my_operation_large);
    RUN_TEST(my_operation_inplace);
    return test_summary("MyOp");
}
```

## Commit Message Format

Use conventional commits:

```
feat: add sin/cos elementwise operations
fix: correct grid size calculation for large tensors
perf: optimize memory access pattern in binary kernels
refactor: consolidate validation logic
docs: update API reference
test: add edge case tests for log operation
build: support sm_90 (Hopper) architecture
```

## Questions

If you have questions, please open an issue. We're happy to help.
