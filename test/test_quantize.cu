#include "test_runner.h"
#include "popcorn.h"

const float TOL_PERCHANNEL = 1e-5f;
const float TOL_PERGROUP = 1e-4f;

// Helper: allocate and copy int8_t array to device
static inline int8_t* to_device_i8(const int8_t* host, int n) {
    int8_t* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(int8_t), cudaMemcpyHostToDevice));
    return dev;
}

// Helper: allocate and copy uint8_t array to device
static inline uint8_t* to_device_u8(const uint8_t* host, int n) {
    uint8_t* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(uint8_t), cudaMemcpyHostToDevice));
    return dev;
}

// CPU reference: dequantize + matmul (per-channel)
static void ref_dequant_matmul_i8(
    float* out,
    const float* x,
    const int8_t* qweight,
    const float* scale,
    int64_t M, int64_t N, int64_t K
) {
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                float w_fp = (float)qweight[n * K + k] * scale[n];
                acc += x[m * K + k] * w_fp;
            }
            out[m * N + n] = acc;
        }
    }
}

// CPU reference: dequantize + matmul (per-group)
static void ref_dequant_matmul_grouped_i8(
    float* out,
    const float* x,
    const int8_t* qweight,
    const float* scale,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size
) {
    int64_t num_groups = K / group_size;
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                int64_t g = k / group_size;
                float sc = scale[n * num_groups + g];
                float w_fp = (float)qweight[n * K + k] * sc;
                acc += x[m * K + k] * w_fp;
            }
            out[m * N + n] = acc;
        }
    }
}

// CPU reference: INT4 per-group with zero points
static void ref_dequant_matmul_grouped_i4(
    float* out,
    const float* x,
    const uint8_t* qweight,
    const float* scale,
    const float* zero,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size
) {
    int64_t num_groups = K / group_size;
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                // Unpack int4
                int64_t byte_idx = k / 2;
                uint8_t packed = qweight[n * (K / 2) + byte_idx];
                int w_int4;
                if (k % 2 == 0) {
                    w_int4 = packed & 0x0F;
                } else {
                    w_int4 = packed >> 4;
                }

                int64_t g = k / group_size;
                float sc = scale[n * num_groups + g];
                float zp = zero[n * num_groups + g];
                float w_fp = ((float)w_int4 - zp) * sc;
                acc += x[m * K + k] * w_fp;
            }
            out[m * N + n] = acc;
        }
    }
}

// -----------------------------------------------------------------------------
// INT8 Per-Channel Tests
// -----------------------------------------------------------------------------

TEST(dequant_matmul_i8_basic) {
    // Small 2x4 @ 3x4 -> 2x3
    int64_t M = 2, N = 3, K = 4;

    float x[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    int8_t qweight[] = {
        1, 2, 3, 4,     // row 0
        -1, -2, -3, -4, // row 1
        2, 0, 2, 0      // row 2
    };
    float scale[] = {0.5f, 0.25f, 1.0f};

    float* expected = (float*)malloc(M * N * sizeof(float));
    ref_dequant_matmul_i8(expected, x, qweight, scale, M, N, K);

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmul_i8f32(d_out, d_x, d_qw, d_scale, M, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERCHANNEL);
    }

    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_i8_identity) {
    // Identity-like: scale=1, weights=identity pattern
    int64_t M = 4, N = 4, K = 4;

    float x[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f
    };
    int8_t qweight[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    float scale[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float expected[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 3.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 4.0f
    };

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmul_i8f32(d_out, d_x, d_qw, d_scale, M, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERCHANNEL);
    }

    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_i8_large) {
    // Larger matrices: 64x128 @ 256x128 -> 64x256
    int64_t M = 64, N = 256, K = 128;

    float* x = (float*)malloc(M * K * sizeof(float));
    int8_t* qweight = (int8_t*)malloc(N * K * sizeof(int8_t));
    float* scale = (float*)malloc(N * sizeof(float));
    float* expected = (float*)malloc(M * N * sizeof(float));

    // Initialize with predictable values
    for (int64_t i = 0; i < M * K; i++) {
        x[i] = ((float)(i % 100) - 50.0f) * 0.01f;
    }
    for (int64_t i = 0; i < N * K; i++) {
        qweight[i] = (int8_t)((i % 256) - 128);
    }
    for (int64_t i = 0; i < N; i++) {
        scale[i] = 0.01f + (float)(i % 10) * 0.001f;
    }

    ref_dequant_matmul_i8(expected, x, qweight, scale, M, N, K);

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmul_i8f32(d_out, d_x, d_qw, d_scale, M, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int64_t i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERCHANNEL);
    }

    free(x);
    free(qweight);
    free(scale);
    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_i8_single_row) {
    // Edge case: M=1
    int64_t M = 1, N = 4, K = 8;

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    int8_t qweight[] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        -1, -1, -1, -1, -1, -1, -1, -1,
        0, 0, 0, 0, 0, 0, 0, 0
    };
    float scale[] = {0.1f, 0.2f, 0.5f, 1.0f};

    float* expected = (float*)malloc(M * N * sizeof(float));
    ref_dequant_matmul_i8(expected, x, qweight, scale, M, N, K);

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmul_i8f32(d_out, d_x, d_qw, d_scale, M, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERCHANNEL);
    }

    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// INT8 Per-Group Tests
// -----------------------------------------------------------------------------

TEST(dequant_matmul_grouped_i8_basic) {
    // 2x8 @ 3x8 -> 2x3, group_size=4
    int64_t M = 2, N = 3, K = 8, group_size = 4;
    int64_t num_groups = K / group_size;

    float x[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f
    };
    int8_t qweight[] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        -1, -2, -3, -4, -5, -6, -7, -8,
        1, 1, 1, 1, 2, 2, 2, 2
    };
    // scales: [N, num_groups] = [3, 2]
    float scale[] = {
        0.5f, 0.25f,   // row 0: group0=0.5, group1=0.25
        1.0f, 0.5f,    // row 1
        0.1f, 0.2f     // row 2
    };

    float* expected = (float*)malloc(M * N * sizeof(float));
    ref_dequant_matmul_grouped_i8(expected, x, qweight, scale, M, N, K, group_size);

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N * num_groups);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmulGrouped_i8f32(d_out, d_x, d_qw, d_scale, M, N, K, group_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERGROUP);
    }

    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_grouped_i8_large) {
    // 32x128 @ 64x128 -> 32x64, group_size=32
    int64_t M = 32, N = 64, K = 128, group_size = 32;
    int64_t num_groups = K / group_size;

    float* x = (float*)malloc(M * K * sizeof(float));
    int8_t* qweight = (int8_t*)malloc(N * K * sizeof(int8_t));
    float* scale = (float*)malloc(N * num_groups * sizeof(float));
    float* expected = (float*)malloc(M * N * sizeof(float));

    for (int64_t i = 0; i < M * K; i++) {
        x[i] = ((float)(i % 50) - 25.0f) * 0.02f;
    }
    for (int64_t i = 0; i < N * K; i++) {
        qweight[i] = (int8_t)((i % 200) - 100);
    }
    for (int64_t i = 0; i < N * num_groups; i++) {
        scale[i] = 0.005f + (float)(i % 20) * 0.001f;
    }

    ref_dequant_matmul_grouped_i8(expected, x, qweight, scale, M, N, K, group_size);

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N * num_groups);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmulGrouped_i8f32(d_out, d_x, d_qw, d_scale, M, N, K, group_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int64_t i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERGROUP);
    }

    free(x);
    free(qweight);
    free(scale);
    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_grouped_i8_invalid_group_size) {
    // K=10 not divisible by group_size=4
    int64_t M = 2, N = 2, K = 10, group_size = 4;

    float x[20] = {0};
    int8_t qweight[20] = {0};
    float scale[6] = {0};  // Would be [2, 2.5] but that's not possible

    float* d_x = to_device(x, M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, 6);
    float* d_out = device_alloc(M * N);

    popcornStatus_t status = popcornDequantizeMatmulGrouped_i8f32(
        d_out, d_x, d_qw, d_scale, M, N, K, group_size, nullptr
    );

    if (status != POPCORN_ERROR_INVALID_VALUE) {
        printf(RED "FAIL\n" RESET);
        fprintf(stderr, "    Expected POPCORN_ERROR_INVALID_VALUE, got %d\n", status);
        tests_failed++;
        cudaFree(d_x);
        cudaFree(d_qw);
        cudaFree(d_scale);
        cudaFree(d_out);
        return;
    }

    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// INT8 Batched Tests
// -----------------------------------------------------------------------------

TEST(dequant_matmul_batched_i8_basic) {
    // B=2, 2x4 @ 3x4 -> 2x2x3
    int64_t B = 2, M = 2, N = 3, K = 4;

    float x[] = {
        // Batch 0
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        // Batch 1
        -1.0f, -2.0f, -3.0f, -4.0f,
        -5.0f, -6.0f, -7.0f, -8.0f
    };
    int8_t qweight[] = {
        1, 2, 3, 4,
        -1, -2, -3, -4,
        2, 0, 2, 0
    };
    float scale[] = {0.5f, 0.25f, 1.0f};

    // Compute expected using per-batch reference
    float* expected = (float*)malloc(B * M * N * sizeof(float));
    for (int64_t b = 0; b < B; b++) {
        ref_dequant_matmul_i8(
            expected + b * M * N,
            x + b * M * K,
            qweight,
            scale,
            M, N, K
        );
    }

    float* d_x = to_device(x, B * M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N);
    float* d_out = device_alloc(B * M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmulBatched_i8f32(d_out, d_x, d_qw, d_scale, B, M, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, B * M * N);
    for (int64_t i = 0; i < B * M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERCHANNEL);
    }

    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_batched_i8_large) {
    // B=8, 16x64 @ 32x64 -> 8x16x32
    int64_t B = 8, M = 16, N = 32, K = 64;

    float* x = (float*)malloc(B * M * K * sizeof(float));
    int8_t* qweight = (int8_t*)malloc(N * K * sizeof(int8_t));
    float* scale = (float*)malloc(N * sizeof(float));
    float* expected = (float*)malloc(B * M * N * sizeof(float));

    for (int64_t i = 0; i < B * M * K; i++) {
        x[i] = ((float)(i % 100) - 50.0f) * 0.01f;
    }
    for (int64_t i = 0; i < N * K; i++) {
        qweight[i] = (int8_t)((i % 200) - 100);
    }
    for (int64_t i = 0; i < N; i++) {
        scale[i] = 0.01f + (float)(i % 10) * 0.002f;
    }

    for (int64_t b = 0; b < B; b++) {
        ref_dequant_matmul_i8(
            expected + b * M * N,
            x + b * M * K,
            qweight,
            scale,
            M, N, K
        );
    }

    float* d_x = to_device(x, B * M * K);
    int8_t* d_qw = to_device_i8(qweight, N * K);
    float* d_scale = to_device(scale, N);
    float* d_out = device_alloc(B * M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmulBatched_i8f32(d_out, d_x, d_qw, d_scale, B, M, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, B * M * N);
    for (int64_t i = 0; i < B * M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERCHANNEL);
    }

    free(x);
    free(qweight);
    free(scale);
    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// INT4 Per-Group Tests
// -----------------------------------------------------------------------------

TEST(dequant_matmul_grouped_i4_basic) {
    // 2x8 @ 2x8 -> 2x2, group_size=4
    // K=8 means 4 packed bytes per weight row
    int64_t M = 2, N = 2, K = 8, group_size = 4;
    int64_t num_groups = K / group_size;

    float x[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f
    };

    // Packed int4: each byte has 2 values
    // Row 0: [1,2,3,4,5,6,7,8] -> bytes [0x21, 0x43, 0x65, 0x87]
    // Row 1: [0,1,2,3,4,5,6,7] -> bytes [0x10, 0x32, 0x54, 0x76]
    uint8_t qweight[] = {
        0x21, 0x43, 0x65, 0x87,  // row 0
        0x10, 0x32, 0x54, 0x76   // row 1
    };

    float scale[] = {
        0.5f, 0.25f,   // row 0: group0, group1
        1.0f, 0.5f     // row 1: group0, group1
    };
    float zero[] = {
        8.0f, 8.0f,    // row 0
        7.0f, 7.0f     // row 1
    };

    float* expected = (float*)malloc(M * N * sizeof(float));
    ref_dequant_matmul_grouped_i4(expected, x, qweight, scale, zero, M, N, K, group_size);

    float* d_x = to_device(x, M * K);
    uint8_t* d_qw = to_device_u8(qweight, N * (K / 2));
    float* d_scale = to_device(scale, N * num_groups);
    float* d_zero = to_device(zero, N * num_groups);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmulGrouped_i4f32(d_out, d_x, d_qw, d_scale, d_zero, M, N, K, group_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERGROUP);
    }

    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_zero);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_grouped_i4_large) {
    // 16x64 @ 32x64 -> 16x32, group_size=32
    int64_t M = 16, N = 32, K = 64, group_size = 32;
    int64_t num_groups = K / group_size;

    float* x = (float*)malloc(M * K * sizeof(float));
    uint8_t* qweight = (uint8_t*)malloc(N * (K / 2) * sizeof(uint8_t));
    float* scale = (float*)malloc(N * num_groups * sizeof(float));
    float* zero = (float*)malloc(N * num_groups * sizeof(float));
    float* expected = (float*)malloc(M * N * sizeof(float));

    for (int64_t i = 0; i < M * K; i++) {
        x[i] = ((float)(i % 50) - 25.0f) * 0.02f;
    }
    for (int64_t i = 0; i < N * (K / 2); i++) {
        // Random-ish packed values
        qweight[i] = (uint8_t)((i * 7 + 3) % 256);
    }
    for (int64_t i = 0; i < N * num_groups; i++) {
        scale[i] = 0.01f + (float)(i % 10) * 0.002f;
        zero[i] = 7.5f + (float)(i % 3) * 0.5f;
    }

    ref_dequant_matmul_grouped_i4(expected, x, qweight, scale, zero, M, N, K, group_size);

    float* d_x = to_device(x, M * K);
    uint8_t* d_qw = to_device_u8(qweight, N * (K / 2));
    float* d_scale = to_device(scale, N * num_groups);
    float* d_zero = to_device(zero, N * num_groups);
    float* d_out = device_alloc(M * N);

    ASSERT_SUCCESS(popcornDequantizeMatmulGrouped_i4f32(d_out, d_x, d_qw, d_scale, d_zero, M, N, K, group_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, M * N);
    for (int64_t i = 0; i < M * N; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL_PERGROUP);
    }

    free(x);
    free(qweight);
    free(scale);
    free(zero);
    free(expected);
    free(result);
    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_zero);
    cudaFree(d_out);
    PASS();
}

TEST(dequant_matmul_grouped_i4_odd_k) {
    // K must be even for INT4
    int64_t M = 2, N = 2, K = 9, group_size = 3;

    float x[18] = {0};
    uint8_t qweight[9] = {0};
    float scale[6] = {0};
    float zero[6] = {0};

    float* d_x = to_device(x, M * K);
    uint8_t* d_qw = to_device_u8(qweight, 9);
    float* d_scale = to_device(scale, 6);
    float* d_zero = to_device(zero, 6);
    float* d_out = device_alloc(M * N);

    popcornStatus_t status = popcornDequantizeMatmulGrouped_i4f32(
        d_out, d_x, d_qw, d_scale, d_zero, M, N, K, group_size, nullptr
    );

    if (status != POPCORN_ERROR_INVALID_VALUE) {
        printf(RED "FAIL\n" RESET);
        fprintf(stderr, "    Expected POPCORN_ERROR_INVALID_VALUE for odd K, got %d\n", status);
        tests_failed++;
        cudaFree(d_x);
        cudaFree(d_qw);
        cudaFree(d_scale);
        cudaFree(d_zero);
        cudaFree(d_out);
        return;
    }

    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_zero);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Quantization Tests (float32 -> int8/int4)
// -----------------------------------------------------------------------------

// CPU reference for INT8 per-channel quantization
static void ref_quantize_i8(
    int8_t* out,
    float* scale,
    const float* input,
    int64_t N,
    int64_t K
) {
    for (int64_t n = 0; n < N; n++) {
        // Find max abs
        float max_abs = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            float v = input[n * K + k];
            float abs_v = v < 0 ? -v : v;
            if (abs_v > max_abs) max_abs = abs_v;
        }

        // Compute scale
        float sc = max_abs / 127.0f;
        if (sc == 0.0f) sc = 1.0f;
        scale[n] = sc;

        // Quantize
        for (int64_t k = 0; k < K; k++) {
            float q = roundf(input[n * K + k] / sc);
            if (q > 127.0f) q = 127.0f;
            if (q < -128.0f) q = -128.0f;
            out[n * K + k] = (int8_t)q;
        }
    }
}

// CPU reference for INT8 per-group quantization
static void ref_quantize_grouped_i8(
    int8_t* out,
    float* scale,
    const float* input,
    int64_t N,
    int64_t K,
    int64_t group_size
) {
    int64_t num_groups = K / group_size;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups; g++) {
            int64_t start = n * K + g * group_size;

            // Find max abs in group
            float max_abs = 0.0f;
            for (int64_t i = 0; i < group_size; i++) {
                float v = input[start + i];
                float abs_v = v < 0 ? -v : v;
                if (abs_v > max_abs) max_abs = abs_v;
            }

            // Compute scale
            float sc = max_abs / 127.0f;
            if (sc == 0.0f) sc = 1.0f;
            scale[n * num_groups + g] = sc;

            // Quantize group
            for (int64_t i = 0; i < group_size; i++) {
                float q = roundf(input[start + i] / sc);
                if (q > 127.0f) q = 127.0f;
                if (q < -128.0f) q = -128.0f;
                out[start + i] = (int8_t)q;
            }
        }
    }
}

// CPU reference for INT4 per-group quantization
static void ref_quantize_grouped_i4(
    uint8_t* out,
    float* scale,
    float* zero,
    const float* input,
    int64_t N,
    int64_t K,
    int64_t group_size
) {
    int64_t num_groups = K / group_size;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < num_groups; g++) {
            int64_t start = n * K + g * group_size;

            // Find min/max in group
            float min_val = input[start];
            float max_val = input[start];
            for (int64_t i = 1; i < group_size; i++) {
                float v = input[start + i];
                if (v < min_val) min_val = v;
                if (v > max_val) max_val = v;
            }

            // Compute scale and zero
            float range = max_val - min_val;
            float sc = range / 15.0f;
            if (sc == 0.0f) sc = 1.0f;
            float zp = -min_val / sc;
            scale[n * num_groups + g] = sc;
            zero[n * num_groups + g] = zp;

            // Quantize and pack
            for (int64_t i = 0; i < group_size; i += 2) {
                float q_lo = roundf(input[start + i] / sc + zp);
                float q_hi = roundf(input[start + i + 1] / sc + zp);
                if (q_lo < 0.0f) q_lo = 0.0f;
                if (q_lo > 15.0f) q_lo = 15.0f;
                if (q_hi < 0.0f) q_hi = 0.0f;
                if (q_hi > 15.0f) q_hi = 15.0f;
                int64_t byte_idx = n * (K / 2) + (g * group_size + i) / 2;
                out[byte_idx] = ((uint8_t)q_hi << 4) | (uint8_t)q_lo;
            }
        }
    }
}

TEST(quantize_i8_basic) {
    int64_t N = 3, K = 4;
    float input[] = {
        1.0f, -2.0f, 3.0f, -4.0f,   // max_abs=4, scale=4/127
        0.5f, 0.5f, 0.5f, 0.5f,     // max_abs=0.5, scale=0.5/127
        127.0f, 0.0f, -127.0f, 0.0f // max_abs=127, scale=1.0
    };

    int8_t* expected_q = (int8_t*)malloc(N * K * sizeof(int8_t));
    float* expected_s = (float*)malloc(N * sizeof(float));
    ref_quantize_i8(expected_q, expected_s, input, N, K);

    float* d_input = to_device(input, N * K);
    int8_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * K * sizeof(int8_t)));
    float* d_scale = device_alloc(N);

    ASSERT_SUCCESS(popcornQuantize_f32i8(d_out, d_scale, d_input, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int8_t* result_q = (int8_t*)malloc(N * K * sizeof(int8_t));
    CUDA_CHECK(cudaMemcpy(result_q, d_out, N * K * sizeof(int8_t), cudaMemcpyDeviceToHost));
    float* result_s = to_host(d_scale, N);

    for (int64_t i = 0; i < N; i++) {
        ASSERT_NEAR(result_s[i], expected_s[i], TOL_PERCHANNEL);
    }
    for (int64_t i = 0; i < N * K; i++) {
        ASSERT_EQ((int)result_q[i], (int)expected_q[i]);
    }

    free(expected_q);
    free(expected_s);
    free(result_q);
    free(result_s);
    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_scale);
    PASS();
}

TEST(quantize_i8_large) {
    int64_t N = 64, K = 256;
    float* input = (float*)malloc(N * K * sizeof(float));
    for (int64_t i = 0; i < N * K; i++) {
        input[i] = ((float)(i % 200) - 100.0f) * 0.5f;
    }

    int8_t* expected_q = (int8_t*)malloc(N * K * sizeof(int8_t));
    float* expected_s = (float*)malloc(N * sizeof(float));
    ref_quantize_i8(expected_q, expected_s, input, N, K);

    float* d_input = to_device(input, N * K);
    int8_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * K * sizeof(int8_t)));
    float* d_scale = device_alloc(N);

    ASSERT_SUCCESS(popcornQuantize_f32i8(d_out, d_scale, d_input, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int8_t* result_q = (int8_t*)malloc(N * K * sizeof(int8_t));
    CUDA_CHECK(cudaMemcpy(result_q, d_out, N * K * sizeof(int8_t), cudaMemcpyDeviceToHost));
    float* result_s = to_host(d_scale, N);

    for (int64_t i = 0; i < N; i++) {
        ASSERT_NEAR(result_s[i], expected_s[i], TOL_PERCHANNEL);
    }
    for (int64_t i = 0; i < N * K; i++) {
        ASSERT_EQ((int)result_q[i], (int)expected_q[i]);
    }

    free(input);
    free(expected_q);
    free(expected_s);
    free(result_q);
    free(result_s);
    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_scale);
    PASS();
}

TEST(quantize_grouped_i8_basic) {
    int64_t N = 2, K = 8, group_size = 4;
    int64_t num_groups = K / group_size;

    float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f,    // group 0: max_abs=4
        -8.0f, -6.0f, -4.0f, -2.0f, // group 1: max_abs=8
        0.1f, 0.2f, 0.3f, 0.4f,    // group 0: max_abs=0.4
        12.7f, 0.0f, -12.7f, 6.35f // group 1: max_abs=12.7
    };

    int8_t* expected_q = (int8_t*)malloc(N * K * sizeof(int8_t));
    float* expected_s = (float*)malloc(N * num_groups * sizeof(float));
    ref_quantize_grouped_i8(expected_q, expected_s, input, N, K, group_size);

    float* d_input = to_device(input, N * K);
    int8_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * K * sizeof(int8_t)));
    float* d_scale = device_alloc(N * num_groups);

    ASSERT_SUCCESS(popcornQuantizeGrouped_f32i8(d_out, d_scale, d_input, N, K, group_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int8_t* result_q = (int8_t*)malloc(N * K * sizeof(int8_t));
    CUDA_CHECK(cudaMemcpy(result_q, d_out, N * K * sizeof(int8_t), cudaMemcpyDeviceToHost));
    float* result_s = to_host(d_scale, N * num_groups);

    for (int64_t i = 0; i < N * num_groups; i++) {
        ASSERT_NEAR(result_s[i], expected_s[i], TOL_PERGROUP);
    }
    for (int64_t i = 0; i < N * K; i++) {
        ASSERT_EQ((int)result_q[i], (int)expected_q[i]);
    }

    free(expected_q);
    free(expected_s);
    free(result_q);
    free(result_s);
    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_scale);
    PASS();
}

TEST(quantize_grouped_i4_basic) {
    int64_t N = 2, K = 8, group_size = 4;
    int64_t num_groups = K / group_size;

    float input[] = {
        0.0f, 1.0f, 2.0f, 3.0f,    // group 0: min=0, max=3
        4.0f, 5.0f, 6.0f, 7.0f,    // group 1: min=4, max=7
        -1.0f, 0.0f, 1.0f, 2.0f,   // group 0: min=-1, max=2
        10.0f, 11.0f, 12.0f, 13.0f // group 1: min=10, max=13
    };

    uint8_t* expected_q = (uint8_t*)malloc(N * (K / 2) * sizeof(uint8_t));
    float* expected_s = (float*)malloc(N * num_groups * sizeof(float));
    float* expected_z = (float*)malloc(N * num_groups * sizeof(float));
    ref_quantize_grouped_i4(expected_q, expected_s, expected_z, input, N, K, group_size);

    float* d_input = to_device(input, N * K);
    uint8_t* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, N * (K / 2) * sizeof(uint8_t)));
    float* d_scale = device_alloc(N * num_groups);
    float* d_zero = device_alloc(N * num_groups);

    ASSERT_SUCCESS(popcornQuantizeGrouped_f32i4(d_out, d_scale, d_zero, d_input, N, K, group_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t* result_q = (uint8_t*)malloc(N * (K / 2) * sizeof(uint8_t));
    CUDA_CHECK(cudaMemcpy(result_q, d_out, N * (K / 2) * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    float* result_s = to_host(d_scale, N * num_groups);
    float* result_z = to_host(d_zero, N * num_groups);

    for (int64_t i = 0; i < N * num_groups; i++) {
        ASSERT_NEAR(result_s[i], expected_s[i], TOL_PERGROUP);
        ASSERT_NEAR(result_z[i], expected_z[i], TOL_PERGROUP);
    }
    for (int64_t i = 0; i < N * (K / 2); i++) {
        ASSERT_EQ((int)result_q[i], (int)expected_q[i]);
    }

    free(expected_q);
    free(expected_s);
    free(expected_z);
    free(result_q);
    free(result_s);
    free(result_z);
    cudaFree(d_input);
    cudaFree(d_out);
    cudaFree(d_scale);
    cudaFree(d_zero);
    PASS();
}

TEST(quantize_roundtrip_i8) {
    // Quantize then dequantize, verify result is close to original
    int64_t N = 4, K = 8;
    float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
        0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    float* d_input = to_device(input, N * K);
    int8_t* d_qweight;
    CUDA_CHECK(cudaMalloc(&d_qweight, N * K * sizeof(int8_t)));
    float* d_scale = device_alloc(N);

    // Quantize
    ASSERT_SUCCESS(popcornQuantize_f32i8(d_qweight, d_scale, d_input, N, K, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Dequantize via matmul with identity-like x
    // x = I (but shaped for matmul: [K, K] identity, take first N rows of result)
    // Actually simpler: manually dequantize on CPU and compare
    int8_t* h_qweight = (int8_t*)malloc(N * K * sizeof(int8_t));
    CUDA_CHECK(cudaMemcpy(h_qweight, d_qweight, N * K * sizeof(int8_t), cudaMemcpyDeviceToHost));
    float* h_scale = to_host(d_scale, N);

    // Dequantize
    for (int64_t n = 0; n < N; n++) {
        for (int64_t k = 0; k < K; k++) {
            float dequant = (float)h_qweight[n * K + k] * h_scale[n];
            float orig = input[n * K + k];
            // Quantization error should be at most scale/2
            float max_error = h_scale[n] * 0.6f;  // slightly larger for rounding
            if (fabsf(dequant - orig) > max_error) {
                printf(RED "FAIL\n" RESET);
                fprintf(stderr, "    Roundtrip error too large at [%ld,%ld]: orig=%f, dequant=%f, scale=%f\n",
                        n, k, orig, dequant, h_scale[n]);
                tests_failed++;
                free(h_qweight);
                free(h_scale);
                cudaFree(d_input);
                cudaFree(d_qweight);
                cudaFree(d_scale);
                return;
            }
        }
    }

    free(h_qweight);
    free(h_scale);
    cudaFree(d_input);
    cudaFree(d_qweight);
    cudaFree(d_scale);
    PASS();
}

// -----------------------------------------------------------------------------
// Null Pointer Tests
// -----------------------------------------------------------------------------

TEST(dequant_matmul_null_ptr) {
    float* d_x = device_alloc(4);
    int8_t* d_qw = to_device_i8((int8_t[]){1,2,3,4}, 4);
    float* d_scale = to_device((float[]){1.0f}, 1);
    float* d_out = device_alloc(1);

    // Test null out
    popcornStatus_t status = popcornDequantizeMatmul_i8f32(nullptr, d_x, d_qw, d_scale, 1, 1, 4, nullptr);
    if (status != POPCORN_ERROR_INVALID_VALUE) {
        printf(RED "FAIL\n" RESET);
        fprintf(stderr, "    Expected POPCORN_ERROR_INVALID_VALUE for null out\n");
        tests_failed++;
        cudaFree(d_x);
        cudaFree(d_qw);
        cudaFree(d_scale);
        cudaFree(d_out);
        return;
    }

    // Test null x
    status = popcornDequantizeMatmul_i8f32(d_out, nullptr, d_qw, d_scale, 1, 1, 4, nullptr);
    if (status != POPCORN_ERROR_INVALID_VALUE) {
        printf(RED "FAIL\n" RESET);
        fprintf(stderr, "    Expected POPCORN_ERROR_INVALID_VALUE for null x\n");
        tests_failed++;
        cudaFree(d_x);
        cudaFree(d_qw);
        cudaFree(d_scale);
        cudaFree(d_out);
        return;
    }

    cudaFree(d_x);
    cudaFree(d_qw);
    cudaFree(d_scale);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Quantized MatMul Tests\n");
    printf("======================\n");

    // INT8 per-channel
    RUN_TEST(dequant_matmul_i8_basic);
    RUN_TEST(dequant_matmul_i8_identity);
    RUN_TEST(dequant_matmul_i8_large);
    RUN_TEST(dequant_matmul_i8_single_row);

    // INT8 per-group
    RUN_TEST(dequant_matmul_grouped_i8_basic);
    RUN_TEST(dequant_matmul_grouped_i8_large);
    RUN_TEST(dequant_matmul_grouped_i8_invalid_group_size);

    // INT8 batched
    RUN_TEST(dequant_matmul_batched_i8_basic);
    RUN_TEST(dequant_matmul_batched_i8_large);

    // INT4 per-group
    RUN_TEST(dequant_matmul_grouped_i4_basic);
    RUN_TEST(dequant_matmul_grouped_i4_large);
    RUN_TEST(dequant_matmul_grouped_i4_odd_k);

    // Error handling
    RUN_TEST(dequant_matmul_null_ptr);

    // Quantization (float32 -> int8/int4)
    RUN_TEST(quantize_i8_basic);
    RUN_TEST(quantize_i8_large);
    RUN_TEST(quantize_grouped_i8_basic);
    RUN_TEST(quantize_grouped_i4_basic);
    RUN_TEST(quantize_roundtrip_i8);

    return test_summary("Quantize");
}
