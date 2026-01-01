#include "test_runner.h"
#include "popcorn.h"

const float TOL = 1e-5f;

// Helper: allocate and copy int64_t array to device
static inline int64_t* to_device_i64(const int64_t* host, int n) {
    int64_t* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    return dev;
}

// Helper: allocate device memory for int64_t
static inline int64_t* device_alloc_i64(int n) {
    int64_t* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(int64_t)));
    return dev;
}

// Helper: copy int64_t device array to host (caller must free)
static inline int64_t* to_host_i64(const int64_t* dev, int n) {
    int64_t* host = (int64_t*)malloc(n * sizeof(int64_t));
    CUDA_CHECK(cudaMemcpy(host, dev, n * sizeof(int64_t), cudaMemcpyDeviceToHost));
    return host;
}

// -----------------------------------------------------------------------------
// Gather
// -----------------------------------------------------------------------------

TEST(gather_basic) {
    // 3 rows, 4 columns (stride=4)
    // Row 0: [0.1, 0.2, 0.3, 0.4] -> select idx=2 -> 0.3
    // Row 1: [0.5, 0.6, 0.7, 0.8] -> select idx=0 -> 0.5
    // Row 2: [0.9, 1.0, 1.1, 1.2] -> select idx=3 -> 1.2
    float input[] = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f
    };
    int64_t indices[] = {2, 0, 3};
    float expected[] = {0.3f, 0.5f, 1.2f};
    int n = 3;
    int stride = 4;

    float* d_in = to_device(input, n * stride);
    int64_t* d_idx = to_device_i64(indices, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornGather_f32(d_out, d_in, d_idx, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_idx);
    cudaFree(d_out);
    PASS();
}

TEST(gather_large) {
    int n = 10000;
    int stride = 100;  // 100 classes

    float* input = (float*)malloc(n * stride * sizeof(float));
    int64_t* indices = (int64_t*)malloc(n * sizeof(int64_t));
    float* expected = (float*)malloc(n * sizeof(float));

    // Fill with predictable data
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < stride; j++) {
            input[i * stride + j] = (float)(i * stride + j);
        }
        indices[i] = i % stride;  // Each row selects different column
        expected[i] = (float)(i * stride + indices[i]);
    }

    float* d_in = to_device(input, n * stride);
    int64_t* d_idx = to_device_i64(indices, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornGather_f32(d_out, d_in, d_idx, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(input);
    free(indices);
    free(expected);
    free(result);
    cudaFree(d_in);
    cudaFree(d_idx);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// ArgMax
// -----------------------------------------------------------------------------

TEST(argmax_basic) {
    // 3 rows, 4 columns
    float input[] = {
        0.1f, 0.4f, 0.2f, 0.3f,   // max at idx 1
        0.8f, 0.6f, 0.7f, 0.5f,   // max at idx 0
        0.1f, 0.2f, 0.3f, 0.9f    // max at idx 3
    };
    int64_t expected[] = {1, 0, 3};
    int n = 3;
    int stride = 4;

    float* d_in = to_device(input, n * stride);
    int64_t* d_out = device_alloc_i64(n);

    ASSERT_SUCCESS(popcornArgMax_f32(d_out, d_in, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int64_t* result = to_host_i64(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ((int)result[i], (int)expected[i]);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(argmax_ties) {
    // When there are ties, first occurrence wins
    float input[] = {
        1.0f, 1.0f, 0.5f, 0.5f,   // max at idx 0 (first 1.0)
        0.0f, 0.5f, 0.5f, 0.0f    // max at idx 1 (first 0.5)
    };
    int64_t expected[] = {0, 1};
    int n = 2;
    int stride = 4;

    float* d_in = to_device(input, n * stride);
    int64_t* d_out = device_alloc_i64(n);

    ASSERT_SUCCESS(popcornArgMax_f32(d_out, d_in, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int64_t* result = to_host_i64(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ((int)result[i], (int)expected[i]);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(argmax_large) {
    int n = 10000;
    int stride = 128;

    float* input = (float*)malloc(n * stride * sizeof(float));
    int64_t* expected = (int64_t*)malloc(n * sizeof(int64_t));

    // Each row has max at position (i % stride)
    for (int i = 0; i < n; i++) {
        int max_pos = i % stride;
        for (int j = 0; j < stride; j++) {
            // Use negative values as base so max_pos with 0.0 is clearly the maximum
            input[i * stride + j] = (j == max_pos) ? 0.0f : -1.0f - (float)j;
        }
        expected[i] = max_pos;
    }

    float* d_in = to_device(input, n * stride);
    int64_t* d_out = device_alloc_i64(n);

    ASSERT_SUCCESS(popcornArgMax_f32(d_out, d_in, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int64_t* result = to_host_i64(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ((int)result[i], (int)expected[i]);
    }

    free(input);
    free(expected);
    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// ArgMin
// -----------------------------------------------------------------------------

TEST(argmin_basic) {
    // 3 rows, 4 columns
    float input[] = {
        0.4f, 0.1f, 0.2f, 0.3f,   // min at idx 1
        0.5f, 0.6f, 0.7f, 0.8f,   // min at idx 0
        0.9f, 1.0f, 0.1f, 1.2f    // min at idx 2
    };
    int64_t expected[] = {1, 0, 2};
    int n = 3;
    int stride = 4;

    float* d_in = to_device(input, n * stride);
    int64_t* d_out = device_alloc_i64(n);

    ASSERT_SUCCESS(popcornArgMin_f32(d_out, d_in, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int64_t* result = to_host_i64(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ((int)result[i], (int)expected[i]);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(argmin_negative) {
    // Test with negative values
    float input[] = {
        -1.0f, -5.0f, -2.0f, 0.0f,   // min at idx 1 (-5)
        10.0f, -3.0f, -3.5f, -1.0f   // min at idx 2 (-3.5)
    };
    int64_t expected[] = {1, 2};
    int n = 2;
    int stride = 4;

    float* d_in = to_device(input, n * stride);
    int64_t* d_out = device_alloc_i64(n);

    ASSERT_SUCCESS(popcornArgMin_f32(d_out, d_in, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int64_t* result = to_host_i64(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ((int)result[i], (int)expected[i]);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// LayerNorm
// -----------------------------------------------------------------------------

TEST(layernorm_basic) {
    // 2 rows, 4 columns (norm_size=4)
    // Row 0: [1, 2, 3, 4] -> mean=2.5, var=1.25, std~=1.118
    // Row 1: [0, 0, 0, 0] -> mean=0, var=0, all zeros (with eps)
    float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    int n = 2;
    int norm_size = 4;
    float eps = 1e-5f;

    // Compute expected for row 0
    float mean0 = 2.5f;
    float var0 = ((1-2.5f)*(1-2.5f) + (2-2.5f)*(2-2.5f) + (3-2.5f)*(3-2.5f) + (4-2.5f)*(4-2.5f)) / 4.0f;
    float inv_std0 = 1.0f / sqrtf(var0 + eps);

    float expected[] = {
        (1.0f - mean0) * inv_std0,
        (2.0f - mean0) * inv_std0,
        (3.0f - mean0) * inv_std0,
        (4.0f - mean0) * inv_std0,
        0.0f, 0.0f, 0.0f, 0.0f  // Row 1: (0-0)/sqrt(0+eps) = 0
    };

    float* d_in = to_device(input, n * norm_size);
    float* d_out = device_alloc(n * norm_size);

    ASSERT_SUCCESS(popcornLayerNorm_f32(d_out, d_in, nullptr, nullptr, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(layernorm_with_weight_bias) {
    // Test with weight (gamma) and bias (beta)
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};  // Single row
    float weight[] = {2.0f, 2.0f, 2.0f, 2.0f}; // Scale by 2
    float bias[] = {1.0f, 1.0f, 1.0f, 1.0f};   // Shift by 1
    int n = 1;
    int norm_size = 4;
    float eps = 1e-5f;

    // Compute expected
    float mean = 2.5f;
    float var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + eps);

    float expected[] = {
        ((1.0f - mean) * inv_std) * 2.0f + 1.0f,
        ((2.0f - mean) * inv_std) * 2.0f + 1.0f,
        ((3.0f - mean) * inv_std) * 2.0f + 1.0f,
        ((4.0f - mean) * inv_std) * 2.0f + 1.0f
    };

    float* d_in = to_device(input, n * norm_size);
    float* d_weight = to_device(weight, norm_size);
    float* d_bias = to_device(bias, norm_size);
    float* d_out = device_alloc(n * norm_size);

    ASSERT_SUCCESS(popcornLayerNorm_f32(d_out, d_in, d_weight, d_bias, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_out);
    PASS();
}

TEST(layernorm_large) {
    int n = 1000;
    int norm_size = 768;  // Common transformer hidden size

    float* input = (float*)malloc(n * norm_size * sizeof(float));
    float* weight = (float*)malloc(norm_size * sizeof(float));
    float* bias = (float*)malloc(norm_size * sizeof(float));

    // Initialize with predictable values
    for (int i = 0; i < n * norm_size; i++) {
        input[i] = (float)(i % 100) * 0.01f;
    }
    for (int j = 0; j < norm_size; j++) {
        weight[j] = 1.0f;
        bias[j] = 0.0f;
    }

    float* d_in = to_device(input, n * norm_size);
    float* d_weight = to_device(weight, norm_size);
    float* d_bias = to_device(bias, norm_size);
    float* d_out = device_alloc(n * norm_size);

    ASSERT_SUCCESS(popcornLayerNorm_f32(d_out, d_in, d_weight, d_bias, n, norm_size, 1e-5f, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify output is normalized (mean ~ 0, var ~ 1 for each row)
    float* result = to_host(d_out, n * norm_size);

    // Check a few rows
    for (int row = 0; row < 10; row++) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int j = 0; j < norm_size; j++) {
            float v = result[row * norm_size + j];
            sum += v;
            sq_sum += v * v;
        }
        float mean = sum / norm_size;
        float var = sq_sum / norm_size - mean * mean;

        // Mean should be close to 0, variance close to 1
        ASSERT_NEAR(mean, 0.0f, 1e-4f);
        ASSERT_NEAR(var, 1.0f, 1e-3f);
    }

    free(input);
    free(weight);
    free(bias);
    free(result);
    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_out);
    PASS();
}

TEST(layernorm_inplace) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int n = 1;
    int norm_size = 4;
    float eps = 1e-5f;

    float mean = 2.5f;
    float var = 1.25f;
    float inv_std = 1.0f / sqrtf(var + eps);

    float expected[] = {
        (1.0f - mean) * inv_std,
        (2.0f - mean) * inv_std,
        (3.0f - mean) * inv_std,
        (4.0f - mean) * inv_std
    };

    float* d_buf = to_device(input, n * norm_size);

    // In-place operation
    ASSERT_SUCCESS(popcornLayerNorm_f32(d_buf, d_buf, nullptr, nullptr, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_buf, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_buf);
    PASS();
}

TEST(layernorm_with_stats) {
    // Test LayerNormWithStats outputs correct mean and invstd
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};  // Single row
    int n = 1;
    int norm_size = 4;
    float eps = 1e-5f;

    // Expected statistics
    float expected_mean = 2.5f;
    float expected_var = 1.25f;
    float expected_invstd = 1.0f / sqrtf(expected_var + eps);

    float* d_in = to_device(input, n * norm_size);
    float* d_out = device_alloc(n * norm_size);
    float* d_mean = device_alloc(n);
    float* d_invstd = device_alloc(n);

    ASSERT_SUCCESS(popcornLayerNormWithStats_f32(d_out, d_mean, d_invstd, d_in, nullptr, nullptr, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result_mean = to_host(d_mean, n);
    float* result_invstd = to_host(d_invstd, n);

    ASSERT_NEAR(result_mean[0], expected_mean, TOL);
    ASSERT_NEAR(result_invstd[0], expected_invstd, TOL);

    free(result_mean);
    free(result_invstd);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mean);
    cudaFree(d_invstd);
    PASS();
}

TEST(argmax_parallel_large_stride) {
    // Test parallel reduction for large strides (> 256)
    int n = 100;
    int stride = 1000;  // Large enough to trigger parallel kernel

    float* input = (float*)malloc(n * stride * sizeof(float));
    int64_t* expected = (int64_t*)malloc(n * sizeof(int64_t));

    // Each row has max at position (i % stride)
    for (int i = 0; i < n; i++) {
        int max_pos = (i * 7) % stride;  // Vary position
        for (int j = 0; j < stride; j++) {
            input[i * stride + j] = (j == max_pos) ? 100.0f : (float)(j % 10);
        }
        expected[i] = max_pos;
    }

    float* d_in = to_device(input, n * stride);
    int64_t* d_out = device_alloc_i64(n);

    ASSERT_SUCCESS(popcornArgMax_f32(d_out, d_in, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    int64_t* result = to_host_i64(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ((int)result[i], (int)expected[i]);
    }

    free(input);
    free(expected);
    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// RMSNorm
// -----------------------------------------------------------------------------

TEST(rmsnorm_basic) {
    // 2 rows, 4 columns
    float input[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 2.0f, 2.0f, 2.0f
    };
    int n = 2;
    int norm_size = 4;
    float eps = 1e-5f;

    // RMS for row 0: sqrt(mean([1,4,9,16])) = sqrt(7.5) = 2.7386
    // RMS for row 1: sqrt(mean([4,4,4,4])) = sqrt(4) = 2
    float rms0 = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    float rms1 = sqrtf((4.0f + 4.0f + 4.0f + 4.0f) / 4.0f + eps);

    float expected[] = {
        1.0f / rms0, 2.0f / rms0, 3.0f / rms0, 4.0f / rms0,
        2.0f / rms1, 2.0f / rms1, 2.0f / rms1, 2.0f / rms1
    };

    float* d_in = to_device(input, n * norm_size);
    float* d_out = device_alloc(n * norm_size);

    ASSERT_SUCCESS(popcornRMSNorm_f32(d_out, d_in, nullptr, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(rmsnorm_with_weight) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[] = {0.5f, 1.0f, 1.5f, 2.0f};
    int n = 1;
    int norm_size = 4;
    float eps = 1e-5f;

    float rms = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);

    float expected[] = {
        (1.0f / rms) * 0.5f,
        (2.0f / rms) * 1.0f,
        (3.0f / rms) * 1.5f,
        (4.0f / rms) * 2.0f
    };

    float* d_in = to_device(input, n * norm_size);
    float* d_weight = to_device(weight, norm_size);
    float* d_out = device_alloc(n * norm_size);

    ASSERT_SUCCESS(popcornRMSNorm_f32(d_out, d_in, d_weight, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_weight);
    cudaFree(d_out);
    PASS();
}

TEST(rmsnorm_with_stats) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int n = 1;
    int norm_size = 4;
    float eps = 1e-5f;

    float expected_rms = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    float expected_rrms = 1.0f / expected_rms;

    float* d_in = to_device(input, n * norm_size);
    float* d_out = device_alloc(n * norm_size);
    float* d_rrms = device_alloc(n);

    ASSERT_SUCCESS(popcornRMSNormWithStats_f32(d_out, d_rrms, d_in, nullptr, n, norm_size, eps, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result_rrms = to_host(d_rrms, n);
    ASSERT_NEAR(result_rrms[0], expected_rrms, TOL);

    free(result_rrms);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_rrms);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Reduction Operations Tests\n");
    printf("==========================\n");

    RUN_TEST(gather_basic);
    RUN_TEST(gather_large);
    RUN_TEST(argmax_basic);
    RUN_TEST(argmax_ties);
    RUN_TEST(argmax_large);
    RUN_TEST(argmax_parallel_large_stride);
    RUN_TEST(argmin_basic);
    RUN_TEST(argmin_negative);
    RUN_TEST(layernorm_basic);
    RUN_TEST(layernorm_with_weight_bias);
    RUN_TEST(layernorm_large);
    RUN_TEST(layernorm_inplace);
    RUN_TEST(layernorm_with_stats);
    RUN_TEST(rmsnorm_basic);
    RUN_TEST(rmsnorm_with_weight);
    RUN_TEST(rmsnorm_with_stats);

    return test_summary("Reduction");
}
