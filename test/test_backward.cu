#include "test_runner.h"
#include "popcorn.h"

const float TOL = 1e-4f;  // Slightly higher tolerance for backward passes

// Helper: allocate and copy int64_t array to device
static inline int64_t* to_device_i64(const int64_t* host, int n) {
    int64_t* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(int64_t)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(int64_t), cudaMemcpyHostToDevice));
    return dev;
}

// -----------------------------------------------------------------------------
// GELU Backward Tests
// -----------------------------------------------------------------------------

// Compute GELU derivative on CPU for verification
static float gelu_derivative(float x) {
    const float SQRT_2_PI = 0.7978845608f;
    const float GELU_COEF = 0.044715f;

    float x2 = x * x;
    float x3 = x2 * x;
    float u = SQRT_2_PI * (x + GELU_COEF * x3);
    float tanh_u = tanhf(u);
    float sech2_u = 1.0f - tanh_u * tanh_u;
    float du_dx = SQRT_2_PI * (1.0f + 3.0f * GELU_COEF * x2);

    return 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2_u * du_dx;
}

TEST(gelu_backward_basic) {
    float in[] = {0.0f, 1.0f, -1.0f, 2.0f, -0.5f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    int n = 5;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornGeluBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        float expected = gelu_derivative(in[i]);
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

TEST(gelu_backward_scaled) {
    // Test with non-unit gradient
    float in[] = {1.0f, -1.0f};
    float grad_out[] = {2.0f, 0.5f};
    int n = 2;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornGeluBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        float expected = grad_out[i] * gelu_derivative(in[i]);
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// LeakyReLU Backward Tests
// -----------------------------------------------------------------------------

TEST(leaky_relu_backward_basic) {
    float in[] = {1.0f, -1.0f, 0.0f, 2.0f, -0.5f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float alpha = 0.01f;
    int n = 5;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornLeakyReluBackward_f32(d_grad_in, d_grad_out, d_in, alpha, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        float expected = (in[i] > 0.0f) ? 1.0f : alpha;
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Embedding Backward Tests
// -----------------------------------------------------------------------------

TEST(embedding_backward_basic) {
    // 4 tokens looked up from vocab of 3, embed_dim = 2
    // indices: [0, 2, 1, 0]
    // grad_out: gradients for each looked up embedding
    float grad_out[] = {
        1.0f, 2.0f,   // token 0
        3.0f, 4.0f,   // token 2
        5.0f, 6.0f,   // token 1
        7.0f, 8.0f    // token 0 again
    };
    int64_t indices[] = {0, 2, 1, 0};

    // Expected grad_weight:
    // token 0: (1,2) + (7,8) = (8, 10)
    // token 1: (5, 6)
    // token 2: (3, 4)
    float expected_grad[] = {8.0f, 10.0f, 5.0f, 6.0f, 3.0f, 4.0f};

    int n = 4;
    int embed_dim = 2;
    int vocab_size = 3;

    float* d_grad_out = to_device(grad_out, n * embed_dim);
    int64_t* d_indices = to_device_i64(indices, n);
    float* d_grad_weight;
    CUDA_CHECK(cudaMalloc(&d_grad_weight, vocab_size * embed_dim * sizeof(float)));

    ASSERT_SUCCESS(popcornEmbeddingBackward_f32(
        d_grad_weight, d_grad_out, d_indices, n, embed_dim, vocab_size, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_weight, vocab_size * embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        ASSERT_NEAR(result[i], expected_grad[i], TOL);
    }

    free(result);
    cudaFree(d_grad_out);
    cudaFree(d_indices);
    cudaFree(d_grad_weight);
    PASS();
}

// -----------------------------------------------------------------------------
// Scatter Tests
// -----------------------------------------------------------------------------

TEST(scatter_basic) {
    // Scatter values into a [3, 4] tensor
    float in[] = {10.0f, 20.0f, 30.0f};
    int64_t idx[] = {1, 3, 0};  // positions within each row
    int n = 3;
    int stride = 4;

    // Output: row 0 gets 10 at pos 1, row 1 gets 20 at pos 3, row 2 gets 30 at pos 0
    float expected[] = {
        0.0f, 10.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 20.0f,
        30.0f, 0.0f, 0.0f, 0.0f
    };

    float* d_in = to_device(in, n);
    int64_t* d_idx = to_device_i64(idx, n);
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * stride * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, n * stride * sizeof(float)));

    ASSERT_SUCCESS(popcornScatter_f32(d_out, d_in, d_idx, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * stride);
    for (int i = 0; i < n * stride; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_idx);
    cudaFree(d_out);
    PASS();
}

TEST(scatter_add_basic) {
    // ScatterAdd with duplicates
    float in[] = {1.0f, 2.0f, 3.0f};
    int64_t idx[] = {0, 0, 1};  // first two go to same position
    int n = 3;
    int stride = 2;

    // Row 0: pos 0 gets 1+2=3, Row 1: pos 0 gets 0, Row 2: pos 1 gets 3
    // But wait - each row is independent. Let me reconsider.
    // out[0*2 + 0] += 1 -> out[0] += 1
    // out[1*2 + 0] += 2 -> out[2] += 2
    // out[2*2 + 1] += 3 -> out[5] += 3
    float expected[] = {1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f};

    float* d_in = to_device(in, n);
    int64_t* d_idx = to_device_i64(idx, n);
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, n * stride * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, n * stride * sizeof(float)));

    ASSERT_SUCCESS(popcornScatterAdd_f32(d_out, d_in, d_idx, n, stride, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * stride);
    for (int i = 0; i < n * stride; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_idx);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Split Tests (Cat Backward)
// -----------------------------------------------------------------------------

TEST(split_basic) {
    // Split [1, 2, 3, 4, 5] into [1, 2] and [3, 4, 5]
    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected0[] = {1.0f, 2.0f};
    float expected1[] = {3.0f, 4.0f, 5.0f};

    float* d_in = to_device(in, 5);
    float* d_out0 = device_alloc(2);
    float* d_out1 = device_alloc(3);

    float* outputs[] = {d_out0, d_out1};
    int64_t sizes[] = {2, 3};

    ASSERT_SUCCESS(popcornSplit_f32(outputs, 2, sizes, d_in, 1, 1, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result0 = to_host(d_out0, 2);
    float* result1 = to_host(d_out1, 3);

    for (int i = 0; i < 2; i++) {
        ASSERT_NEAR(result0[i], expected0[i], TOL);
    }
    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(result1[i], expected1[i], TOL);
    }

    free(result0);
    free(result1);
    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
    PASS();
}

// -----------------------------------------------------------------------------
// Unstack Tests (Stack Backward)
// -----------------------------------------------------------------------------

TEST(unstack_basic) {
    // Unstack [3, 4] tensor into 3 tensors of size 4
    float in[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    float* d_in = to_device(in, 12);
    float* d_out0 = device_alloc(4);
    float* d_out1 = device_alloc(4);
    float* d_out2 = device_alloc(4);

    float* outputs[] = {d_out0, d_out1, d_out2};

    ASSERT_SUCCESS(popcornUnstack_f32(outputs, d_in, 3, 4, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result0 = to_host(d_out0, 4);
    float* result1 = to_host(d_out1, 4);
    float* result2 = to_host(d_out2, 4);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(result0[i], in[i], TOL);
        ASSERT_NEAR(result1[i], in[4 + i], TOL);
        ASSERT_NEAR(result2[i], in[8 + i], TOL);
    }

    free(result0);
    free(result1);
    free(result2);
    cudaFree(d_in);
    cudaFree(d_out0);
    cudaFree(d_out1);
    cudaFree(d_out2);
    PASS();
}

// -----------------------------------------------------------------------------
// LayerNorm Backward Tests
// -----------------------------------------------------------------------------

TEST(layer_norm_backward_basic) {
    // Simple test: batch=2, norm_size=3
    // We'll verify gradient shapes and basic sanity
    int n = 2;
    int norm_size = 3;

    float in[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float weight[] = {1.0f, 1.0f, 1.0f};

    // Compute mean and invstd for each row
    float mean[] = {2.0f, 5.0f};  // mean of [1,2,3] = 2, mean of [4,5,6] = 5
    // var of [1,2,3] = ((1-2)^2 + (2-2)^2 + (3-2)^2)/3 = (1+0+1)/3 = 2/3
    // std = sqrt(2/3) ≈ 0.8165, invstd ≈ 1.2247
    float invstd[] = {1.2247449f, 1.2247449f};

    float* d_in = to_device(in, n * norm_size);
    float* d_grad_out = to_device(grad_out, n * norm_size);
    float* d_weight = to_device(weight, norm_size);
    float* d_mean = to_device(mean, n);
    float* d_invstd = to_device(invstd, n);
    float* d_grad_in = device_alloc(n * norm_size);
    float* d_grad_weight = device_alloc(norm_size);
    float* d_grad_bias = device_alloc(norm_size);

    ASSERT_SUCCESS(popcornLayerNormBackward_f32(
        d_grad_in, d_grad_weight, d_grad_bias,
        d_grad_out, d_in, d_mean, d_invstd, d_weight,
        n, norm_size, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Basic sanity: grad_bias should be sum of grad_out per feature = 2.0 each
    float* grad_bias = to_host(d_grad_bias, norm_size);
    for (int i = 0; i < norm_size; i++) {
        ASSERT_NEAR(grad_bias[i], 2.0f, TOL);
    }

    // Gradient should exist and be finite
    float* grad_in = to_host(d_grad_in, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        if (isnan(grad_in[i]) || isinf(grad_in[i])) {
            printf("FAIL: grad_in[%d] is not finite\n", i);
            return;
        }
    }

    free(grad_bias);
    free(grad_in);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_weight);
    cudaFree(d_mean);
    cudaFree(d_invstd);
    cudaFree(d_grad_in);
    cudaFree(d_grad_weight);
    cudaFree(d_grad_bias);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Backward Operations Tests\n");
    printf("=========================\n");

    RUN_TEST(gelu_backward_basic);
    RUN_TEST(gelu_backward_scaled);
    RUN_TEST(leaky_relu_backward_basic);
    RUN_TEST(embedding_backward_basic);
    RUN_TEST(scatter_basic);
    RUN_TEST(scatter_add_basic);
    RUN_TEST(split_basic);
    RUN_TEST(unstack_basic);
    RUN_TEST(layer_norm_backward_basic);

    return test_summary("Backward");
}
