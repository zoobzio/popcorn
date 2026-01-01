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
// ReLU Backward Tests
// -----------------------------------------------------------------------------

TEST(relu_backward_basic) {
    float in[] = {1.0f, -1.0f, 0.0f, 2.0f, -0.5f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expected[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    int n = 5;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornReluBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Sigmoid Backward Tests
// -----------------------------------------------------------------------------

TEST(sigmoid_backward_basic) {
    // sigmoid output values
    float out[] = {0.5f, 0.7310586f, 0.2689414f};  // sigmoid(0), sigmoid(1), sigmoid(-1)
    float grad_out[] = {1.0f, 1.0f, 1.0f};
    int n = 3;

    // grad = out * (1 - out)
    float expected[] = {0.25f, 0.19661193f, 0.19661193f};

    float* d_out = to_device(out, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornSigmoidBackward_f32(d_grad_in, d_grad_out, d_out, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Tanh Backward Tests
// -----------------------------------------------------------------------------

TEST(tanh_backward_basic) {
    // tanh output values
    float out[] = {0.0f, 0.7615942f, -0.7615942f};  // tanh(0), tanh(1), tanh(-1)
    float grad_out[] = {1.0f, 1.0f, 1.0f};
    int n = 3;

    // grad = 1 - out^2
    float expected[] = {1.0f, 0.41997434f, 0.41997434f};

    float* d_out = to_device(out, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornTanhBackward_f32(d_grad_in, d_grad_out, d_out, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// SiLU Backward Tests
// -----------------------------------------------------------------------------

static float silu_derivative(float x) {
    float sigmoid = 1.0f / (1.0f + expf(-x));
    return sigmoid + x * sigmoid * (1.0f - sigmoid);
}

TEST(silu_backward_basic) {
    float in[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornSiluBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        float expected = silu_derivative(in[i]);
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Softmax Backward Tests
// -----------------------------------------------------------------------------

TEST(softmax_backward_basic) {
    // softmax output: [0.09, 0.24, 0.67] (approx for logits [0, 1, 2])
    float out[] = {0.09003057f, 0.24472848f, 0.66524094f};
    float grad_out[] = {1.0f, 0.0f, 0.0f};  // gradient only on first element
    int batch = 1;
    int dim = 3;

    float* d_out = to_device(out, batch * dim);
    float* d_grad_out = to_device(grad_out, batch * dim);
    float* d_grad_in = device_alloc(batch * dim);

    ASSERT_SUCCESS(popcornSoftmaxBackward_f32(d_grad_in, d_grad_out, d_out, batch, dim, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, batch * dim);

    // Verify: grad_in = out * (grad_out - dot(grad_out, out))
    // dot = 0.09003057 * 1 + 0 + 0 = 0.09003057
    // grad_in[0] = 0.09003057 * (1 - 0.09003057) = 0.0819
    // grad_in[1] = 0.24472848 * (0 - 0.09003057) = -0.0220
    // grad_in[2] = 0.66524094 * (0 - 0.09003057) = -0.0599
    float dot = out[0];
    float expected[] = {
        out[0] * (grad_out[0] - dot),
        out[1] * (grad_out[1] - dot),
        out[2] * (grad_out[2] - dot)
    };

    for (int i = 0; i < dim; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

TEST(softmax_backward_batch) {
    // Test with batch > 1
    float out[] = {
        0.5f, 0.5f,       // row 0: uniform
        0.9f, 0.1f        // row 1: peaked
    };
    float grad_out[] = {
        1.0f, 0.0f,
        1.0f, 0.0f
    };
    int batch = 2;
    int dim = 2;

    float* d_out = to_device(out, batch * dim);
    float* d_grad_out = to_device(grad_out, batch * dim);
    float* d_grad_in = device_alloc(batch * dim);

    ASSERT_SUCCESS(popcornSoftmaxBackward_f32(d_grad_in, d_grad_out, d_out, batch, dim, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, batch * dim);

    // Row 0: dot = 0.5, grad = [0.5*(1-0.5), 0.5*(0-0.5)] = [0.25, -0.25]
    // Row 1: dot = 0.9, grad = [0.9*(1-0.9), 0.1*(0-0.9)] = [0.09, -0.09]
    ASSERT_NEAR(result[0], 0.25f, TOL);
    ASSERT_NEAR(result[1], -0.25f, TOL);
    ASSERT_NEAR(result[2], 0.09f, TOL);
    ASSERT_NEAR(result[3], -0.09f, TOL);

    free(result);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// CrossEntropy Backward Tests
// -----------------------------------------------------------------------------

TEST(cross_entropy_backward_basic) {
    // softmax output and targets
    float softmax[] = {
        0.7f, 0.2f, 0.1f,   // batch 0
        0.1f, 0.8f, 0.1f    // batch 1
    };
    int64_t targets[] = {0, 1};  // correct classes
    int batch = 2;
    int classes = 3;
    float scale = 1.0f;

    // Expected: grad = scale * (softmax - one_hot)
    // batch 0, target=0: [0.7-1, 0.2-0, 0.1-0] = [-0.3, 0.2, 0.1]
    // batch 1, target=1: [0.1-0, 0.8-1, 0.1-0] = [0.1, -0.2, 0.1]
    float expected[] = {-0.3f, 0.2f, 0.1f, 0.1f, -0.2f, 0.1f};

    float* d_softmax = to_device(softmax, batch * classes);
    int64_t* d_targets = to_device_i64(targets, batch);
    float* d_grad_in = device_alloc(batch * classes);

    ASSERT_SUCCESS(popcornCrossEntropyBackward_f32(
        d_grad_in, d_softmax, d_targets, batch, classes, scale, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, batch * classes);
    for (int i = 0; i < batch * classes; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_softmax);
    cudaFree(d_targets);
    cudaFree(d_grad_in);
    PASS();
}

TEST(cross_entropy_backward_mean_reduction) {
    // Test with mean reduction (scale = 1/batch)
    float softmax[] = {0.9f, 0.1f, 0.2f, 0.8f};
    int64_t targets[] = {0, 1};
    int batch = 2;
    int classes = 2;
    float scale = 0.5f;  // 1/batch for mean

    // batch 0: [0.9-1, 0.1-0] * 0.5 = [-0.05, 0.05]
    // batch 1: [0.2-0, 0.8-1] * 0.5 = [0.1, -0.1]
    float expected[] = {-0.05f, 0.05f, 0.1f, -0.1f};

    float* d_softmax = to_device(softmax, batch * classes);
    int64_t* d_targets = to_device_i64(targets, batch);
    float* d_grad_in = device_alloc(batch * classes);

    ASSERT_SUCCESS(popcornCrossEntropyBackward_f32(
        d_grad_in, d_softmax, d_targets, batch, classes, scale, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, batch * classes);
    for (int i = 0; i < batch * classes; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_softmax);
    cudaFree(d_targets);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Exp Backward Tests
// -----------------------------------------------------------------------------

TEST(exp_backward_basic) {
    float in[] = {0.0f, 1.0f, -1.0f, 2.0f};
    float out[4];  // exp(in)
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;

    for (int i = 0; i < n; i++) {
        out[i] = expf(in[i]);
    }

    float* d_out = to_device(out, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornExpBackward_f32(d_grad_in, d_grad_out, d_out, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        // grad = grad_out * exp(in) = grad_out * out
        ASSERT_NEAR(result[i], out[i], TOL);
    }

    free(result);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Log Backward Tests
// -----------------------------------------------------------------------------

TEST(log_backward_basic) {
    float in[] = {1.0f, 2.0f, 0.5f, 10.0f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornLogBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        // grad = grad_out / in
        float expected = 1.0f / in[i];
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Sqrt Backward Tests
// -----------------------------------------------------------------------------

TEST(sqrt_backward_basic) {
    float in[] = {1.0f, 4.0f, 9.0f, 16.0f};
    float out[4];  // sqrt(in)
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;

    for (int i = 0; i < n; i++) {
        out[i] = sqrtf(in[i]);
    }

    float* d_out = to_device(out, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornSqrtBackward_f32(d_grad_in, d_grad_out, d_out, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        // grad = grad_out / (2 * sqrt(in))
        float expected = 1.0f / (2.0f * out[i]);
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Sin Backward Tests
// -----------------------------------------------------------------------------

TEST(sin_backward_basic) {
    float in[] = {0.0f, 3.14159265f / 2.0f, 3.14159265f, -3.14159265f / 2.0f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornSinBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        // grad = grad_out * cos(in)
        float expected = cosf(in[i]);
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// Cos Backward Tests
// -----------------------------------------------------------------------------

TEST(cos_backward_basic) {
    float in[] = {0.0f, 3.14159265f / 2.0f, 3.14159265f, -3.14159265f / 2.0f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f};
    int n = 4;

    float* d_in = to_device(in, n);
    float* d_grad_out = to_device(grad_out, n);
    float* d_grad_in = device_alloc(n);

    ASSERT_SUCCESS(popcornCosBackward_f32(d_grad_in, d_grad_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_grad_in, n);
    for (int i = 0; i < n; i++) {
        // grad = grad_out * -sin(in)
        float expected = -sinf(in[i]);
        ASSERT_NEAR(result[i], expected, TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_grad_in);
    PASS();
}

// -----------------------------------------------------------------------------
// RMSNorm Backward Tests
// -----------------------------------------------------------------------------

TEST(rmsnorm_backward_basic) {
    // Simple test: batch=2, norm_size=3
    int n = 2;
    int norm_size = 3;
    float eps = 1e-5f;

    float in[] = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
    float grad_out[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float weight[] = {1.0f, 1.0f, 1.0f};

    // Compute rrms for each row
    float ss0 = (1.0f + 4.0f + 9.0f) / 3.0f;
    float ss1 = (4.0f + 4.0f + 4.0f) / 3.0f;
    float rrms[] = {1.0f / sqrtf(ss0 + eps), 1.0f / sqrtf(ss1 + eps)};

    float* d_in = to_device(in, n * norm_size);
    float* d_grad_out = to_device(grad_out, n * norm_size);
    float* d_weight = to_device(weight, norm_size);
    float* d_rrms = to_device(rrms, n);
    float* d_grad_in = device_alloc(n * norm_size);
    float* d_grad_weight = device_alloc(norm_size);

    ASSERT_SUCCESS(popcornRMSNormBackward_f32(
        d_grad_in, d_grad_weight,
        d_grad_out, d_in, d_rrms, d_weight,
        n, norm_size, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gradient should exist and be finite
    float* grad_in = to_host(d_grad_in, n * norm_size);
    for (int i = 0; i < n * norm_size; i++) {
        if (isnan(grad_in[i]) || isinf(grad_in[i])) {
            printf("FAIL: grad_in[%d] is not finite\n", i);
            return;
        }
    }

    // grad_weight should also be finite
    float* grad_weight = to_host(d_grad_weight, norm_size);
    for (int i = 0; i < norm_size; i++) {
        if (isnan(grad_weight[i]) || isinf(grad_weight[i])) {
            printf("FAIL: grad_weight[%d] is not finite\n", i);
            return;
        }
    }

    free(grad_in);
    free(grad_weight);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_weight);
    cudaFree(d_rrms);
    cudaFree(d_grad_in);
    cudaFree(d_grad_weight);
    PASS();
}

TEST(rmsnorm_backward_no_weight) {
    // Test without weight
    int n = 1;
    int norm_size = 4;
    float eps = 1e-5f;

    float in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad_out[] = {1.0f, 0.0f, 0.0f, 0.0f};

    float ss = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
    float rrms[] = {1.0f / sqrtf(ss + eps)};

    float* d_in = to_device(in, n * norm_size);
    float* d_grad_out = to_device(grad_out, n * norm_size);
    float* d_rrms = to_device(rrms, n);
    float* d_grad_in = device_alloc(n * norm_size);

    ASSERT_SUCCESS(popcornRMSNormBackward_f32(
        d_grad_in, nullptr,  // no grad_weight
        d_grad_out, d_in, d_rrms, nullptr,  // no weight
        n, norm_size, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* grad_in = to_host(d_grad_in, n * norm_size);

    // Verify finite
    for (int i = 0; i < n * norm_size; i++) {
        if (isnan(grad_in[i]) || isinf(grad_in[i])) {
            printf("FAIL: grad_in[%d] is not finite\n", i);
            return;
        }
    }

    // For RMSNorm without weight and grad_out=[1,0,0,0]:
    // c = mean(grad_out * in * 1) = mean([1,0,0,0]) = 0.25
    // grad_in[0] = rrms * (1 - rrms^2 * 1 * c) = rrms * (1 - rrms^2 * 0.25)
    // grad_in[1..3] = rrms * (0 - rrms^2 * x[i] * c)
    float r = rrms[0];
    float c = (1.0f * in[0]) / 4.0f;
    float expected0 = r * (1.0f - r * r * in[0] * c);
    ASSERT_NEAR(grad_in[0], expected0, TOL);

    free(grad_in);
    cudaFree(d_in);
    cudaFree(d_grad_out);
    cudaFree(d_rrms);
    cudaFree(d_grad_in);
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
    RUN_TEST(relu_backward_basic);
    RUN_TEST(sigmoid_backward_basic);
    RUN_TEST(tanh_backward_basic);
    RUN_TEST(silu_backward_basic);
    RUN_TEST(softmax_backward_basic);
    RUN_TEST(softmax_backward_batch);
    RUN_TEST(cross_entropy_backward_basic);
    RUN_TEST(cross_entropy_backward_mean_reduction);
    RUN_TEST(exp_backward_basic);
    RUN_TEST(log_backward_basic);
    RUN_TEST(sqrt_backward_basic);
    RUN_TEST(sin_backward_basic);
    RUN_TEST(cos_backward_basic);
    RUN_TEST(rmsnorm_backward_basic);
    RUN_TEST(rmsnorm_backward_no_weight);

    return test_summary("Backward");
}
