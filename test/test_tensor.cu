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

// -----------------------------------------------------------------------------
// Embedding Tests
// -----------------------------------------------------------------------------

TEST(embedding_basic) {
    // Embedding table: 4 tokens, 3-dim embeddings
    // Token 0: [1, 2, 3]
    // Token 1: [4, 5, 6]
    // Token 2: [7, 8, 9]
    // Token 3: [10, 11, 12]
    float weight[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f
    };
    int64_t indices[] = {2, 0, 3, 1};  // Look up tokens 2, 0, 3, 1
    float expected[] = {
        7.0f, 8.0f, 9.0f,      // Token 2
        1.0f, 2.0f, 3.0f,      // Token 0
        10.0f, 11.0f, 12.0f,   // Token 3
        4.0f, 5.0f, 6.0f       // Token 1
    };
    int n = 4;
    int embed_dim = 3;
    int vocab_size = 4;

    float* d_weight = to_device(weight, 12);
    int64_t* d_indices = to_device_i64(indices, n);
    float* d_out = device_alloc(n * embed_dim);

    ASSERT_SUCCESS(popcornEmbedding_f32(d_out, d_weight, d_indices, n, embed_dim, vocab_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * embed_dim);
    for (int i = 0; i < n * embed_dim; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_weight);
    cudaFree(d_indices);
    cudaFree(d_out);
    PASS();
}

TEST(embedding_single) {
    float weight[] = {0.1f, 0.2f, 0.3f, 0.4f};  // 2 tokens, 2-dim
    int64_t indices[] = {1};
    float expected[] = {0.3f, 0.4f};
    int vocab_size = 2;

    float* d_weight = to_device(weight, 4);
    int64_t* d_indices = to_device_i64(indices, 1);
    float* d_out = device_alloc(2);

    ASSERT_SUCCESS(popcornEmbedding_f32(d_out, d_weight, d_indices, 1, 2, vocab_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 2);
    ASSERT_NEAR(result[0], expected[0], TOL);
    ASSERT_NEAR(result[1], expected[1], TOL);

    free(result);
    cudaFree(d_weight);
    cudaFree(d_indices);
    cudaFree(d_out);
    PASS();
}

TEST(embedding_large) {
    int vocab_size = 1000;
    int embed_dim = 128;
    int n = 512;

    // Create embedding table
    float* weight = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        weight[i] = (float)(i % 100) * 0.01f;
    }

    // Create random indices
    int64_t* indices = (int64_t*)malloc(n * sizeof(int64_t));
    for (int i = 0; i < n; i++) {
        indices[i] = i % vocab_size;
    }

    float* d_weight = to_device(weight, vocab_size * embed_dim);
    int64_t* d_indices = to_device_i64(indices, n);
    float* d_out = device_alloc(n * embed_dim);

    ASSERT_SUCCESS(popcornEmbedding_f32(d_out, d_weight, d_indices, n, embed_dim, vocab_size, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n * embed_dim);

    // Verify a few samples
    for (int i = 0; i < 10; i++) {
        int token = indices[i];
        for (int j = 0; j < embed_dim; j++) {
            float expected = weight[token * embed_dim + j];
            ASSERT_NEAR(result[i * embed_dim + j], expected, TOL);
        }
    }

    free(weight);
    free(indices);
    free(result);
    cudaFree(d_weight);
    cudaFree(d_indices);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Stack Tests
// -----------------------------------------------------------------------------

TEST(stack_basic) {
    // Stack 3 tensors of size 4
    float t0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float t1[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float t2[] = {9.0f, 10.0f, 11.0f, 12.0f};
    float expected[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };

    float* d_t0 = to_device(t0, 4);
    float* d_t1 = to_device(t1, 4);
    float* d_t2 = to_device(t2, 4);
    float* d_out = device_alloc(12);

    const float* inputs[] = {d_t0, d_t1, d_t2};

    ASSERT_SUCCESS(popcornStack_f32(d_out, inputs, 3, 4, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 12);
    for (int i = 0; i < 12; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_t0);
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_out);
    PASS();
}

TEST(stack_single) {
    float t0[] = {1.0f, 2.0f};

    float* d_t0 = to_device(t0, 2);
    float* d_out = device_alloc(2);

    const float* inputs[] = {d_t0};

    ASSERT_SUCCESS(popcornStack_f32(d_out, inputs, 1, 2, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 2);
    ASSERT_NEAR(result[0], 1.0f, TOL);
    ASSERT_NEAR(result[1], 2.0f, TOL);

    free(result);
    cudaFree(d_t0);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Cat Tests
// -----------------------------------------------------------------------------

TEST(cat_basic) {
    // Concatenate along dim 0 (outer_size=1, inner_size=1)
    // t0: [1, 2]
    // t1: [3, 4, 5]
    // result: [1, 2, 3, 4, 5]
    float t0[] = {1.0f, 2.0f};
    float t1[] = {3.0f, 4.0f, 5.0f};
    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float* d_t0 = to_device(t0, 2);
    float* d_t1 = to_device(t1, 3);
    float* d_out = device_alloc(5);

    const float* inputs[] = {d_t0, d_t1};
    int64_t sizes[] = {2, 3};

    ASSERT_SUCCESS(popcornCat_f32(d_out, inputs, 2, sizes, 1, 1, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 5);
    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_t0);
    cudaFree(d_t1);
    cudaFree(d_out);
    PASS();
}

TEST(cat_middle_dim) {
    // Cat along middle dim: [2, ?, 2]
    // t0: shape [2, 1, 2] = [[1,2], [5,6]]
    // t1: shape [2, 2, 2] = [[3,4], [7,8], [9,10], [11,12]] reshaped
    // Actually let's simplify:
    // t0: [2, 1, 2] with data [1, 2, 3, 4]
    //     meaning: outer[0] = [[1,2]], outer[1] = [[3,4]]
    // t1: [2, 2, 2] with data [5, 6, 7, 8, 9, 10, 11, 12]
    //     meaning: outer[0] = [[5,6],[7,8]], outer[1] = [[9,10],[11,12]]
    // Result: [2, 3, 2]
    //     outer[0] = [[1,2], [5,6], [7,8]]
    //     outer[1] = [[3,4], [9,10], [11,12]]
    // Flattened: [1,2,5,6,7,8,3,4,9,10,11,12]
    float t0[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float t1[] = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    float expected[] = {1.0f, 2.0f, 5.0f, 6.0f, 7.0f, 8.0f, 3.0f, 4.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    float* d_t0 = to_device(t0, 4);
    float* d_t1 = to_device(t1, 8);
    float* d_out = device_alloc(12);

    const float* inputs[] = {d_t0, d_t1};
    int64_t sizes[] = {1, 2};  // sizes along cat dim

    // outer_size = 2, inner_size = 2
    ASSERT_SUCCESS(popcornCat_f32(d_out, inputs, 2, sizes, 2, 2, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 12);
    for (int i = 0; i < 12; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_t0);
    cudaFree(d_t1);
    cudaFree(d_out);
    PASS();
}

TEST(cat_three_tensors) {
    // Cat 3 tensors along dim 0
    float t0[] = {1.0f};
    float t1[] = {2.0f, 3.0f};
    float t2[] = {4.0f};
    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f};

    float* d_t0 = to_device(t0, 1);
    float* d_t1 = to_device(t1, 2);
    float* d_t2 = to_device(t2, 1);
    float* d_out = device_alloc(4);

    const float* inputs[] = {d_t0, d_t1, d_t2};
    int64_t sizes[] = {1, 2, 1};

    ASSERT_SUCCESS(popcornCat_f32(d_out, inputs, 3, sizes, 1, 1, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 4);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_t0);
    cudaFree(d_t1);
    cudaFree(d_t2);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Tril Tests
// -----------------------------------------------------------------------------

TEST(tril_basic) {
    // 3x3 matrix, k=0 (main diagonal)
    float in[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float expected[] = {
        1.0f, 0.0f, 0.0f,
        4.0f, 5.0f, 0.0f,
        7.0f, 8.0f, 9.0f
    };

    float* d_in = to_device(in, 9);
    float* d_out = device_alloc(9);

    ASSERT_SUCCESS(popcornTril_f32(d_out, d_in, 3, 3, 0, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 9);
    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(tril_k_positive) {
    // 3x3 matrix, k=1 (include one diagonal above main)
    float in[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float expected[] = {
        1.0f, 2.0f, 0.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };

    float* d_in = to_device(in, 9);
    float* d_out = device_alloc(9);

    ASSERT_SUCCESS(popcornTril_f32(d_out, d_in, 3, 3, 1, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 9);
    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(tril_k_negative) {
    // 3x3 matrix, k=-1 (exclude main diagonal)
    float in[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    };
    float expected[] = {
        0.0f, 0.0f, 0.0f,
        4.0f, 0.0f, 0.0f,
        7.0f, 8.0f, 0.0f
    };

    float* d_in = to_device(in, 9);
    float* d_out = device_alloc(9);

    ASSERT_SUCCESS(popcornTril_f32(d_out, d_in, 3, 3, -1, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 9);
    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

TEST(tril_rectangular) {
    // 2x4 matrix, k=0
    float in[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    float expected[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        5.0f, 6.0f, 0.0f, 0.0f
    };

    float* d_in = to_device(in, 8);
    float* d_out = device_alloc(8);

    ASSERT_SUCCESS(popcornTril_f32(d_out, d_in, 2, 4, 0, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, 8);
    for (int i = 0; i < 8; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Tensor Operations Tests\n");
    printf("=======================\n");

    RUN_TEST(embedding_basic);
    RUN_TEST(embedding_single);
    RUN_TEST(embedding_large);
    RUN_TEST(stack_basic);
    RUN_TEST(stack_single);
    RUN_TEST(cat_basic);
    RUN_TEST(cat_middle_dim);
    RUN_TEST(cat_three_tensors);
    RUN_TEST(tril_basic);
    RUN_TEST(tril_k_positive);
    RUN_TEST(tril_k_negative);
    RUN_TEST(tril_rectangular);

    return test_summary("Tensor");
}
