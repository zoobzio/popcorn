#include "test_runner.h"
#include "popcorn.h"

const float TOL = 1e-5f;

// -----------------------------------------------------------------------------
// Neg
// -----------------------------------------------------------------------------

TEST(neg_basic) {
    float input[] = {1.0f, -2.0f, 0.0f, 3.5f};
    float expected[] = {-1.0f, 2.0f, 0.0f, -3.5f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornNeg_f32(d_out, d_in, n, nullptr));
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

// -----------------------------------------------------------------------------
// Abs
// -----------------------------------------------------------------------------

TEST(abs_basic) {
    float input[] = {1.0f, -2.0f, 0.0f, -3.5f};
    float expected[] = {1.0f, 2.0f, 0.0f, 3.5f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornAbs_f32(d_out, d_in, n, nullptr));
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

// -----------------------------------------------------------------------------
// Exp
// -----------------------------------------------------------------------------

TEST(exp_basic) {
    float input[] = {0.0f, 1.0f, -1.0f, 2.0f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornExp_f32(d_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expf(input[i]), TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Log
// -----------------------------------------------------------------------------

TEST(log_basic) {
    float input[] = {1.0f, 2.718281828f, 10.0f, 0.5f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornLog_f32(d_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], logf(input[i]), TOL);
    }

    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Sqrt
// -----------------------------------------------------------------------------

TEST(sqrt_basic) {
    float input[] = {0.0f, 1.0f, 4.0f, 9.0f};
    float expected[] = {0.0f, 1.0f, 2.0f, 3.0f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornSqrt_f32(d_out, d_in, n, nullptr));
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

// -----------------------------------------------------------------------------
// Square
// -----------------------------------------------------------------------------

TEST(square_basic) {
    float input[] = {0.0f, 2.0f, -3.0f, 0.5f};
    float expected[] = {0.0f, 4.0f, 9.0f, 0.25f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornSquare_f32(d_out, d_in, n, nullptr));
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

// -----------------------------------------------------------------------------
// Sign
// -----------------------------------------------------------------------------

TEST(sign_basic) {
    float input[] = {5.0f, -3.0f, 0.0f, -0.001f};
    float expected[] = {1.0f, -1.0f, 0.0f, -1.0f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornSign_f32(d_out, d_in, n, nullptr));
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

// -----------------------------------------------------------------------------
// GELU
// -----------------------------------------------------------------------------

TEST(gelu_basic) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float input[] = {0.0f, 1.0f, -1.0f, 2.0f};
    int n = 4;

    // Compute expected values
    float expected[4];
    for (int i = 0; i < n; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        expected[i] = 0.5f * x * (1.0f + tanhf(inner));
    }

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornGelu_f32(d_out, d_in, n, nullptr));
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

// -----------------------------------------------------------------------------
// LeakyReLU
// -----------------------------------------------------------------------------

TEST(leaky_relu_basic) {
    float input[] = {1.0f, -1.0f, 0.0f, -5.0f};
    float alpha = 0.01f;
    float expected[] = {1.0f, -0.01f, 0.0f, -0.05f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornLeakyRelu_f32(d_out, d_in, alpha, n, nullptr));
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

// -----------------------------------------------------------------------------
// Large array test (verifies grid calculation)
// -----------------------------------------------------------------------------

TEST(exp_large) {
    int n = 100000;
    float* input = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        input[i] = (float)(i % 10) * 0.1f;  // Values 0.0 to 0.9
    }

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornExp_f32(d_out, d_in, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expf(input[i]), TOL);
    }

    free(input);
    free(result);
    cudaFree(d_in);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// In-place operation test
// -----------------------------------------------------------------------------

TEST(neg_inplace) {
    float input[] = {1.0f, -2.0f, 3.0f, -4.0f};
    float expected[] = {-1.0f, 2.0f, -3.0f, 4.0f};
    int n = 4;

    float* d_buf = to_device(input, n);

    // out == in (in-place)
    ASSERT_SUCCESS(popcornNeg_f32(d_buf, d_buf, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_buf, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_buf);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Unary Operations Tests\n");
    printf("======================\n");

    RUN_TEST(neg_basic);
    RUN_TEST(abs_basic);
    RUN_TEST(exp_basic);
    RUN_TEST(log_basic);
    RUN_TEST(sqrt_basic);
    RUN_TEST(square_basic);
    RUN_TEST(sign_basic);
    RUN_TEST(gelu_basic);
    RUN_TEST(leaky_relu_basic);
    RUN_TEST(exp_large);
    RUN_TEST(neg_inplace);

    return test_summary("Unary");
}
