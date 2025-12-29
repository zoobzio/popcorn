#include "test_runner.h"
#include "popcorn.h"

const float TOL = 1e-5f;

// -----------------------------------------------------------------------------
// Add
// -----------------------------------------------------------------------------

TEST(add_basic) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float expected[] = {6.0f, 8.0f, 10.0f, 12.0f};
    int n = 4;

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornAdd_f32(d_out, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Sub
// -----------------------------------------------------------------------------

TEST(sub_basic) {
    float a[] = {5.0f, 6.0f, 7.0f, 8.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected[] = {4.0f, 4.0f, 4.0f, 4.0f};
    int n = 4;

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornSub_f32(d_out, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Mul
// -----------------------------------------------------------------------------

TEST(mul_basic) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {2.0f, 3.0f, 4.0f, 5.0f};
    float expected[] = {2.0f, 6.0f, 12.0f, 20.0f};
    int n = 4;

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornMul_f32(d_out, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Div
// -----------------------------------------------------------------------------

TEST(div_basic) {
    float a[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float b[] = {2.0f, 4.0f, 5.0f, 8.0f};
    float expected[] = {5.0f, 5.0f, 6.0f, 5.0f};
    int n = 4;

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornDiv_f32(d_out, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Pow
// -----------------------------------------------------------------------------

TEST(pow_basic) {
    float a[] = {2.0f, 3.0f, 4.0f, 2.0f};
    float b[] = {2.0f, 2.0f, 0.5f, 10.0f};
    float expected[] = {4.0f, 9.0f, 2.0f, 1024.0f};
    int n = 4;

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornPow_f32(d_out, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Scalar Operations
// -----------------------------------------------------------------------------

TEST(add_scalar) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 10.0f;
    float expected[] = {11.0f, 12.0f, 13.0f, 14.0f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornAddScalar_f32(d_out, d_in, scalar, n, nullptr));
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

TEST(mul_scalar) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.5f;
    float expected[] = {2.5f, 5.0f, 7.5f, 10.0f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornMulScalar_f32(d_out, d_in, scalar, n, nullptr));
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

TEST(pow_scalar) {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.0f;
    float expected[] = {1.0f, 4.0f, 9.0f, 16.0f};
    int n = 4;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornPowScalar_f32(d_out, d_in, scalar, n, nullptr));
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
// Clamp
// -----------------------------------------------------------------------------

TEST(clamp_basic) {
    float input[] = {-5.0f, 0.0f, 5.0f, 10.0f, 15.0f};
    float expected[] = {0.0f, 0.0f, 5.0f, 10.0f, 10.0f};
    int n = 5;

    float* d_in = to_device(input, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornClamp_f32(d_out, d_in, 0.0f, 10.0f, n, nullptr));
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
// Where
// -----------------------------------------------------------------------------

TEST(where_basic) {
    float cond[] = {1.0f, 0.0f, 1.0f, -1.0f, 0.5f};  // >0 selects a
    float a[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected[] = {10.0f, 2.0f, 30.0f, 4.0f, 50.0f};
    int n = 5;

    float* d_cond = to_device(cond, n);
    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornWhere_f32(d_out, d_cond, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_cond);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// Large array test
// -----------------------------------------------------------------------------

TEST(add_large) {
    int n = 100000;
    float* a = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);
    float* d_out = device_alloc(n);

    ASSERT_SUCCESS(popcornAdd_f32(d_out, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_out, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], (float)n, TOL);
    }

    free(a);
    free(b);
    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    PASS();
}

// -----------------------------------------------------------------------------
// In-place operation test
// -----------------------------------------------------------------------------

TEST(add_inplace) {
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {10.0f, 20.0f, 30.0f, 40.0f};
    float expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
    int n = 4;

    float* d_a = to_device(a, n);
    float* d_b = to_device(b, n);

    // out == a (in-place)
    ASSERT_SUCCESS(popcornAdd_f32(d_a, d_a, d_b, n, nullptr));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* result = to_host(d_a, n);
    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(result[i], expected[i], TOL);
    }

    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Binary Operations Tests\n");
    printf("=======================\n");

    RUN_TEST(add_basic);
    RUN_TEST(sub_basic);
    RUN_TEST(mul_basic);
    RUN_TEST(div_basic);
    RUN_TEST(pow_basic);
    RUN_TEST(add_scalar);
    RUN_TEST(mul_scalar);
    RUN_TEST(pow_scalar);
    RUN_TEST(clamp_basic);
    RUN_TEST(where_basic);
    RUN_TEST(add_large);
    RUN_TEST(add_inplace);

    return test_summary("Binary");
}
