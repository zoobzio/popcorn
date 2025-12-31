#include "test_runner.h"
#include "popcorn.h"

const float TOL = 1e-5f;

// Reference CPU implementation for verification
static void adamw_cpu(
    float* param, const float* grad, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1, float bc2, int n
) {
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        float p = param[i];

        float m_new = beta1 * m[i] + (1.0f - beta1) * g;
        float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
        m[i] = m_new;
        v[i] = v_new;

        float m_hat = m_new / bc1;
        float v_hat = v_new / bc2;

        param[i] = p - lr * (m_hat / (sqrtf(v_hat) + eps) + wd * p);
    }
}

// -----------------------------------------------------------------------------
// AdamW Basic
// -----------------------------------------------------------------------------

TEST(adamw_basic) {
    float param[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int n = 4;

    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.01f;
    int step = 1;
    float bc1 = 1.0f - powf(beta1, step);
    float bc2 = 1.0f - powf(beta2, step);

    // Compute expected values on CPU
    float exp_param[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float exp_m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float exp_v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    adamw_cpu(exp_param, grad, exp_m, exp_v, lr, beta1, beta2, eps, wd, bc1, bc2, n);

    // Run GPU kernel
    float* d_param = to_device(param, n);
    float* d_grad = to_device(grad, n);
    float* d_m = to_device(m, n);
    float* d_v = to_device(v, n);

    ASSERT_SUCCESS(popcornAdamW_f32(
        d_param, d_grad, d_m, d_v,
        lr, beta1, beta2, eps, wd, bc1, bc2,
        n, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* res_param = to_host(d_param, n);
    float* res_m = to_host(d_m, n);
    float* res_v = to_host(d_v, n);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(res_param[i], exp_param[i], TOL);
        ASSERT_NEAR(res_m[i], exp_m[i], TOL);
        ASSERT_NEAR(res_v[i], exp_v[i], TOL);
    }

    free(res_param);
    free(res_m);
    free(res_v);
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

// -----------------------------------------------------------------------------
// AdamW Multiple Steps
// -----------------------------------------------------------------------------

TEST(adamw_multi_step) {
    float param[] = {1.0f, -1.0f, 0.5f, -0.5f};
    float grad[] = {0.1f, -0.1f, 0.05f, -0.05f};
    float m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int n = 4;

    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.01f;

    // Expected values after 3 steps
    float exp_param[] = {1.0f, -1.0f, 0.5f, -0.5f};
    float exp_m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float exp_v[] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int step = 1; step <= 3; step++) {
        float bc1 = 1.0f - powf(beta1, step);
        float bc2 = 1.0f - powf(beta2, step);
        adamw_cpu(exp_param, grad, exp_m, exp_v, lr, beta1, beta2, eps, wd, bc1, bc2, n);
    }

    // Run GPU kernel for 3 steps
    float* d_param = to_device(param, n);
    float* d_grad = to_device(grad, n);
    float* d_m = to_device(m, n);
    float* d_v = to_device(v, n);

    for (int step = 1; step <= 3; step++) {
        float bc1 = 1.0f - powf(beta1, step);
        float bc2 = 1.0f - powf(beta2, step);
        ASSERT_SUCCESS(popcornAdamW_f32(
            d_param, d_grad, d_m, d_v,
            lr, beta1, beta2, eps, wd, bc1, bc2,
            n, nullptr
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float* res_param = to_host(d_param, n);
    float* res_m = to_host(d_m, n);
    float* res_v = to_host(d_v, n);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(res_param[i], exp_param[i], TOL);
        ASSERT_NEAR(res_m[i], exp_m[i], TOL);
        ASSERT_NEAR(res_v[i], exp_v[i], TOL);
    }

    free(res_param);
    free(res_m);
    free(res_v);
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

// -----------------------------------------------------------------------------
// AdamW Zero Gradient
// -----------------------------------------------------------------------------

TEST(adamw_zero_grad) {
    float param[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int n = 4;

    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.01f;
    float bc1 = 0.1f;  // 1 - 0.9^1
    float bc2 = 0.001f;  // 1 - 0.999^1

    // With zero gradients, only weight decay applies
    float exp_param[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float exp_m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float exp_v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    adamw_cpu(exp_param, grad, exp_m, exp_v, lr, beta1, beta2, eps, wd, bc1, bc2, n);

    float* d_param = to_device(param, n);
    float* d_grad = to_device(grad, n);
    float* d_m = to_device(m, n);
    float* d_v = to_device(v, n);

    ASSERT_SUCCESS(popcornAdamW_f32(
        d_param, d_grad, d_m, d_v,
        lr, beta1, beta2, eps, wd, bc1, bc2,
        n, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* res_param = to_host(d_param, n);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(res_param[i], exp_param[i], TOL);
    }

    free(res_param);
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

// -----------------------------------------------------------------------------
// AdamW No Weight Decay
// -----------------------------------------------------------------------------

TEST(adamw_no_wd) {
    float param[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float grad[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int n = 4;

    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.0f;  // No weight decay
    float bc1 = 0.1f;
    float bc2 = 0.001f;

    float exp_param[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float exp_m[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float exp_v[] = {0.0f, 0.0f, 0.0f, 0.0f};
    adamw_cpu(exp_param, grad, exp_m, exp_v, lr, beta1, beta2, eps, wd, bc1, bc2, n);

    float* d_param = to_device(param, n);
    float* d_grad = to_device(grad, n);
    float* d_m = to_device(m, n);
    float* d_v = to_device(v, n);

    ASSERT_SUCCESS(popcornAdamW_f32(
        d_param, d_grad, d_m, d_v,
        lr, beta1, beta2, eps, wd, bc1, bc2,
        n, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* res_param = to_host(d_param, n);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(res_param[i], exp_param[i], TOL);
    }

    free(res_param);
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

// -----------------------------------------------------------------------------
// AdamW Large Array
// -----------------------------------------------------------------------------

TEST(adamw_large) {
    int n = 100000;
    float* param = (float*)malloc(n * sizeof(float));
    float* grad = (float*)malloc(n * sizeof(float));
    float* m = (float*)malloc(n * sizeof(float));
    float* v = (float*)malloc(n * sizeof(float));
    float* exp_param = (float*)malloc(n * sizeof(float));
    float* exp_m = (float*)malloc(n * sizeof(float));
    float* exp_v = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        param[i] = (float)(i % 100) * 0.01f - 0.5f;
        grad[i] = (float)((i * 7) % 100) * 0.001f - 0.05f;
        m[i] = 0.0f;
        v[i] = 0.0f;
        exp_param[i] = param[i];
        exp_m[i] = 0.0f;
        exp_v[i] = 0.0f;
    }

    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float wd = 0.01f;
    float bc1 = 0.1f;
    float bc2 = 0.001f;

    adamw_cpu(exp_param, grad, exp_m, exp_v, lr, beta1, beta2, eps, wd, bc1, bc2, n);

    float* d_param = to_device(param, n);
    float* d_grad = to_device(grad, n);
    float* d_m = to_device(m, n);
    float* d_v = to_device(v, n);

    ASSERT_SUCCESS(popcornAdamW_f32(
        d_param, d_grad, d_m, d_v,
        lr, beta1, beta2, eps, wd, bc1, bc2,
        n, nullptr
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    float* res_param = to_host(d_param, n);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(res_param[i], exp_param[i], TOL);
    }

    free(param);
    free(grad);
    free(m);
    free(v);
    free(exp_param);
    free(exp_m);
    free(exp_v);
    free(res_param);
    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

// -----------------------------------------------------------------------------
// AdamW Invalid Input
// -----------------------------------------------------------------------------

TEST(adamw_null_param) {
    float grad[] = {0.1f};
    float m[] = {0.0f};
    float v[] = {0.0f};

    float* d_grad = to_device(grad, 1);
    float* d_m = to_device(m, 1);
    float* d_v = to_device(v, 1);

    popcornStatus_t status = popcornAdamW_f32(
        nullptr, d_grad, d_m, d_v,
        0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.1f, 0.001f,
        1, nullptr
    );

    ASSERT_EQ(status, POPCORN_ERROR_INVALID_VALUE);

    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

TEST(adamw_zero_bc) {
    float param[] = {1.0f};
    float grad[] = {0.1f};
    float m[] = {0.0f};
    float v[] = {0.0f};

    float* d_param = to_device(param, 1);
    float* d_grad = to_device(grad, 1);
    float* d_m = to_device(m, 1);
    float* d_v = to_device(v, 1);

    // Zero bias correction should be rejected
    popcornStatus_t status = popcornAdamW_f32(
        d_param, d_grad, d_m, d_v,
        0.001f, 0.9f, 0.999f, 1e-8f, 0.01f, 0.0f, 0.001f,
        1, nullptr
    );

    ASSERT_EQ(status, POPCORN_ERROR_INVALID_VALUE);

    cudaFree(d_param);
    cudaFree(d_grad);
    cudaFree(d_m);
    cudaFree(d_v);
    PASS();
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main() {
    printf("Optimizer Tests\n");
    printf("===============\n");

    RUN_TEST(adamw_basic);
    RUN_TEST(adamw_multi_step);
    RUN_TEST(adamw_zero_grad);
    RUN_TEST(adamw_no_wd);
    RUN_TEST(adamw_large);
    RUN_TEST(adamw_null_param);
    RUN_TEST(adamw_zero_bc);

    return test_summary("Optimizer");
}
