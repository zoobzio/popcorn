#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Test counters
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

// Colors for output
#define RED     "\033[0;31m"
#define GREEN   "\033[0;32m"
#define YELLOW  "\033[0;33m"
#define RESET   "\033[0m"

// Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, RED "CUDA error at %s:%d: %s\n" RESET, \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Start a test
#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        tests_run++; \
        printf("  %-40s ", #name); \
        fflush(stdout); \
        test_##name(); \
    } \
    static void test_##name(void)

// Assert with tolerance for floating point
#define ASSERT_NEAR(actual, expected, tol) do { \
    float _a = (actual); \
    float _e = (expected); \
    float _t = (tol); \
    if (fabsf(_a - _e) > _t) { \
        printf(RED "FAIL\n" RESET); \
        fprintf(stderr, "    Expected: %f, Got: %f (diff: %e, tol: %e)\n", \
                _e, _a, fabsf(_a - _e), _t); \
        tests_failed++; \
        return; \
    } \
} while(0)

// Assert equality for integers
#define ASSERT_EQ(actual, expected) do { \
    if ((actual) != (expected)) { \
        printf(RED "FAIL\n" RESET); \
        fprintf(stderr, "    Expected: %d, Got: %d\n", (expected), (actual)); \
        tests_failed++; \
        return; \
    } \
} while(0)

// Assert popcorn status is success
#define ASSERT_SUCCESS(status) do { \
    popcornStatus_t _s = (status); \
    if (_s != POPCORN_SUCCESS) { \
        printf(RED "FAIL\n" RESET); \
        fprintf(stderr, "    popcorn error: %s\n", popcornGetErrorString(_s)); \
        tests_failed++; \
        return; \
    } \
} while(0)

// Mark test as passed (call at end of successful test)
#define PASS() do { \
    printf(GREEN "PASS\n" RESET); \
    tests_passed++; \
} while(0)

// Run a test by name
#define RUN_TEST(name) run_test_##name()

// Print summary and return exit code
static inline int test_summary(const char* suite_name) {
    printf("\n%s: ", suite_name);
    if (tests_failed == 0) {
        printf(GREEN "%d/%d tests passed\n" RESET, tests_passed, tests_run);
        return 0;
    } else {
        printf(RED "%d/%d tests passed (%d failed)\n" RESET,
               tests_passed, tests_run, tests_failed);
        return 1;
    }
}

// Helper: allocate and copy host array to device
static inline float* to_device(const float* host, int n) {
    float* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice));
    return dev;
}

// Helper: allocate device memory
static inline float* device_alloc(int n) {
    float* dev;
    CUDA_CHECK(cudaMalloc(&dev, n * sizeof(float)));
    return dev;
}

// Helper: copy device array to host (caller must free)
static inline float* to_host(const float* dev, int n) {
    float* host = (float*)malloc(n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
}

#endif // TEST_RUNNER_H
