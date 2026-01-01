#ifndef POPCORN_H
#define POPCORN_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
typedef enum {
    POPCORN_SUCCESS = 0,
    POPCORN_ERROR_INVALID_VALUE = 1,
    POPCORN_ERROR_CUDA = 2,
} popcornStatus_t;

// Get error string for status code
const char* popcornGetErrorString(popcornStatus_t status);

// Get the underlying CUDA error from the last POPCORN_ERROR_CUDA return
// Returns cudaSuccess if no CUDA error has occurred
cudaError_t popcornGetLastCudaError(void);

// Get the CUDA error string for the last CUDA error
// More detailed than popcornGetErrorString for CUDA errors
const char* popcornGetLastCudaErrorString(void);

// -----------------------------------------------------------------------------
// Unary Elementwise Operations
// All operations: out[i] = f(in[i]) for i in [0, n)
// -----------------------------------------------------------------------------

popcornStatus_t popcornNeg_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornAbs_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornExp_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLog_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSqrt_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSquare_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSign_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSin_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornCos_f32(float* out, const float* in, int64_t n, cudaStream_t stream);

// Activations not covered by cuDNN
popcornStatus_t popcornGelu_f32(float* out, const float* in, int64_t n, cudaStream_t stream);
popcornStatus_t popcornLeakyRelu_f32(float* out, const float* in, float alpha, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSilu_f32(float* out, const float* in, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Binary Elementwise Operations
// All operations: out[i] = f(a[i], b[i]) for i in [0, n)
// -----------------------------------------------------------------------------

popcornStatus_t popcornAdd_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSub_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornMul_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornDiv_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);
popcornStatus_t popcornPow_f32(float* out, const float* a, const float* b, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Binary Scalar Operations
// All operations: out[i] = f(in[i], scalar) for i in [0, n)
// -----------------------------------------------------------------------------

popcornStatus_t popcornAddScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornSubScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornMulScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornDivScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);
popcornStatus_t popcornPowScalar_f32(float* out, const float* in, float scalar, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Comparison / Selection Operations
// -----------------------------------------------------------------------------

popcornStatus_t popcornClamp_f32(float* out, const float* in, float minVal, float maxVal, int64_t n, cudaStream_t stream);
popcornStatus_t popcornWhere_f32(float* out, const float* cond, const float* a, const float* b, int64_t n, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Gather / Index Operations
// -----------------------------------------------------------------------------

// Gathers values from input at indices specified by idx
// out[i] = in[i * stride + idx[i]]
// Used for selecting class probabilities at target indices (NLLLoss/CrossEntropyLoss)
popcornStatus_t popcornGather_f32(
    float* out,           // [n] output values
    const float* in,      // [n, classes] input tensor (row-major)
    const int64_t* idx,   // [n] indices (0 to stride-1)
    int64_t n,            // batch size
    int64_t stride,       // number of classes (inner dimension)
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Reduction Operations
// -----------------------------------------------------------------------------

// Returns indices of max values along last dimension
// out[i] = argmax(in[i*stride : i*stride+stride])
popcornStatus_t popcornArgMax_f32(
    int64_t* out,         // [n] output indices
    const float* in,      // [n, dim] input tensor (row-major)
    int64_t n,            // number of rows
    int64_t stride,       // size of dimension to reduce
    cudaStream_t stream
);

// Returns indices of min values along last dimension
// out[i] = argmin(in[i*stride : i*stride+stride])
popcornStatus_t popcornArgMin_f32(
    int64_t* out,         // [n] output indices
    const float* in,      // [n, dim] input tensor (row-major)
    int64_t n,            // number of rows
    int64_t stride,       // size of dimension to reduce
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Normalization Operations
// -----------------------------------------------------------------------------

// Applies layer normalization: out = (in - mean) / sqrt(var + eps) * weight + bias
// Normalizes over last `norm_size` elements
popcornStatus_t popcornLayerNorm_f32(
    float* out,           // [n, norm_size] output
    const float* in,      // [n, norm_size] input
    const float* weight,  // [norm_size] scale (gamma), nullable for no scaling
    const float* bias,    // [norm_size] shift (beta), nullable for no shift
    int64_t n,            // batch size (product of all dims except normalized)
    int64_t norm_size,    // size of normalized dimension(s)
    float eps,            // epsilon for numerical stability (typically 1e-5)
    cudaStream_t stream
);

// Layer normalization with statistics output for backward pass
// Same as popcornLayerNorm_f32 but optionally outputs mean and invstd
popcornStatus_t popcornLayerNormWithStats_f32(
    float* out,           // [n, norm_size] output
    float* out_mean,      // [n] mean per row (nullable to skip)
    float* out_invstd,    // [n] inverse std per row (nullable to skip)
    const float* in,      // [n, norm_size] input
    const float* weight,  // [norm_size] scale (gamma), nullable for no scaling
    const float* bias,    // [norm_size] shift (beta), nullable for no shift
    int64_t n,            // batch size
    int64_t norm_size,    // size of normalized dimension(s)
    float eps,            // epsilon for numerical stability
    cudaStream_t stream
);

// Applies RMS normalization: out = in / rms(in) * weight
// where rms(in) = sqrt(mean(in^2) + eps)
popcornStatus_t popcornRMSNorm_f32(
    float* out,           // [n, norm_size] output
    const float* in,      // [n, norm_size] input
    const float* weight,  // [norm_size] scale, nullable for no scaling
    int64_t n,            // batch size
    int64_t norm_size,    // size of normalized dimension(s)
    float eps,            // epsilon for numerical stability (typically 1e-5)
    cudaStream_t stream
);

// RMS normalization with statistics output for backward pass
// Same as popcornRMSNorm_f32 but optionally outputs rrms (1/rms)
popcornStatus_t popcornRMSNormWithStats_f32(
    float* out,           // [n, norm_size] output
    float* out_rrms,      // [n] reciprocal RMS per row (nullable to skip)
    const float* in,      // [n, norm_size] input
    const float* weight,  // [norm_size] scale, nullable for no scaling
    int64_t n,            // batch size
    int64_t norm_size,    // size of normalized dimension(s)
    float eps,            // epsilon for numerical stability
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Tensor Operations
// -----------------------------------------------------------------------------

// Embedding lookup: out[i] = weight[indices[i]]
// Retrieves embedding vectors for given token indices
popcornStatus_t popcornEmbedding_f32(
    float* out,               // [n, embed_dim] output embeddings
    const float* weight,      // [vocab_size, embed_dim] embedding table
    const int64_t* indices,   // [n] token indices (must be in range [0, vocab_size))
    int64_t n,                // number of tokens
    int64_t embed_dim,        // embedding dimension
    int64_t vocab_size,       // vocabulary size (for bounds checking in debug builds)
    cudaStream_t stream
);

// Concatenate tensors along an existing dimension
// Layout is [outer_size, cat_dim, inner_size] where cat_dim varies per input
popcornStatus_t popcornCat_f32(
    float* out,                   // output buffer
    const float* const* inputs,   // array of input tensor pointers
    int64_t num_inputs,           // number of tensors to concatenate
    const int64_t* sizes,         // [num_inputs] size along cat dim for each input
    int64_t outer_size,           // product of dims before cat dim
    int64_t inner_size,           // product of dims after cat dim
    cudaStream_t stream
);

// Stack tensors along a new first dimension
// out[i, ...] = inputs[i][...]
popcornStatus_t popcornStack_f32(
    float* out,                   // [num_inputs, tensor_size] output
    const float* const* inputs,   // array of input tensor pointers
    int64_t num_inputs,           // number of tensors to stack
    int64_t tensor_size,          // elements per input tensor
    cudaStream_t stream
);

// Lower triangular mask: zeros out elements above diagonal + k
// out[row, col] = in[row, col] if col <= row + k, else 0
// k=0: main diagonal, k<0: below main, k>0: above main
popcornStatus_t popcornTril_f32(
    float* out,                   // [rows, cols] output
    const float* in,              // [rows, cols] input
    int64_t rows,
    int64_t cols,
    int64_t k,                    // diagonal offset
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Backward Pass Operations (for autograd)
// -----------------------------------------------------------------------------

// GELU backward: grad_in = grad_out * gelu'(in)
popcornStatus_t popcornGeluBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    int64_t n,
    cudaStream_t stream
);

// LeakyReLU backward: grad_in = grad_out * (in > 0 ? 1 : alpha)
popcornStatus_t popcornLeakyReluBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    float alpha,
    int64_t n,
    cudaStream_t stream
);

// LayerNorm backward: computes grad_input, grad_weight, grad_bias
// Requires saved mean and inverse std from forward pass
popcornStatus_t popcornLayerNormBackward_f32(
    float* grad_in,           // [n, norm_size] output gradient for input
    float* grad_weight,       // [norm_size] output gradient for weight (nullable)
    float* grad_bias,         // [norm_size] output gradient for bias (nullable)
    const float* grad_out,    // [n, norm_size] incoming gradient
    const float* in,          // [n, norm_size] saved input from forward
    const float* mean,        // [n] saved mean from forward
    const float* invstd,      // [n] saved inverse std from forward
    const float* weight,      // [norm_size] weight (nullable)
    int64_t n,                // batch size
    int64_t norm_size,        // normalization dimension size
    cudaStream_t stream
);

// Embedding backward: accumulates gradients into embedding table
// grad_weight[indices[i]] += grad_out[i]
popcornStatus_t popcornEmbeddingBackward_f32(
    float* grad_weight,       // [vocab_size, embed_dim] output gradient (zeroed then accumulated)
    const float* grad_out,    // [n, embed_dim] incoming gradient
    const int64_t* indices,   // [n] token indices from forward
    int64_t n,                // number of tokens
    int64_t embed_dim,        // embedding dimension
    int64_t vocab_size,       // vocabulary size
    cudaStream_t stream
);

// ReLU backward: grad_in = grad_out * (in > 0 ? 1 : 0)
popcornStatus_t popcornReluBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    int64_t n,
    cudaStream_t stream
);

// Sigmoid backward: grad_in = grad_out * out * (1 - out)
popcornStatus_t popcornSigmoidBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* out,         // [n] sigmoid output from forward
    int64_t n,
    cudaStream_t stream
);

// Tanh backward: grad_in = grad_out * (1 - out^2)
popcornStatus_t popcornTanhBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* out,         // [n] tanh output from forward
    int64_t n,
    cudaStream_t stream
);

// SiLU backward: grad_in = grad_out * (sigmoid(in) + in * sigmoid(in) * (1 - sigmoid(in)))
popcornStatus_t popcornSiluBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    int64_t n,
    cudaStream_t stream
);

// Softmax backward: grad_in = out * (grad_out - sum(grad_out * out))
popcornStatus_t popcornSoftmaxBackward_f32(
    float* grad_in,           // [batch, dim] output gradient
    const float* grad_out,    // [batch, dim] incoming gradient
    const float* out,         // [batch, dim] softmax output from forward
    int64_t batch,            // batch size
    int64_t dim,              // softmax dimension
    cudaStream_t stream
);

// CrossEntropy backward (fused softmax + NLL): grad_in = scale * (softmax - one_hot(target))
popcornStatus_t popcornCrossEntropyBackward_f32(
    float* grad_in,           // [batch, classes] output gradient w.r.t. logits
    const float* softmax,     // [batch, classes] softmax output from forward
    const int64_t* targets,   // [batch] target class indices
    int64_t batch,            // batch size
    int64_t classes,          // number of classes
    float scale,              // 1/batch for mean reduction, 1 for sum
    cudaStream_t stream
);

// Exp backward: grad_in = grad_out * out (where out = exp(in))
popcornStatus_t popcornExpBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* out,         // [n] exp output from forward
    int64_t n,
    cudaStream_t stream
);

// Log backward: grad_in = grad_out / in
popcornStatus_t popcornLogBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    int64_t n,
    cudaStream_t stream
);

// Sqrt backward: grad_in = grad_out / (2 * out) (where out = sqrt(in))
popcornStatus_t popcornSqrtBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* out,         // [n] sqrt output from forward
    int64_t n,
    cudaStream_t stream
);

// Sin backward: grad_in = grad_out * cos(in)
popcornStatus_t popcornSinBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    int64_t n,
    cudaStream_t stream
);

// Cos backward: grad_in = grad_out * -sin(in)
popcornStatus_t popcornCosBackward_f32(
    float* grad_in,           // [n] output gradient
    const float* grad_out,    // [n] incoming gradient
    const float* in,          // [n] saved input from forward
    int64_t n,
    cudaStream_t stream
);

// RMSNorm backward: computes grad_input and grad_weight
// Requires saved rrms (1/rms) from forward pass
popcornStatus_t popcornRMSNormBackward_f32(
    float* grad_in,           // [n, norm_size] output gradient for input
    float* grad_weight,       // [norm_size] output gradient for weight (nullable)
    const float* grad_out,    // [n, norm_size] incoming gradient
    const float* in,          // [n, norm_size] saved input from forward
    const float* rrms,        // [n] saved 1/rms from forward
    const float* weight,      // [norm_size] weight (nullable)
    int64_t n,                // batch size
    int64_t norm_size,        // normalization dimension size
    cudaStream_t stream
);

// Scatter: write values to indexed positions
// out[i * stride + idx[i]] = in[i]
popcornStatus_t popcornScatter_f32(
    float* out,               // [n, stride] output tensor
    const float* in,          // [n] input values
    const int64_t* idx,       // [n] indices
    int64_t n,                // number of elements
    int64_t stride,           // inner dimension size
    cudaStream_t stream
);

// ScatterAdd: accumulate values at indexed positions (for Gather backward)
// out[i * stride + idx[i]] += in[i]
popcornStatus_t popcornScatterAdd_f32(
    float* out,               // [n, stride] output tensor (accumulated into)
    const float* in,          // [n] input values
    const int64_t* idx,       // [n] indices
    int64_t n,                // number of elements
    int64_t stride,           // inner dimension size
    cudaStream_t stream
);

// Split: inverse of Cat, splits tensor into multiple outputs
popcornStatus_t popcornSplit_f32(
    float* const* outputs,    // array of output tensor pointers
    int64_t num_outputs,      // number of output tensors
    const int64_t* sizes,     // [num_outputs] size along split dim for each output
    const float* in,          // input tensor
    int64_t outer_size,       // product of dims before split dim
    int64_t inner_size,       // product of dims after split dim
    cudaStream_t stream
);

// Unstack: inverse of Stack, splits into individual tensors
popcornStatus_t popcornUnstack_f32(
    float* const* outputs,    // array of output tensor pointers
    const float* in,          // [num_outputs, tensor_size] input
    int64_t num_outputs,      // number of output tensors
    int64_t tensor_size,      // elements per output tensor
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Optimizer Operations
// -----------------------------------------------------------------------------

// Fused AdamW optimizer step
// Updates param, m, v in-place: param -= lr * (m_hat / (sqrt(v_hat) + eps) + wd * param)
// Caller computes bias corrections: bc1 = 1 - beta1^t, bc2 = 1 - beta2^t
popcornStatus_t popcornAdamW_f32(
    float* param,             // [n] parameter tensor (updated in-place)
    const float* grad,        // [n] gradient tensor
    float* m,                 // [n] first moment (updated in-place)
    float* v,                 // [n] second moment (updated in-place)
    float lr,                 // learning rate
    float beta1,              // first moment decay (typically 0.9)
    float beta2,              // second moment decay (typically 0.999)
    float epsilon,            // numerical stability (typically 1e-8)
    float weight_decay,       // L2 penalty (typically 0.01)
    float bias_correction1,   // 1 - beta1^t (precomputed by caller)
    float bias_correction2,   // 1 - beta2^t (precomputed by caller)
    int64_t n,                // number of elements
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif // POPCORN_H
