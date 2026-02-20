// SPDX-License-Identifier: PMPL-1.0-or-later
//! Activation Functions
//!
//! SIMD-optimized activation functions for neural networks.

const std = @import("std");
const math = std.math;

const VEC_SIZE = 8;
const Vec = @Vector(VEC_SIZE, f32);

// ============================================================================
// ReLU Family
// ============================================================================

/// ReLU: max(0, x)
pub fn relu(input: []const f32, output: []f32) void {
    const zero: Vec = @splat(0);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x: Vec = input[i..][0..VEC_SIZE].*;
        output[i..][0..VEC_SIZE].* = @max(zero, x);
    }

    while (i < n) : (i += 1) {
        output[i] = @max(0, input[i]);
    }
}

/// ReLU in-place
pub fn relu_inplace(data: []f32) void {
    const zero: Vec = @splat(0);
    const n = data.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x: Vec = data[i..][0..VEC_SIZE].*;
        data[i..][0..VEC_SIZE].* = @max(zero, x);
    }

    while (i < n) : (i += 1) {
        data[i] = @max(0, data[i]);
    }
}

/// Leaky ReLU: x if x > 0, else alpha * x
pub fn leaky_relu(input: []const f32, output: []f32, alpha: f32) void {
    const alpha_vec: Vec = @splat(alpha);
    const zero: Vec = @splat(0);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x: Vec = input[i..][0..VEC_SIZE].*;
        const mask = x > zero;
        output[i..][0..VEC_SIZE].* = @select(f32, mask, x, alpha_vec * x);
    }

    while (i < n) : (i += 1) {
        const x = input[i];
        output[i] = if (x > 0) x else alpha * x;
    }
}

/// ReLU6: min(max(0, x), 6)
pub fn relu6(input: []const f32, output: []f32) void {
    const zero: Vec = @splat(0);
    const six: Vec = @splat(6);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x: Vec = input[i..][0..VEC_SIZE].*;
        output[i..][0..VEC_SIZE].* = @min(six, @max(zero, x));
    }

    while (i < n) : (i += 1) {
        output[i] = @min(6, @max(0, input[i]));
    }
}

// ============================================================================
// GELU (Gaussian Error Linear Unit)
// ============================================================================

/// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// SIMD-vectorized using tanh(z) = 1 - 2/(exp(2z) + 1)
pub fn gelu(input: []const f32, output: []f32) void {
    const sqrt_2_over_pi_vec: Vec = @splat(0.7978845608028654);
    const coeff_vec: Vec = @splat(0.044715);
    const half_vec: Vec = @splat(0.5);
    const one_vec: Vec = @splat(1.0);
    const two_vec: Vec = @splat(2.0);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x_vec: Vec = input[i..][0..VEC_SIZE].*;
        const x3 = x_vec * x_vec * x_vec;
        const inner = sqrt_2_over_pi_vec * (x_vec + coeff_vec * x3);
        // tanh(z) = 1 - 2/(exp(2z) + 1)
        const exp_2inner = @exp(two_vec * inner);
        const tanh_val = one_vec - two_vec / (exp_2inner + one_vec);
        output[i..][0..VEC_SIZE].* = half_vec * x_vec * (one_vec + tanh_val);
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const x = input[i];
        const x3 = x * x * x;
        const inner = 0.7978845608028654 * (x + 0.044715 * x3);
        const exp_2i = @exp(2.0 * inner);
        output[i] = 0.5 * x * (1.0 + (1.0 - 2.0 / (exp_2i + 1.0)));
    }
}

/// Fast GELU approximation using sigmoid: x * sigmoid(1.702 * x)
/// SIMD-vectorized.
pub fn gelu_fast(input: []const f32, output: []f32) void {
    const coeff_vec: Vec = @splat(1.702);
    const one_vec: Vec = @splat(1.0);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x_vec: Vec = input[i..][0..VEC_SIZE].*;
        const neg_cx = @as(Vec, @splat(@as(f32, 0))) - coeff_vec * x_vec;
        const sig = one_vec / (one_vec + @exp(neg_cx));
        output[i..][0..VEC_SIZE].* = x_vec * sig;
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const x = input[i];
        const sig = 1.0 / (1.0 + @exp(-1.702 * x));
        output[i] = x * sig;
    }
}

// ============================================================================
// Sigmoid & Tanh
// ============================================================================

/// Sigmoid: 1 / (1 + exp(-x))
/// SIMD-vectorized.
pub fn sigmoid(input: []const f32, output: []f32) void {
    const one_vec: Vec = @splat(1.0);
    const zero_vec: Vec = @splat(0.0);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x_vec: Vec = input[i..][0..VEC_SIZE].*;
        const neg_x = zero_vec - x_vec;
        output[i..][0..VEC_SIZE].* = one_vec / (one_vec + @exp(neg_x));
    }

    while (i < n) : (i += 1) {
        output[i] = 1.0 / (1.0 + @exp(-input[i]));
    }
}

/// Tanh: (exp(2x) - 1) / (exp(2x) + 1)
/// SIMD-vectorized using exp identity.
pub fn tanh_activation(input: []const f32, output: []f32) void {
    const one_vec: Vec = @splat(1.0);
    const two_vec: Vec = @splat(2.0);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x_vec: Vec = input[i..][0..VEC_SIZE].*;
        const exp_2x = @exp(two_vec * x_vec);
        output[i..][0..VEC_SIZE].* = (exp_2x - one_vec) / (exp_2x + one_vec);
    }

    while (i < n) : (i += 1) {
        const exp_2x = @exp(2.0 * input[i]);
        output[i] = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
}

/// Hard sigmoid: clip((x + 3) / 6, 0, 1)
pub fn hard_sigmoid(input: []const f32, output: []f32) void {
    const zero: Vec = @splat(0);
    const one: Vec = @splat(1);
    const three: Vec = @splat(3);
    const six: Vec = @splat(6);
    const n = input.len;

    var i: usize = 0;
    while (i + VEC_SIZE <= n) : (i += VEC_SIZE) {
        const x: Vec = input[i..][0..VEC_SIZE].*;
        const result = @min(one, @max(zero, (x + three) / six));
        output[i..][0..VEC_SIZE].* = result;
    }

    while (i < n) : (i += 1) {
        output[i] = @min(1, @max(0, (input[i] + 3) / 6));
    }
}

// ============================================================================
// Softmax
// ============================================================================

/// Softmax for a single vector
pub fn softmax(input: []const f32, output: []f32) void {
    const n = input.len;

    // Find max for numerical stability
    var max_val: f32 = input[0];
    for (input[1..]) |x| {
        if (x > max_val) max_val = x;
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0;
    for (input, 0..) |x, i| {
        const exp_val = @exp(x - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    // Normalize
    const inv_sum = 1.0 / sum;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        output[i] *= inv_sum;
    }
}

/// Softmax for batched input
pub fn softmax_batched(
    input_ptr: [*]const f32,
    output_ptr: [*]f32,
    batch_size: usize,
    num_classes: usize,
) void {
    var b: usize = 0;
    while (b < batch_size) : (b += 1) {
        const offset = b * num_classes;
        const input = input_ptr[offset..][0..num_classes];
        const output = output_ptr[offset..][0..num_classes];

        softmax(input, output);
    }
}

/// Log softmax (more numerically stable)
pub fn log_softmax(input: []const f32, output: []f32) void {
    var max_val: f32 = input[0];
    for (input[1..]) |x| {
        if (x > max_val) max_val = x;
    }

    // Compute log(sum(exp(x - max)))
    var sum: f32 = 0;
    for (input) |x| {
        sum += @exp(x - max_val);
    }
    const log_sum_exp = @log(sum) + max_val;

    // Output
    for (input, 0..) |x, i| {
        output[i] = x - log_sum_exp;
    }
}

// ============================================================================
// Swish / SiLU / Mish
// ============================================================================

/// Swish: x * sigmoid(x)
pub fn swish(input: []const f32, output: []f32) void {
    for (input, 0..) |x, i| {
        output[i] = x / (1.0 + @exp(-x));
    }
}

/// Hard swish: x * relu6(x + 3) / 6
pub fn hard_swish(input: []const f32, output: []f32) void {
    for (input, 0..) |x, i| {
        const relu6_val = @min(@as(f32, 6), @max(@as(f32, 0), x + 3));
        output[i] = x * relu6_val / 6.0;
    }
}

/// Mish: x * tanh(softplus(x))
pub fn mish(input: []const f32, output: []f32) void {
    for (input, 0..) |x, i| {
        const sp = if (x > 20.0) x else @log(1.0 + @exp(x));
        output[i] = x * math.tanh(sp);
    }
}

// ============================================================================
// ELU Family
// ============================================================================

/// ELU: x if x > 0, else alpha * (exp(x) - 1)
pub fn elu(input: []const f32, output: []f32, alpha: f32) void {
    for (input, 0..) |x, i| {
        output[i] = if (x > 0) x else alpha * (@exp(x) - 1);
    }
}

/// SELU (Scaled ELU)
pub fn selu(input: []const f32, output: []f32) void {
    const alpha: f32 = 1.6732632423543772;
    const scale: f32 = 1.0507009873554805;

    for (input, 0..) |x, i| {
        const elu_val = if (x > 0) x else alpha * (@exp(x) - 1);
        output[i] = scale * elu_val;
    }
}

// ============================================================================
// Softplus
// ============================================================================

/// Softplus: log(1 + exp(x))
pub fn softplus(input: []const f32, output: []f32) void {
    for (input, 0..) |x, i| {
        // Numerically stable version
        output[i] = if (x > 20.0)
            x
        else if (x < -20.0)
            @exp(x)
        else
            @log(1.0 + @exp(x));
    }
}
