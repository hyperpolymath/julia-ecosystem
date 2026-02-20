// SPDX-License-Identifier: PMPL-1.0-or-later
//! Threading support for Axiom.jl Zig backend
//!
//! Provides parallel dispatch for element-wise operations on large arrays.
//! Below THREAD_THRESHOLD, operations run single-threaded with SIMD.
//! Above THREAD_THRESHOLD, work is split across CPU threads.

const std = @import("std");
const activations = @import("activations.zig");

/// Minimum elements before spawning threads (64K)
pub const THREAD_THRESHOLD: usize = 65536;

/// Maximum worker threads (main thread also does work)
const MAX_WORKERS: usize = 3; // + main thread = 4 total

// ============================================================================
// Thread context types for each activation signature
// ============================================================================

const UnaryCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    start: usize,
    len: usize,
};

const UnaryWithAlphaCtx = struct {
    input: [*]const f32,
    output: [*]f32,
    start: usize,
    len: usize,
    alpha: f32,
};

// ============================================================================
// Worker functions (called by spawned threads)
// ============================================================================

fn worker_sigmoid(ctx: UnaryCtx) void {
    activations.sigmoid(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_gelu(ctx: UnaryCtx) void {
    activations.gelu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_tanh(ctx: UnaryCtx) void {
    activations.tanh_activation(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_relu(ctx: UnaryCtx) void {
    activations.relu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_swish(ctx: UnaryCtx) void {
    activations.swish(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_mish(ctx: UnaryCtx) void {
    activations.mish(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_selu(ctx: UnaryCtx) void {
    activations.selu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_hard_sigmoid(ctx: UnaryCtx) void {
    activations.hard_sigmoid(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_hard_swish(ctx: UnaryCtx) void {
    activations.hard_swish(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_softplus(ctx: UnaryCtx) void {
    activations.softplus(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len]);
}

fn worker_leaky_relu(ctx: UnaryWithAlphaCtx) void {
    activations.leaky_relu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len], ctx.alpha);
}

fn worker_elu(ctx: UnaryWithAlphaCtx) void {
    activations.elu(ctx.input[ctx.start..][0..ctx.len], ctx.output[ctx.start..][0..ctx.len], ctx.alpha);
}

// ============================================================================
// Parallel dispatch functions (called from axiom.zig FFI exports)
// ============================================================================

/// Parallel sigmoid for large arrays
pub fn parallel_sigmoid(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.sigmoid(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_sigmoid, input, output, n);
}

/// Parallel GELU for large arrays
pub fn parallel_gelu(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.gelu(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_gelu, input, output, n);
}

/// Parallel tanh for large arrays
pub fn parallel_tanh(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.tanh_activation(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_tanh, input, output, n);
}

/// Parallel relu for large arrays
pub fn parallel_relu(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.relu(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_relu, input, output, n);
}

/// Parallel swish for large arrays
pub fn parallel_swish(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.swish(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_swish, input, output, n);
}

/// Parallel mish for large arrays
pub fn parallel_mish(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.mish(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_mish, input, output, n);
}

/// Parallel selu for large arrays
pub fn parallel_selu(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.selu(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_selu, input, output, n);
}

/// Parallel hard_sigmoid for large arrays
pub fn parallel_hard_sigmoid(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.hard_sigmoid(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_hard_sigmoid, input, output, n);
}

/// Parallel hard_swish for large arrays
pub fn parallel_hard_swish(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.hard_swish(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_hard_swish, input, output, n);
}

/// Parallel softplus for large arrays
pub fn parallel_softplus(input: [*]const f32, output: [*]f32, n: usize) void {
    if (n < THREAD_THRESHOLD) {
        activations.softplus(input[0..n], output[0..n]);
        return;
    }
    parallel_unary(worker_softplus, input, output, n);
}

/// Parallel leaky_relu for large arrays
pub fn parallel_leaky_relu(input: [*]const f32, output: [*]f32, n: usize, alpha: f32) void {
    if (n < THREAD_THRESHOLD) {
        activations.leaky_relu(input[0..n], output[0..n], alpha);
        return;
    }
    parallel_unary_alpha(worker_leaky_relu, input, output, n, alpha);
}

/// Parallel elu for large arrays
pub fn parallel_elu(input: [*]const f32, output: [*]f32, n: usize, alpha: f32) void {
    if (n < THREAD_THRESHOLD) {
        activations.elu(input[0..n], output[0..n], alpha);
        return;
    }
    parallel_unary_alpha(worker_elu, input, output, n, alpha);
}

// ============================================================================
// Internal: generic parallel dispatch
// ============================================================================

fn parallel_unary(
    comptime worker_fn: fn (UnaryCtx) void,
    input: [*]const f32,
    output: [*]f32,
    n: usize,
) void {
    const num_threads = @min(MAX_WORKERS + 1, (n + THREAD_THRESHOLD - 1) / THREAD_THRESHOLD);
    const chunk = (n + num_threads - 1) / num_threads;

    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    // Spawn worker threads for all but last chunk
    for (0..num_threads - 1) |t| {
        const start = t * chunk;
        const end = @min(start + chunk, n);
        handles[t] = std.Thread.spawn(.{}, worker_fn, .{UnaryCtx{
            .input = input,
            .output = output,
            .start = start,
            .len = end - start,
        }}) catch null;
    }

    // Main thread does last chunk
    const last_start = (num_threads - 1) * chunk;
    const last_len = n - last_start;
    worker_fn(UnaryCtx{
        .input = input,
        .output = output,
        .start = last_start,
        .len = last_len,
    });

    // Join workers
    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}

fn parallel_unary_alpha(
    comptime worker_fn: fn (UnaryWithAlphaCtx) void,
    input: [*]const f32,
    output: [*]f32,
    n: usize,
    alpha: f32,
) void {
    const num_threads = @min(MAX_WORKERS + 1, (n + THREAD_THRESHOLD - 1) / THREAD_THRESHOLD);
    const chunk = (n + num_threads - 1) / num_threads;

    var handles: [MAX_WORKERS]?std.Thread = .{null} ** MAX_WORKERS;

    for (0..num_threads - 1) |t| {
        const start = t * chunk;
        const end = @min(start + chunk, n);
        handles[t] = std.Thread.spawn(.{}, worker_fn, .{UnaryWithAlphaCtx{
            .input = input,
            .output = output,
            .start = start,
            .len = end - start,
            .alpha = alpha,
        }}) catch null;
    }

    const last_start = (num_threads - 1) * chunk;
    const last_len = n - last_start;
    worker_fn(UnaryWithAlphaCtx{
        .input = input,
        .output = output,
        .start = last_start,
        .len = last_len,
        .alpha = alpha,
    });

    for (&handles) |*maybe_handle| {
        if (maybe_handle.*) |handle| {
            handle.join();
            maybe_handle.* = null;
        }
    }
}
