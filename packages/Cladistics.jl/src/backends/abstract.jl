# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <jonathan.jewell@open.ac.uk>
#
# Cladistics.jl Backend Abstraction
# Coprocessor dispatch infrastructure adapted from Axiom.jl.

using Dates

# ============================================================================
# Backend Type Hierarchy
# ============================================================================

abstract type AbstractBackend end

struct JuliaBackend <: AbstractBackend end
struct RustBackend <: AbstractBackend
    lib_path::String
end
struct ZigBackend <: AbstractBackend
    lib_path::String
end
struct CUDABackend <: AbstractBackend
    device::Int
end
struct ROCmBackend <: AbstractBackend
    device::Int
end
struct MetalBackend <: AbstractBackend
    device::Int
end
struct TPUBackend <: AbstractBackend
    device::Int
end
struct NPUBackend <: AbstractBackend
    device::Int
end
struct DSPBackend <: AbstractBackend
    device::Int
end
struct PPUBackend <: AbstractBackend
    device::Int
end
struct MathBackend <: AbstractBackend
    device::Int
end
struct FPGABackend <: AbstractBackend
    device::Int
end
struct VPUBackend <: AbstractBackend
    device::Int
end
struct QPUBackend <: AbstractBackend
    device::Int
end
struct CryptoBackend <: AbstractBackend
    device::Int
end

# ============================================================================
# Backend Selection
# ============================================================================

const _current_backend = Ref{AbstractBackend}(JuliaBackend())
current_backend() = _current_backend[]
function set_backend!(backend::AbstractBackend)
    _current_backend[] = backend
    backend
end

macro with_backend(backend_expr, body)
    quote
        old = current_backend()
        set_backend!($(esc(backend_expr)))
        try
            $(esc(body))
        finally
            set_backend!(old)
        end
    end
end

# ============================================================================
# Environment Helpers
# ============================================================================

function _backend_env_available(key::String)
    raw = lowercase(strip(get(ENV, key, "")))
    isempty(raw) && return nothing
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    nothing
end

function _backend_env_count(available_key::String, count_key::String)
    forced = _backend_env_available(available_key)
    forced === false && return 0
    raw = strip(get(ENV, count_key, ""))
    isempty(raw) && return nothing
    parsed = tryparse(Int, raw)
    parsed === nothing ? nothing : max(parsed, 0)
end

_accelerator_env_flag(key) = let f = _backend_env_available(key); f === nothing ? false : f end
function _accelerator_env_count(avail_key, count_key)
    c = _backend_env_count(avail_key, count_key)
    c !== nothing && return c
    _accelerator_env_flag(avail_key) || return 0
    1
end

# ============================================================================
# Accelerator Availability
# ============================================================================

cuda_available()   = _accelerator_env_flag("AXIOM_CUDA_AVAILABLE")
rocm_available()   = _accelerator_env_flag("AXIOM_ROCM_AVAILABLE")
metal_available()  = _accelerator_env_flag("AXIOM_METAL_AVAILABLE")
tpu_available()    = _accelerator_env_flag("AXIOM_TPU_AVAILABLE")
npu_available()    = _accelerator_env_flag("AXIOM_NPU_AVAILABLE")
dsp_available()    = _accelerator_env_flag("AXIOM_DSP_AVAILABLE")
ppu_available()    = _accelerator_env_flag("AXIOM_PPU_AVAILABLE")
math_available()   = _accelerator_env_flag("AXIOM_MATH_AVAILABLE")
fpga_available()   = _accelerator_env_flag("AXIOM_FPGA_AVAILABLE")
vpu_available()    = _accelerator_env_flag("AXIOM_VPU_AVAILABLE")
qpu_available()    = _accelerator_env_flag("AXIOM_QPU_AVAILABLE")
crypto_available() = _accelerator_env_flag("AXIOM_CRYPTO_AVAILABLE")

cuda_device_count()   = _accelerator_env_count("AXIOM_CUDA_AVAILABLE", "AXIOM_CUDA_DEVICE_COUNT")
rocm_device_count()   = _accelerator_env_count("AXIOM_ROCM_AVAILABLE", "AXIOM_ROCM_DEVICE_COUNT")
metal_device_count()  = _accelerator_env_count("AXIOM_METAL_AVAILABLE", "AXIOM_METAL_DEVICE_COUNT")
tpu_device_count()    = _accelerator_env_count("AXIOM_TPU_AVAILABLE", "AXIOM_TPU_DEVICE_COUNT")
npu_device_count()    = _accelerator_env_count("AXIOM_NPU_AVAILABLE", "AXIOM_NPU_DEVICE_COUNT")
dsp_device_count()    = _accelerator_env_count("AXIOM_DSP_AVAILABLE", "AXIOM_DSP_DEVICE_COUNT")
ppu_device_count()    = _accelerator_env_count("AXIOM_PPU_AVAILABLE", "AXIOM_PPU_DEVICE_COUNT")
math_device_count()   = _accelerator_env_count("AXIOM_MATH_AVAILABLE", "AXIOM_MATH_DEVICE_COUNT")
fpga_device_count()   = _accelerator_env_count("AXIOM_FPGA_AVAILABLE", "AXIOM_FPGA_DEVICE_COUNT")
vpu_device_count()    = _accelerator_env_count("AXIOM_VPU_AVAILABLE", "AXIOM_VPU_DEVICE_COUNT")
qpu_device_count()    = _accelerator_env_count("AXIOM_QPU_AVAILABLE", "AXIOM_QPU_DEVICE_COUNT")
crypto_device_count() = _accelerator_env_count("AXIOM_CRYPTO_AVAILABLE", "AXIOM_CRYPTO_DEVICE_COUNT")

# ============================================================================
# Backend Unions
# ============================================================================

const CoprocessorBackend = Union{TPUBackend, NPUBackend, DSPBackend, PPUBackend,
                                 MathBackend, FPGABackend, VPUBackend, QPUBackend,
                                 CryptoBackend}

const GPUBackend = Union{CUDABackend, ROCmBackend, MetalBackend}

# ============================================================================
# Coprocessor Helpers
# ============================================================================

function _coprocessor_label(b::CoprocessorBackend)
    b isa TPUBackend    ? "TPU" :
    b isa NPUBackend    ? "NPU" :
    b isa DSPBackend    ? "DSP" :
    b isa PPUBackend    ? "PPU" :
    b isa MathBackend   ? "MATH" :
    b isa FPGABackend   ? "FPGA" :
    b isa VPUBackend    ? "VPU" :
    b isa QPUBackend    ? "QPU" :
    b isa CryptoBackend ? "CRYPTO" :
    string(typeof(b))
end

_coprocessor_key(b::CoprocessorBackend) = lowercase(_coprocessor_label(b))
_coprocessor_required_env(b::CoprocessorBackend) = "AXIOM_$(_coprocessor_label(b))_REQUIRED"

function _coprocessor_required(b::CoprocessorBackend)
    specific = _backend_env_available(_coprocessor_required_env(b))
    specific !== nothing && return specific
    global_req = _backend_env_available("AXIOM_COPROCESSOR_REQUIRED")
    global_req === nothing ? false : global_req
end

# ============================================================================
# Runtime Diagnostics
# ============================================================================

const _DIAGNOSTICS = Dict{String, Dict{String, Int}}()
for key in ("cuda", "rocm", "metal", "tpu", "npu", "dsp", "ppu", "math", "fpga", "vpu", "qpu", "crypto")
    _DIAGNOSTICS[key] = Dict("compile_fallbacks" => 0, "runtime_errors" => 0,
                             "runtime_fallbacks" => 0, "recoveries" => 0)
end

function _record_diagnostic!(key::String, event::String)
    haskey(_DIAGNOSTICS, key) || return
    _DIAGNOSTICS[key][event] = get(_DIAGNOSTICS[key], event, 0) + 1
end

_record_diagnostic!(b::CoprocessorBackend, event) = _record_diagnostic!(_coprocessor_key(b), event)
_record_diagnostic!(::CUDABackend, event) = _record_diagnostic!("cuda", event)
_record_diagnostic!(::ROCmBackend, event) = _record_diagnostic!("rocm", event)
_record_diagnostic!(::MetalBackend, event) = _record_diagnostic!("metal", event)

function reset_diagnostics!()
    for counters in values(_DIAGNOSTICS)
        for k in keys(counters); counters[k] = 0; end
    end
end

function runtime_diagnostics()
    Dict(
        "generated_at" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "self_healing_enabled" => _self_healing_enabled(),
        "backends" => deepcopy(_DIAGNOSTICS),
    )
end

_self_healing_enabled() = let f = _backend_env_available("AXIOM_SELF_HEAL"); f === nothing ? true : f end

# ============================================================================
# Auto-Detection
# ============================================================================

function detect_gpu()
    cuda_available()  && cuda_device_count()  > 0 && return CUDABackend(0)
    rocm_available()  && rocm_device_count()  > 0 && return ROCmBackend(0)
    metal_available() && metal_device_count() > 0 && return MetalBackend(0)
    nothing
end

function detect_coprocessor()
    tpu_available()    && tpu_device_count()    > 0 && return TPUBackend(0)
    npu_available()    && npu_device_count()    > 0 && return NPUBackend(0)
    vpu_available()    && vpu_device_count()    > 0 && return VPUBackend(0)
    qpu_available()    && qpu_device_count()    > 0 && return QPUBackend(0)
    ppu_available()    && ppu_device_count()    > 0 && return PPUBackend(0)
    math_available()   && math_device_count()   > 0 && return MathBackend(0)
    crypto_available() && crypto_device_count() > 0 && return CryptoBackend(0)
    fpga_available()   && fpga_device_count()   > 0 && return FPGABackend(0)
    dsp_available()    && dsp_device_count()    > 0 && return DSPBackend(0)
    nothing
end

function detect_accelerator()
    gpu = detect_gpu()
    gpu !== nothing && return gpu
    detect_coprocessor()
end

# ============================================================================
# Capability Report
# ============================================================================

function capability_report()
    selected = detect_accelerator()
    backends = Dict{String,Any}()
    for (label, avail_fn, count_fn) in (
        ("CUDA",   cuda_available,   cuda_device_count),
        ("ROCm",   rocm_available,   rocm_device_count),
        ("Metal",  metal_available,  metal_device_count),
        ("TPU",    tpu_available,    tpu_device_count),
        ("NPU",    npu_available,    npu_device_count),
        ("VPU",    vpu_available,    vpu_device_count),
        ("QPU",    qpu_available,    qpu_device_count),
        ("PPU",    ppu_available,    ppu_device_count),
        ("MATH",   math_available,   math_device_count),
        ("CRYPTO", crypto_available, crypto_device_count),
        ("FPGA",   fpga_available,   fpga_device_count),
        ("DSP",    dsp_available,    dsp_device_count),
    )
        avail = avail_fn()
        count = count_fn()
        backends[label] = Dict("available" => avail, "device_count" => count,
                               "compilable" => avail && count > 0)
    end
    Dict(
        "generated_at" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "strategy_order" => ["CUDA","ROCm","Metal","TPU","NPU","VPU","QPU","PPU","MATH","CRYPTO","FPGA","DSP"],
        "selected_backend" => selected === nothing ? nothing : string(typeof(selected)),
        "diagnostics" => runtime_diagnostics(),
        "backends" => backends,
    )
end

# ============================================================================
# Domain-Specific Operation Hooks -- Cladistics
# ============================================================================

# Julia fallback implementations
backend_distance_matrix(::JuliaBackend, args...) = nothing
backend_neighbor_join(::JuliaBackend, args...) = nothing
backend_parsimony_score(::JuliaBackend, args...) = nothing
backend_bootstrap_replicate(::JuliaBackend, args...) = nothing
backend_tree_search(::JuliaBackend, args...) = nothing

# Coprocessor extension hooks (concrete extensions overload these)
function backend_coprocessor_distance_matrix end
function backend_coprocessor_neighbor_join end
function backend_coprocessor_parsimony_score end
function backend_coprocessor_bootstrap_replicate end
function backend_coprocessor_tree_search end

# Self-healing fallback generation for coprocessor hooks
for (hook, cpu_op) in (
    (:backend_coprocessor_distance_matrix, :backend_distance_matrix),
    (:backend_coprocessor_neighbor_join, :backend_neighbor_join),
    (:backend_coprocessor_parsimony_score, :backend_parsimony_score),
    (:backend_coprocessor_bootstrap_replicate, :backend_bootstrap_replicate),
    (:backend_coprocessor_tree_search, :backend_tree_search),
)
    hook_name = String(hook)
    @eval function $hook(backend::CoprocessorBackend, args...)
        if _coprocessor_required(backend)
            throw(ErrorException(string(
                _coprocessor_label(backend),
                " extension not loaded for `", $hook_name,
                "` while strict mode enabled (set ",
                _coprocessor_required_env(backend), "=0 to allow fallback)."
            )))
        end
        _record_diagnostic!(backend, "runtime_fallbacks")
        @warn string(_coprocessor_label(backend),
            " hook not loaded for `", $hook_name, "`, falling back to Julia") maxlog=1
        return $cpu_op(JuliaBackend(), args...)
    end
end

# Dispatch: CoprocessorBackend -> coprocessor hook
backend_distance_matrix(b::CoprocessorBackend, args...) = backend_coprocessor_distance_matrix(b, args...)
backend_neighbor_join(b::CoprocessorBackend, args...) = backend_coprocessor_neighbor_join(b, args...)
backend_parsimony_score(b::CoprocessorBackend, args...) = backend_coprocessor_parsimony_score(b, args...)
backend_bootstrap_replicate(b::CoprocessorBackend, args...) = backend_coprocessor_bootstrap_replicate(b, args...)
backend_tree_search(b::CoprocessorBackend, args...) = backend_coprocessor_tree_search(b, args...)

# Dispatch: GPUBackend -> Julia fallback (extensions override)
backend_distance_matrix(b::GPUBackend, args...) = backend_distance_matrix(JuliaBackend(), args...)
backend_neighbor_join(b::GPUBackend, args...) = backend_neighbor_join(JuliaBackend(), args...)
backend_parsimony_score(b::GPUBackend, args...) = backend_parsimony_score(JuliaBackend(), args...)
backend_bootstrap_replicate(b::GPUBackend, args...) = backend_bootstrap_replicate(JuliaBackend(), args...)
backend_tree_search(b::GPUBackend, args...) = backend_tree_search(JuliaBackend(), args...)
