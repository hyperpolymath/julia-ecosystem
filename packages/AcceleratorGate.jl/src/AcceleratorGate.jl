# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# AcceleratorGate.jl — Shared coprocessor dispatch infrastructure
# Extracted from Axiom.jl backend abstraction to eliminate duplication
# across the hyperpolymath Julia ecosystem.

"""
    AcceleratorGate

Shared coprocessor dispatch infrastructure for Julia packages. Provides a unified
backend type hierarchy (GPU, TPU, NPU, FPGA, QPU, etc.) with automatic detection,
platform-aware selection, memory tracking, and self-healing fallback hooks.

# Key Features
- Backend type hierarchy covering CUDA, ROCm, Metal, and 9 coprocessor families
- Platform-aware auto-detection (mobile, embedded, server, desktop)
- Resource-aware backend selection with cost estimation
- Self-healing fallback hook generation for domain packages

# Example
```julia
using AcceleratorGate
report = capability_report()
backend = select_backend(:matmul, 1_000_000)
```
"""
module AcceleratorGate

using Dates

export AbstractBackend, JuliaBackend, RustBackend, ZigBackend,
       CUDABackend, ROCmBackend, MetalBackend,
       TPUBackend, NPUBackend, DSPBackend, PPUBackend,
       MathBackend, FPGABackend, VPUBackend, QPUBackend, CryptoBackend,
       CoprocessorBackend, GPUBackend,
       current_backend, set_backend!, @with_backend,
       cuda_available, rocm_available, metal_available,
       tpu_available, npu_available, dsp_available, ppu_available,
       math_available, fpga_available, vpu_available, qpu_available,
       crypto_available,
       cuda_device_count, rocm_device_count, metal_device_count,
       tpu_device_count, npu_device_count, dsp_device_count, ppu_device_count,
       math_device_count, fpga_device_count, vpu_device_count, qpu_device_count,
       crypto_device_count,
       detect_gpu, detect_coprocessor, detect_accelerator,
       capability_report, runtime_diagnostics, reset_diagnostics!,
       _record_diagnostic!, _coprocessor_label, _coprocessor_key,
       _coprocessor_required, _coprocessor_required_env,
       _self_healing_enabled,
       generate_self_healing_hooks,
       # Platform awareness
       PlatformInfo, detect_platform, _arch_compatible,
       # Resource-aware dispatch
       DeviceCapabilities, device_capabilities, fits_on_device,
       estimate_cost, select_backend,
       # Memory tracking
       track_allocation!, track_deallocation!, memory_usage, memory_report,
       # Operation registry
       register_operation!, supports_operation, supported_operations,
       # Backend specialties
       BACKEND_SPECIALTIES, is_specialized

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
# Platform Detection
# ============================================================================

"""
    PlatformInfo

Describes the host platform's operating system, CPU architecture, and
environment class (mobile, embedded, server). Used by `select_backend` and
`_arch_compatible` to make platform-aware dispatch decisions.
"""
struct PlatformInfo
    os::Symbol              # :linux, :macos, :windows, :freebsd, :openbsd, :minix, :android, :ios, :unknown
    arch::Symbol            # :x86_64, :aarch64, :arm, :riscv64, :powerpc64, :mips, :unknown
    is_mobile::Bool         # iOS or Android
    is_embedded::Bool       # MINIX, bare metal, RTOS-like environments
    is_server::Bool         # Detected server environment (many cores / large RAM)
    julia_version::VersionNumber
    word_size::Int          # 32 or 64
    endianness::Symbol      # :little or :big
end

"""
    detect_platform() -> PlatformInfo

Probe the current host and return a `PlatformInfo` describing the OS,
architecture, and environment class.
"""
function detect_platform()::PlatformInfo
    os = _detect_os()
    arch = Sys.ARCH
    is_mobile = os in (:android, :ios)
    is_embedded = os === :minix || _is_embedded_env()
    is_server = _is_server_env()
    endian = ENDIAN_BOM == 0x04030201 ? :little : :big
    PlatformInfo(os, arch, is_mobile, is_embedded, is_server,
                 VERSION, Sys.WORD_SIZE, endian)
end

"""
    _detect_os() -> Symbol

Determine the operating system, distinguishing Android from Linux and iOS
from macOS where possible.
"""
function _detect_os()::Symbol
    Sys.islinux()   && return _check_android() ? :android : :linux
    Sys.isapple()   && return _check_ios() ? :ios : :macos
    Sys.iswindows()  && return :windows
    Sys.isbsd()      && return :freebsd
    # Attempt uname for exotic OSes (MINIX, OpenBSD, etc.)
    try
        uname = lowercase(strip(read(`uname -s`, String)))
        contains(uname, "minix")   && return :minix
        contains(uname, "openbsd") && return :openbsd
    catch
        # uname unavailable (Windows, sandboxed, etc.) — already handled above
    end
    :unknown
end

"""
    _check_android() -> Bool

Heuristic: Android sets `ANDROID_ROOT` and ships `/system/app`.
"""
function _check_android()::Bool
    haskey(ENV, "ANDROID_ROOT") || isdir("/system/app")
end

"""
    _check_ios() -> Bool

Rough heuristic: Apple + aarch64 + no /usr/local (macOS Homebrew path) likely
indicates an iOS/iPadOS/Catalyst environment.
"""
function _check_ios()::Bool
    Sys.isapple() && Sys.ARCH === :aarch64 && !isdir("/usr/local")
end

"""
    _is_embedded_env() -> Bool

Heuristic for resource-constrained embedded targets: very few CPU threads
and less than 512 MiB total RAM.
"""
function _is_embedded_env()::Bool
    Sys.CPU_THREADS <= 2 && Sys.total_memory() < 512 * 1024 * 1024
end

"""
    _is_server_env() -> Bool

Heuristic for server-class hardware: 16+ CPU threads or 32+ GiB RAM.
"""
function _is_server_env()::Bool
    Sys.CPU_THREADS >= 16 || Sys.total_memory() > 32 * Int64(1024)^3
end

# ============================================================================
# Architecture-Aware Coprocessor Compatibility
# ============================================================================

"""
    _arch_compatible(backend::AbstractBackend, arch::Symbol) -> Bool

Check whether `backend` is compatible with CPU architecture `arch`.
Some accelerators are only reachable on certain host architectures
(e.g. CUDA requires x86_64 or aarch64 for Jetson/Grace).
"""
function _arch_compatible(backend::AbstractBackend, arch::Symbol)::Bool
    # CUDA requires x86_64 or aarch64 (Jetson / Grace Hopper)
    backend isa CUDABackend  && return arch in (:x86_64, :aarch64)
    # Metal requires Apple Silicon (aarch64 on macOS)
    backend isa MetalBackend && return arch === :aarch64
    # ROCm requires x86_64
    backend isa ROCmBackend  && return arch === :x86_64
    # VPU / DSP are common on ARM SoCs (mobile, embedded)
    backend isa VPUBackend   && return arch in (:aarch64, :arm)
    backend isa DSPBackend   && return arch in (:aarch64, :arm, :x86_64)
    # NPU available on modern ARM SoCs and some Intel x86
    backend isa NPUBackend   && return arch in (:aarch64, :arm, :x86_64)
    # QPU is cloud-accessible from any architecture
    backend isa QPUBackend   && return true
    # FPGA typically via PCIe on x86 / ARM servers
    backend isa FPGABackend  && return arch in (:x86_64, :aarch64)
    # TPU — cloud-accessible or edge TPU (USB/PCIe) on x86/ARM
    backend isa TPUBackend   && return arch in (:x86_64, :aarch64)
    # Julia, Rust, Zig, Math, Crypto, PPU — always compatible
    true
end

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
    platform = detect_platform()
    if cuda_available() && cuda_device_count() > 0
        b = CUDABackend(0)
        _arch_compatible(b, platform.arch) && return b
    end
    if rocm_available() && rocm_device_count() > 0
        b = ROCmBackend(0)
        _arch_compatible(b, platform.arch) && return b
    end
    if metal_available() && metal_device_count() > 0
        b = MetalBackend(0)
        _arch_compatible(b, platform.arch) && return b
    end
    nothing
end

function detect_coprocessor()
    platform = detect_platform()
    for (avail_fn, count_fn, ctor) in (
        (tpu_available,    tpu_device_count,    TPUBackend),
        (npu_available,    npu_device_count,    NPUBackend),
        (vpu_available,    vpu_device_count,    VPUBackend),
        (qpu_available,    qpu_device_count,    QPUBackend),
        (ppu_available,    ppu_device_count,    PPUBackend),
        (math_available,   math_device_count,   MathBackend),
        (crypto_available, crypto_device_count, CryptoBackend),
        (fpga_available,   fpga_device_count,   FPGABackend),
        (dsp_available,    dsp_device_count,    DSPBackend),
    )
        if avail_fn() && count_fn() > 0
            b = ctor(0)
            _arch_compatible(b, platform.arch) && return b
        end
    end
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
    platform = detect_platform()
    Dict(
        "generated_at" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SS.sssZ"),
        "platform" => Dict(
            "os" => String(platform.os),
            "arch" => String(platform.arch),
            "is_mobile" => platform.is_mobile,
            "is_embedded" => platform.is_embedded,
            "is_server" => platform.is_server,
            "word_size" => platform.word_size,
            "endianness" => String(platform.endianness),
            "julia_version" => string(platform.julia_version),
        ),
        "strategy_order" => ["CUDA","ROCm","Metal","TPU","NPU","VPU","QPU","PPU","MATH","CRYPTO","FPGA","DSP"],
        "selected_backend" => selected === nothing ? nothing : string(typeof(selected)),
        "diagnostics" => runtime_diagnostics(),
        "backends" => backends,
    )
end

# ============================================================================
# Self-Healing Hook Generator
# ============================================================================

"""
    generate_self_healing_hooks(mod, hooks)

Generate self-healing fallback methods for domain-specific coprocessor hooks.

`hooks` is a vector of `(coprocessor_hook_name, cpu_fallback_name)` symbol pairs.
For each pair, this generates:
- A `function coprocessor_hook_name end` declaration
- A `CoprocessorBackend` method that falls back to the CPU op with self-healing
- `GPUBackend` dispatch that falls back to the CPU op

Call this in the consuming package's `backends/abstract.jl` after defining
the Julia fallback implementations.

# Example
```julia
using AcceleratorGate
using AcceleratorGate: generate_self_healing_hooks

# Define Julia fallback implementations
backend_my_op(::JuliaBackend, args...) = nothing

# Generate coprocessor dispatch
generate_self_healing_hooks(@__MODULE__, [
    (:backend_coprocessor_my_op, :backend_my_op),
])

# Wire up dispatch
backend_my_op(b::CoprocessorBackend, args...) = backend_coprocessor_my_op(b, args...)
backend_my_op(b::GPUBackend, args...) = backend_my_op(JuliaBackend(), args...)
```
"""
function generate_self_healing_hooks(mod::Module, hooks::Vector{Tuple{Symbol,Symbol}})
    for (hook, cpu_op) in hooks
        hook_name = String(hook)
        # Declare the function if it doesn't exist
        if !isdefined(mod, hook)
            Core.eval(mod, :(function $hook end))
        end
        # Generate the self-healing fallback method
        Core.eval(mod, quote
            function $hook(backend::$CoprocessorBackend, args...)
                if $(_coprocessor_required)(backend)
                    throw(ErrorException(string(
                        $(_coprocessor_label)(backend),
                        " extension not loaded for `", $hook_name,
                        "` while strict mode enabled (set ",
                        $(_coprocessor_required_env)(backend), "=0 to allow fallback)."
                    )))
                end
                $(_record_diagnostic!)(backend, "runtime_fallbacks")
                @warn string($(_coprocessor_label)(backend),
                    " hook not loaded for `", $hook_name, "`, falling back to Julia") maxlog=1
                return $cpu_op($JuliaBackend(), args...)
            end
        end)
    end
end

# ============================================================================
# Device Capabilities
# ============================================================================

"""
    DeviceCapabilities

Describes the hardware capabilities of a specific backend device, including
compute resources, memory, precision support, and vendor information.
Extensions should override `device_capabilities` to return populated instances.
"""
struct DeviceCapabilities
    backend::AbstractBackend
    compute_units::Int          # cores/SMs/CUs/slices
    clock_mhz::Int              # clock speed
    memory_bytes::Int64         # total device memory
    memory_available::Int64     # currently available memory
    max_workgroup_size::Int     # max threads per workgroup
    supports_f64::Bool          # double precision support
    supports_f16::Bool          # half precision support
    supports_int8::Bool         # int8 quantized ops
    vendor::String              # "NVIDIA", "AMD", "Apple", "Intel", "Google", "Qualcomm", etc.
    driver_version::String      # driver/SDK version
end

# ============================================================================
# Resource Query Functions
# ============================================================================

"""
    device_capabilities(b::AbstractBackend) -> Union{Nothing, DeviceCapabilities}

Query device capabilities for a given backend. Returns `nothing` by default;
backend extensions should override this with real hardware queries.
"""
function device_capabilities(b::AbstractBackend)::Union{Nothing, DeviceCapabilities}
    # Default: return nothing (extensions override)
    nothing
end

"""
    fits_on_device(b::AbstractBackend, required_memory::Int64) -> Bool

Check whether a workload requiring `required_memory` bytes can fit on the
device associated with backend `b`.
"""
function fits_on_device(b::AbstractBackend, required_memory::Int64)::Bool
    caps = device_capabilities(b)
    caps === nothing && return false
    caps.memory_available >= required_memory
end

"""
    estimate_cost(b::AbstractBackend, op::Symbol, data_size::Int) -> Float64

Estimate the relative cost of performing operation `op` on `data_size` elements
using backend `b`. Lower values are better. Returns `Inf` for backends that
have not registered cost models (extensions override with real estimates).
"""
function estimate_cost(b::AbstractBackend, op::Symbol, data_size::Int)::Float64
    # Default: return Inf for non-Julia backends (extensions override with real estimates)
    b isa JuliaBackend && return Float64(data_size)  # CPU cost ~ data size
    Inf
end

# ============================================================================
# Resource-Aware Backend Selection
# ============================================================================

"""
    select_backend(op, data_size; required_memory=0, prefer_precision=:f64, exclude=DataType[]) -> AbstractBackend

Auto-select the best backend for a given workload by considering available
hardware, memory requirements, precision needs, and estimated operation cost.
"""
function select_backend(op::Symbol, data_size::Int;
                        required_memory::Int64=Int64(0),
                        prefer_precision::Symbol=:f64,
                        exclude::Vector{DataType}=DataType[])::AbstractBackend
    platform = detect_platform()
    candidates = AbstractBackend[]

    # ---- Platform-aware candidate gathering --------------------------------

    if platform.is_mobile
        # Mobile: prefer low-power accelerators (NPU > VPU), avoid GPU for battery
        for (avail_fn, count_fn, ctor) in (
            (npu_available, npu_device_count, NPUBackend),
            (vpu_available, vpu_device_count, VPUBackend),
            (dsp_available, dsp_device_count, DSPBackend),
        )
            if avail_fn() && count_fn() > 0
                b = ctor(0)
                _arch_compatible(b, platform.arch) && !(typeof(b) in exclude) && push!(candidates, b)
            end
        end
    elseif platform.is_embedded
        # Embedded: prefer FPGA/DSP (low memory footprint), check strictly
        for (avail_fn, count_fn, ctor) in (
            (fpga_available, fpga_device_count, FPGABackend),
            (dsp_available,  dsp_device_count,  DSPBackend),
        )
            if avail_fn() && count_fn() > 0
                b = ctor(0)
                _arch_compatible(b, platform.arch) && !(typeof(b) in exclude) && push!(candidates, b)
            end
        end
    else
        # Desktop / Server: try GPU first for maximum throughput
        gpu = detect_gpu()
        gpu !== nothing && !(typeof(gpu) in exclude) && push!(candidates, gpu)

        # Then coprocessors (server environments get all of them)
        coproc = detect_coprocessor()
        coproc !== nothing && !(typeof(coproc) in exclude) && push!(candidates, coproc)
    end

    # Always include Julia fallback
    push!(candidates, JuliaBackend())

    # ---- Filter by architecture compatibility ------------------------------
    filter!(b -> _arch_compatible(b, platform.arch), candidates)

    # ---- Filter by memory requirement --------------------------------------
    if required_memory > 0
        filter!(b -> fits_on_device(b, required_memory), candidates)
    end

    # ---- Filter by precision requirement -----------------------------------
    if prefer_precision == :f64
        filter!(b -> begin
            caps = device_capabilities(b)
            caps === nothing || caps.supports_f64
        end, candidates)
    end

    # ---- Select lowest cost ------------------------------------------------
    best = JuliaBackend()
    best_cost = estimate_cost(best, op, data_size)
    for b in candidates
        c = estimate_cost(b, op, data_size)
        if c < best_cost
            best = b
            best_cost = c
        end
    end
    best
end

# ============================================================================
# Memory Tracking
# ============================================================================

"""Per-backend memory usage tracker (bytes allocated, keyed by backend type name)."""
const _MEMORY_USAGE = Dict{String, Int64}()

"""
    track_allocation!(b::AbstractBackend, bytes::Int64)

Record that `bytes` were allocated on backend `b`.
"""
function track_allocation!(b::AbstractBackend, bytes::Int64)
    key = string(typeof(b))
    _MEMORY_USAGE[key] = get(_MEMORY_USAGE, key, Int64(0)) + bytes
end

"""
    track_deallocation!(b::AbstractBackend, bytes::Int64)

Record that `bytes` were freed on backend `b`. Usage will not go below zero.
"""
function track_deallocation!(b::AbstractBackend, bytes::Int64)
    key = string(typeof(b))
    _MEMORY_USAGE[key] = max(Int64(0), get(_MEMORY_USAGE, key, Int64(0)) - bytes)
end

"""
    memory_usage(b::AbstractBackend) -> Int64

Return the currently tracked memory usage (bytes) for backend `b`.
"""
function memory_usage(b::AbstractBackend)::Int64
    get(_MEMORY_USAGE, string(typeof(b)), Int64(0))
end

"""
    memory_report() -> Dict{String, Int64}

Return a snapshot of memory usage across all backends that have tracked allocations.
"""
function memory_report()::Dict{String, Int64}
    copy(_MEMORY_USAGE)
end

# ============================================================================
# Backend-Specific Operation Registry
# ============================================================================

"""Registry mapping backend types to the set of operations they support efficiently."""
const _BACKEND_OPS = Dict{DataType, Set{Symbol}}()

"""
    register_operation!(backend_type::DataType, op::Symbol)

Register that `backend_type` supports operation `op` efficiently.
"""
function register_operation!(backend_type::DataType, op::Symbol)
    if !haskey(_BACKEND_OPS, backend_type)
        _BACKEND_OPS[backend_type] = Set{Symbol}()
    end
    push!(_BACKEND_OPS[backend_type], op)
end

"""
    supports_operation(b::AbstractBackend, op::Symbol) -> Bool

Check whether backend `b` has registered support for operation `op`.
"""
function supports_operation(b::AbstractBackend, op::Symbol)::Bool
    ops = get(_BACKEND_OPS, typeof(b), Set{Symbol}())
    op in ops
end

"""
    supported_operations(b::AbstractBackend) -> Set{Symbol}

Return the set of operations that backend `b` has registered support for.
"""
function supported_operations(b::AbstractBackend)::Set{Symbol}
    get(_BACKEND_OPS, typeof(b), Set{Symbol}())
end

# ============================================================================
# Backend Specialties (Coprocessor Hints)
# ============================================================================

"""
Static mapping of what each coprocessor backend type is best at.
Used by `is_specialized` to quickly check if a backend is a good fit
for a particular operation class.
"""
const BACKEND_SPECIALTIES = Dict{DataType, Vector{Symbol}}(
    CUDABackend => [:matmul, :fft, :conv, :gemm, :ntt, :sampling, :reduction],
    ROCmBackend => [:matmul, :fft, :conv, :gemm, :ntt, :sampling, :reduction],
    MetalBackend => [:matmul, :fft, :conv, :gemm, :ntt, :sampling],
    TPUBackend => [:matmul, :conv, :einsum, :systolic_array, :batch_inference],
    NPUBackend => [:inference, :quantized_matmul, :conv, :activation],
    FPGABackend => [:pipeline, :bitwise, :custom_datapath, :low_latency, :streaming],
    VPUBackend => [:simd, :vector_add, :vector_mul, :dot_product, :reduction],
    QPUBackend => [:quantum_gate, :measurement, :entanglement, :state_evolution],
    DSPBackend => [:fft, :fir_filter, :iir_filter, :convolution, :correlation],
    PPUBackend => [:physics_sim, :collision, :rigid_body, :fluid_sim],
    MathBackend => [:bignum, :arbitrary_precision, :symbolic, :polynomial],
    CryptoBackend => [:aes, :sha, :ntt, :lattice_multiply, :modular_exp],
)

"""
    is_specialized(b::AbstractBackend, op::Symbol) -> Bool

Check whether backend `b` is specialized for operation `op` according to
the static `BACKEND_SPECIALTIES` table.
"""
function is_specialized(b::AbstractBackend, op::Symbol)::Bool
    specialties = get(BACKEND_SPECIALTIES, typeof(b), Symbol[])
    op in specialties
end

end # module AcceleratorGate
