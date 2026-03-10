# SPDX-License-Identifier: PMPL-1.0-or-later
using Test
using AcceleratorGate

@testset "AcceleratorGate.jl" begin
    @testset "Backend types" begin
        @test JuliaBackend() isa AbstractBackend
        @test CUDABackend(0) isa AbstractBackend
        @test ROCmBackend(0) isa AbstractBackend
        @test MetalBackend(0) isa AbstractBackend
        @test TPUBackend(0) isa AbstractBackend
        @test NPUBackend(0) isa AbstractBackend
        @test DSPBackend(0) isa AbstractBackend
        @test PPUBackend(0) isa AbstractBackend
        @test MathBackend(0) isa AbstractBackend
        @test FPGABackend(0) isa AbstractBackend
        @test VPUBackend(0) isa AbstractBackend
        @test QPUBackend(0) isa AbstractBackend
        @test CryptoBackend(0) isa AbstractBackend
        @test ZigBackend("/tmp/lib.so") isa AbstractBackend
        @test RustBackend("/tmp/lib.so") isa AbstractBackend
    end

    @testset "Backend unions" begin
        @test CUDABackend(0) isa GPUBackend
        @test ROCmBackend(0) isa GPUBackend
        @test MetalBackend(0) isa GPUBackend
        @test TPUBackend(0) isa CoprocessorBackend
        @test NPUBackend(0) isa CoprocessorBackend
        @test FPGABackend(0) isa CoprocessorBackend
        @test !(JuliaBackend() isa GPUBackend)
        @test !(CUDABackend(0) isa CoprocessorBackend)
    end

    @testset "Backend selection" begin
        @test current_backend() isa JuliaBackend
        set_backend!(CUDABackend(1))
        @test current_backend() isa CUDABackend
        @test current_backend().device == 1
        set_backend!(JuliaBackend())
        @test current_backend() isa JuliaBackend
    end

    @testset "@with_backend" begin
        @test current_backend() isa JuliaBackend
        @with_backend MetalBackend(0) begin
            @test current_backend() isa MetalBackend
        end
        @test current_backend() isa JuliaBackend
    end

    @testset "Detection (no hardware)" begin
        # Without env vars set, everything should return false/0
        @test cuda_available() == false
        @test rocm_available() == false
        @test metal_available() == false
        @test cuda_device_count() == 0
        @test detect_gpu() === nothing
        @test detect_coprocessor() === nothing
        @test detect_accelerator() === nothing
    end

    @testset "Detection with env vars" begin
        withenv("AXIOM_CUDA_AVAILABLE" => "true", "AXIOM_CUDA_DEVICE_COUNT" => "2") do
            @test cuda_available() == true
            @test cuda_device_count() == 2
            gpu = detect_gpu()
            @test gpu isa CUDABackend
            @test gpu.device == 0
        end
    end

    @testset "Capability report" begin
        report = capability_report()
        @test haskey(report, "generated_at")
        @test haskey(report, "strategy_order")
        @test haskey(report, "backends")
        @test haskey(report, "platform")
        @test report["selected_backend"] === nothing
        @test length(report["strategy_order"]) == 12
        # Platform sub-dict has expected keys
        plat = report["platform"]
        @test haskey(plat, "os")
        @test haskey(plat, "arch")
        @test haskey(plat, "is_mobile")
        @test haskey(plat, "is_embedded")
        @test haskey(plat, "is_server")
        @test haskey(plat, "word_size")
        @test haskey(plat, "endianness")
        @test haskey(plat, "julia_version")
    end

    @testset "Diagnostics" begin
        reset_diagnostics!()
        _record_diagnostic!("cuda", "runtime_fallbacks")
        _record_diagnostic!("cuda", "runtime_fallbacks")
        diag = runtime_diagnostics()
        @test diag["backends"]["cuda"]["runtime_fallbacks"] == 2
        reset_diagnostics!()
        @test runtime_diagnostics()["backends"]["cuda"]["runtime_fallbacks"] == 0
    end

    @testset "Coprocessor helpers" begin
        @test _coprocessor_label(TPUBackend(0)) == "TPU"
        @test _coprocessor_label(NPUBackend(0)) == "NPU"
        @test _coprocessor_key(FPGABackend(0)) == "fpga"
        @test _coprocessor_required_env(QPUBackend(0)) == "AXIOM_QPU_REQUIRED"
        @test _coprocessor_required(TPUBackend(0)) == false
    end

    # ========================================================================
    # Resource-Aware Dispatch Tests
    # ========================================================================

    @testset "DeviceCapabilities struct" begin
        caps = DeviceCapabilities(
            JuliaBackend(), 8, 3200, Int64(16_000_000_000), Int64(12_000_000_000),
            1024, true, true, true, "Generic", "1.0.0"
        )
        @test caps.backend isa JuliaBackend
        @test caps.compute_units == 8
        @test caps.clock_mhz == 3200
        @test caps.memory_bytes == Int64(16_000_000_000)
        @test caps.memory_available == Int64(12_000_000_000)
        @test caps.max_workgroup_size == 1024
        @test caps.supports_f64 == true
        @test caps.supports_f16 == true
        @test caps.supports_int8 == true
        @test caps.vendor == "Generic"
        @test caps.driver_version == "1.0.0"
    end

    @testset "device_capabilities default" begin
        # Default implementation returns nothing for all backends
        @test device_capabilities(JuliaBackend()) === nothing
        @test device_capabilities(CUDABackend(0)) === nothing
        @test device_capabilities(TPUBackend(0)) === nothing
    end

    @testset "fits_on_device" begin
        # Without capabilities, nothing fits
        @test fits_on_device(JuliaBackend(), Int64(1024)) == false
        @test fits_on_device(CUDABackend(0), Int64(1024)) == false
    end

    @testset "estimate_cost" begin
        # JuliaBackend cost equals data_size
        @test estimate_cost(JuliaBackend(), :matmul, 1000) == 1000.0
        # Other backends default to Inf
        @test estimate_cost(CUDABackend(0), :matmul, 1000) == Inf
        @test estimate_cost(TPUBackend(0), :conv, 500) == Inf
    end

    @testset "select_backend (no hardware)" begin
        # Without any accelerators available, should always select JuliaBackend
        b = select_backend(:matmul, 1000)
        @test b isa JuliaBackend

        # With exclusions
        b2 = select_backend(:fft, 500; exclude=[CUDABackend])
        @test b2 isa JuliaBackend

        # With memory requirement (JuliaBackend has no caps, so if we require
        # memory and no backends report caps, candidates get filtered out —
        # but JuliaBackend is the fallback best via initial assignment)
        b3 = select_backend(:conv, 100; required_memory=Int64(0))
        @test b3 isa JuliaBackend
    end

    @testset "select_backend with simulated GPU" begin
        withenv("AXIOM_CUDA_AVAILABLE" => "true", "AXIOM_CUDA_DEVICE_COUNT" => "1") do
            # CUDA is detected but has Inf cost (no extension), so Julia wins
            b = select_backend(:matmul, 1000)
            @test b isa JuliaBackend

            # With CUDA excluded, still Julia
            b2 = select_backend(:matmul, 1000; exclude=[CUDABackend])
            @test b2 isa JuliaBackend
        end
    end

    @testset "Memory tracking" begin
        # Clear any prior state
        empty!(AcceleratorGate._MEMORY_USAGE)

        julia_b = JuliaBackend()
        cuda_b = CUDABackend(0)

        # Initially zero
        @test memory_usage(julia_b) == Int64(0)
        @test memory_usage(cuda_b) == Int64(0)

        # Track allocations
        track_allocation!(julia_b, Int64(1024))
        @test memory_usage(julia_b) == Int64(1024)

        track_allocation!(julia_b, Int64(2048))
        @test memory_usage(julia_b) == Int64(3072)

        track_allocation!(cuda_b, Int64(4096))
        @test memory_usage(cuda_b) == Int64(4096)

        # Track deallocations
        track_deallocation!(julia_b, Int64(512))
        @test memory_usage(julia_b) == Int64(2560)

        # Deallocation should not go below zero
        track_deallocation!(cuda_b, Int64(999999))
        @test memory_usage(cuda_b) == Int64(0)

        # Memory report
        report = memory_report()
        @test report isa Dict{String, Int64}
        @test report["JuliaBackend"] == Int64(2560)
        @test report["CUDABackend"] == Int64(0)

        # Cleanup
        empty!(AcceleratorGate._MEMORY_USAGE)
    end

    @testset "Operation registry" begin
        # Clear any prior state
        empty!(AcceleratorGate._BACKEND_OPS)

        cuda_b = CUDABackend(0)
        tpu_b = TPUBackend(0)

        # Initially no operations registered
        @test supports_operation(cuda_b, :matmul) == false
        @test supported_operations(cuda_b) == Set{Symbol}()

        # Register operations
        register_operation!(CUDABackend, :matmul)
        register_operation!(CUDABackend, :fft)
        register_operation!(TPUBackend, :einsum)

        @test supports_operation(cuda_b, :matmul) == true
        @test supports_operation(cuda_b, :fft) == true
        @test supports_operation(cuda_b, :einsum) == false
        @test supports_operation(tpu_b, :einsum) == true

        ops = supported_operations(cuda_b)
        @test :matmul in ops
        @test :fft in ops
        @test length(ops) == 2

        # JuliaBackend has no registered ops
        @test supported_operations(JuliaBackend()) == Set{Symbol}()

        # Cleanup
        empty!(AcceleratorGate._BACKEND_OPS)
    end

    @testset "Backend specialties" begin
        # BACKEND_SPECIALTIES is populated at module load
        @test haskey(BACKEND_SPECIALTIES, CUDABackend)
        @test haskey(BACKEND_SPECIALTIES, TPUBackend)
        @test haskey(BACKEND_SPECIALTIES, QPUBackend)
        @test haskey(BACKEND_SPECIALTIES, CryptoBackend)
        @test length(BACKEND_SPECIALTIES) == 12  # all coprocessor + GPU types

        # is_specialized checks
        @test is_specialized(CUDABackend(0), :matmul) == true
        @test is_specialized(CUDABackend(0), :fft) == true
        @test is_specialized(CUDABackend(0), :quantum_gate) == false

        @test is_specialized(TPUBackend(0), :einsum) == true
        @test is_specialized(TPUBackend(0), :systolic_array) == true
        @test is_specialized(TPUBackend(0), :fft) == false

        @test is_specialized(QPUBackend(0), :quantum_gate) == true
        @test is_specialized(QPUBackend(0), :entanglement) == true
        @test is_specialized(QPUBackend(0), :matmul) == false

        @test is_specialized(CryptoBackend(0), :aes) == true
        @test is_specialized(CryptoBackend(0), :ntt) == true

        @test is_specialized(DSPBackend(0), :fir_filter) == true
        @test is_specialized(PPUBackend(0), :physics_sim) == true
        @test is_specialized(FPGABackend(0), :streaming) == true
        @test is_specialized(VPUBackend(0), :simd) == true
        @test is_specialized(MathBackend(0), :bignum) == true
        @test is_specialized(NPUBackend(0), :inference) == true
        @test is_specialized(MetalBackend(0), :gemm) == true
        @test is_specialized(ROCmBackend(0), :reduction) == true

        # JuliaBackend has no specialties
        @test is_specialized(JuliaBackend(), :matmul) == false
    end

    # ========================================================================
    # Platform Detection Tests
    # ========================================================================

    @testset "PlatformInfo struct" begin
        pi = detect_platform()
        @test pi isa PlatformInfo
        @test pi.os isa Symbol
        @test pi.arch isa Symbol
        @test pi.arch == Sys.ARCH
        @test pi.julia_version == VERSION
        @test pi.word_size == Sys.WORD_SIZE
        @test pi.endianness in (:little, :big)
        @test pi.is_mobile isa Bool
        @test pi.is_embedded isa Bool
        @test pi.is_server isa Bool
    end

    @testset "detect_platform OS detection" begin
        pi = detect_platform()
        # On the test host, OS should be one of the known symbols
        @test pi.os in (:linux, :macos, :windows, :freebsd, :openbsd, :minix, :android, :ios, :unknown)
        # We know we are NOT on mobile during CI/local tests
        @test pi.is_mobile == false
    end

    @testset "_arch_compatible" begin
        # CUDA: x86_64 and aarch64 only
        @test _arch_compatible(CUDABackend(0), :x86_64)  == true
        @test _arch_compatible(CUDABackend(0), :aarch64)  == true
        @test _arch_compatible(CUDABackend(0), :riscv64)  == false
        @test _arch_compatible(CUDABackend(0), :arm)      == false

        # Metal: aarch64 only (Apple Silicon)
        @test _arch_compatible(MetalBackend(0), :aarch64)  == true
        @test _arch_compatible(MetalBackend(0), :x86_64)   == false

        # ROCm: x86_64 only
        @test _arch_compatible(ROCmBackend(0), :x86_64)   == true
        @test _arch_compatible(ROCmBackend(0), :aarch64)   == false

        # VPU: ARM architectures
        @test _arch_compatible(VPUBackend(0), :aarch64)    == true
        @test _arch_compatible(VPUBackend(0), :arm)        == true
        @test _arch_compatible(VPUBackend(0), :x86_64)     == false

        # DSP: ARM + x86_64
        @test _arch_compatible(DSPBackend(0), :aarch64)    == true
        @test _arch_compatible(DSPBackend(0), :x86_64)     == true
        @test _arch_compatible(DSPBackend(0), :riscv64)    == false

        # NPU: ARM + x86_64
        @test _arch_compatible(NPUBackend(0), :aarch64)    == true
        @test _arch_compatible(NPUBackend(0), :x86_64)     == true
        @test _arch_compatible(NPUBackend(0), :mips)       == false

        # QPU: cloud-accessible, always compatible
        @test _arch_compatible(QPUBackend(0), :x86_64)     == true
        @test _arch_compatible(QPUBackend(0), :riscv64)    == true
        @test _arch_compatible(QPUBackend(0), :mips)       == true

        # FPGA: x86_64 and aarch64 (PCIe)
        @test _arch_compatible(FPGABackend(0), :x86_64)    == true
        @test _arch_compatible(FPGABackend(0), :aarch64)   == true
        @test _arch_compatible(FPGABackend(0), :arm)       == false

        # TPU: x86_64 and aarch64
        @test _arch_compatible(TPUBackend(0), :x86_64)     == true
        @test _arch_compatible(TPUBackend(0), :aarch64)    == true
        @test _arch_compatible(TPUBackend(0), :riscv64)    == false

        # Julia/Rust/Zig/Math/Crypto/PPU — always compatible
        @test _arch_compatible(JuliaBackend(), :riscv64)   == true
        @test _arch_compatible(RustBackend("/x"), :mips)   == true
        @test _arch_compatible(ZigBackend("/x"), :arm)     == true
        @test _arch_compatible(MathBackend(0), :powerpc64) == true
        @test _arch_compatible(CryptoBackend(0), :x86_64)  == true
        @test _arch_compatible(PPUBackend(0), :aarch64)    == true
    end

    @testset "detect_gpu respects arch" begin
        # ROCm is x86_64-only; on aarch64 it should be skipped
        # We simulate by checking that detect_gpu filters by arch
        withenv("AXIOM_ROCM_AVAILABLE" => "true", "AXIOM_ROCM_DEVICE_COUNT" => "1") do
            gpu = detect_gpu()
            pi = detect_platform()
            if pi.arch === :x86_64
                @test gpu isa ROCmBackend
            else
                # On non-x86_64, ROCm should be filtered out
                @test gpu === nothing
            end
        end
    end

    @testset "select_backend platform-aware (no hardware)" begin
        # Without accelerators, should always fall back to JuliaBackend
        # regardless of platform heuristics
        b = select_backend(:inference, 100)
        @test b isa JuliaBackend
    end
end
