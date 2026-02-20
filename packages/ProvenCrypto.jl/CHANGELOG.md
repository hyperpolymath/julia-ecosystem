# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/semver/1.0.0.html).

---

## [Unreleased]
### Added
- Placeholder for future changes.

---

## [0.1.1] - 2026-01-27
### Fixed
- **Backend Detection**: Resolved duplicate `cuda_available()` method definition, ensuring GPU extensions load correctly.
- **Precompilation**: Eliminated method overwriting errors during module precompilation.
- **Fallback Logic**: Improved automatic fallback to CPU with SIMD/multithreading support when GPU backends are unavailable.

### Changed
- **Dependencies**: Added `Preferences.jl` for user-configurable fallback behavior (e.g., force CPU-only mode).
- **Documentation**: Clarified installation instructions for GPU backends (CUDA/ROCm/Metal) in `README.md`.
- **Version Bump**: Updated to `0.1.1` to reflect fixes and compatibility improvements.

### Added
- **Automatic SIMD Detection**: Robust detection of AVX2/AVX512/NEON/SVE for CPU fallback.
- **Thread Auto-Detection**: Automatically leverages all available CPU threads (`Threads.nthreads()`).
- **Extension System**: Formalized `ext/` directory for optional GPU backends (CUDA, ROCm, Metal, oneAPI).

### Security
- **Isolation**: Ensured GPU extensions respect **rootless container** environments (compatible with **svalinn/vordr**, **nerdctl**, and **SELinux/AppArmor**).
- **Zero Trust Alignment**: GPU backends are **optional** and **non-proprietary**, loading only when explicitly available.

### Deployment
- **Compatibility**: Tested with **Julia 1.9+**, aligning with **ReScript/Hypatia** toolchains.
- **Containerization**: Validated for **OCI-standard** deployments (e.g., **Chainguard images** via **podman**).

---

## [0.1.0] - 2026-01-25
### Added
- **Initial Release**: Core cryptographic backend abstraction for:
  - NVIDIA CUDA (Tensor cores).
  - AMD ROCm (Matrix cores).
  - Apple Metal (Neural Engine).
  - Intel oneAPI (NPU/GPU).
  - CPU SIMD (AVX2, AVX-512, NEON, SVE).
- **Extensions**: Modular support for optional hardware backends.
- **Formal Methods**: Weak dependency on `SMTLib.jl` for verification integration.
- **Project Structure**: `src/`, `ext/`, and `Project.toml` for modular deployment.

### Notes
- Designed for integration with **Hypatia** (CI/CD) and **ReScript** projects.
- Defaults to **CPU fallback** in **Software-Defined Perimeter (SDP)** environments (e.g., behind **Cloudflare Zero Trust**).
- Container-ready: Compatible with **rootless containers** and **WASM proxies**.

---
