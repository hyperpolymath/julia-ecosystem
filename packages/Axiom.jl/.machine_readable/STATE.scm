;; SPDX-License-Identifier: PMPL-1.0-or-later
;; STATE.scm - Current project state for Axiom.jl

(define project-state
  `((metadata
      ((version . "1.0.0")
       (schema-version . "1")
       (created . "2026-01-10T13:47:48+00:00")
       (updated . "2026-02-20T00:00:00+00:00")
       (project . "Axiom.jl")
       (repo . "Axiom.jl")))

    (current-position
      ((phase . "Active Development")
       (overall-completion . 78)
       (working-features
         ("tensor-types" "dense-layer" "conv2d-layer" "activations"
          "normalization-layers" "pooling-layers" "sequential-pipeline"
          "optimizers-sgd-adam-adamw-rmsprop" "loss-functions"
          "training-loop-zygote-autograd" "ensure-macro" "axiom-macro" "prove-macro-heuristic-smt"
          "property-checking" "proof-certificates-json-text" "proof-serialization"
          "smtlib-solver-interface" "julia-backend-all-ops"
          "rust-ffi-all-ops" "zig-ffi-all-ops"
          "gpu-hooks-all-ops-cuda-rocm-metal"
          "pytorch-import-export-layernorm" "model-metadata"
          "proof-export-lean-coq-isabelle-real-tactics"
          "proof-import-lean-coq-isabelle"
          "backend-aware-forward-dispatch"
          "data-loader" "benchmarks"
          "coprocessor-dispatch-15-backends"
          "self-healing-fallback"
          "env-based-accelerator-detection"
          "runtime-diagnostics"
          "capability-reporting"
          "vpu-qpu-crypto-backend-types"
          "huggingface-safetensors-loader"
          "huggingface-gpt2-vit-resnet-builders"
          "fold-batchnorm" "fold-constants" "dead-code-elimination"
          "aggressive-optimization-pass"
          "resource-aware-backend-dispatch"
          "device-resources-scoring"
          "proof-type-inference-static-empirical-formal"))))

    (route-to-mvp
      ((milestones
        ((v0.1.0 . ((items . ("Core layers" "Julia backend" "Basic verification"
                               "Training loop" "Rust FFI" "Zig FFI"
                               "GPU extension hooks" "SMTLib integration"
                               "PyTorch interop" "Proof certificates"))
                    (status . "mostly-complete")))
         (v0.2.0 . ((items . ("Enable @prove macro" "Real autograd via Zygote"
                               "Complete GPU extensions" "Rust end-to-end dispatch"
                               "Build zig .so artifact" "Complete HuggingFace converters"
                               "Compile optimizations"))
                    (status . "mostly-complete")
                    (completed . ("Complete HuggingFace converters"
                                  "Compile optimizations"
                                  "Enable @prove macro"
                                  "Real autograd via Zygote"
                                  "Complete GPU extensions"))))
         (v1.0.0 . ((items . ("Proof assistant import" "Real proof translation"
                               "Production model save/load" "Full HuggingFace model zoo"
                               "Performance parity benchmarks" "Security audit"
                               "RSR compliance"))
                    (status . "not-started")))))))

    (blockers-and-issues
      ((critical
         ())
       (high
         (("Zig library not compiled" .
           "No .so artifact committed. Users must run `just build-zig` manually.")))
       (medium
         (("Model save/load broken" .
           "save_model uses repr(), load_model! is a no-op. Needs JLD2/BSON.")
          ("Coprocessor stubs" .
           "TPU/NPU/DSP/PPU/Math/FPGA/VPU/QPU/Crypto backends are env-detection stubs only.")
          ("Mixed precision incomplete" .
           "MixedPrecisionWrapper exists but only basic float16/float32 casting.")))
       (low ()))))

    (critical-next-actions
      ((immediate
         ("Build zig .so artifact and commit"
          "Fix SPDX headers to PMPL-1.0-or-later"
          "Production model save/load via JLD2"))
       (this-week
         ("Implement real coprocessor backends (TPU/NPU at minimum)"
          "Complete mixed precision support"
          "Performance parity benchmarks"))
       (this-month
         ("Security audit"
          "Full RSR compliance"
          "v1.0.0 release preparation"))))

    (session-history
      ((session-2026-02-20a
         ((actions . ("Full codebase audit" "Created TOPOLOGY.md"
                      "Moved SCM files to .machine_readable/"
                      "Updated STATE.scm with accurate status"
                      "Rewrote ECOSYSTEM.scm (was corrupted with badge URL)"
                      "Updated META.scm with architecture decisions"
                      "Replaced Nix Justfile with Julia-native recipes"))
          (agent . "claude-opus-4-6")))
       (session-2026-02-20b
         ((actions . ("Added VPU/QPU/Crypto backend types to abstract.jl"
                      "Created backend infrastructure for 13 COMPUTE packages"
                      "Updated TOPOLOGY.md with coprocessor dispatch section"
                      "Updated RSR compliance from 20% to 80%"
                      "Moved repo into julia-ecosystem monorepo"
                      "Pushed to GitHub and GitLab"))
          (agent . "claude-opus-4-6")))
       (session-2026-02-20c
         ((actions . ("Proof certificates: real proof-type inference, JSON export"
                      "HuggingFace: SafeTensors loader, GPT-2/ViT/ResNet builders"
                      "PyTorch extension: LayerNorm import/export, fallback handler"
                      "Compile optimizations: fold_batchnorm, fold_constants, dead code elimination"
                      "Resource-aware dispatch: DeviceResources, select_best_backend, resource_report"
                      "Fixed ROCmBackend definition ordering (moved to abstract.jl)"
                      "204 tests passing, pushed to GitHub and GitLab"))
          (agent . "claude-opus-4-6")))
       (session-2026-02-20d
         ((actions . ("Tier 1: Exported @prove macro and autograd functions (were already implemented)"
                      "Tier 1: Backend-aware forward dispatch for Dense/Conv2d/BatchNorm"
                      "Tier 1: Full GPU extensions - conv2d/batchnorm/pooling for CUDA/ROCm/Metal"
                      "Tier 1: Real proof tactics for Lean/Coq/Isabelle (ValidProbabilities, BoundedOutput, Monotonic, NonNegative, Lipschitz)"
                      "Tier 1: Fixed method overwriting by moving forward() from layer files to abstract.jl"
                      "Tier 1: JuliaBackend default ops for conv2d/batchnorm/pooling"
                      "204 tests passing, overall completion 63%→78%"))
          (agent . "claude-opus-4-6")))))))
