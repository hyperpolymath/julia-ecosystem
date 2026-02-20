module ProvenCryptoCUDAExt
using ..ProvenCrypto, CUDA

ProvenCrypto.cuda_available() = true

function ProvenCrypto.create_cuda_backend()
    dev = CUDA.device()
    cc = CUDA.capability(dev)
    return ProvenCrypto.CUDABackend(dev, CUDA.has_tensor_cores(dev), VersionNumber(cc.major, cc.minor))
end

function ProvenCrypto.backend_lattice_multiply(backend::ProvenCrypto.CUDABackend, args...)
    # CUDA-specific implementation
end
end
