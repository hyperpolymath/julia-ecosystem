# SPDX-License-Identifier: MPL-2.0
module ModularMath

export Mod, value, modulus

struct Mod{T<:Integer}
    val::T
    mod::T
    Mod(v::T, m::T) where T = new{T}(mod(v, m), m)
end

value(m::Mod) = m.val
modulus(m::Mod) = m.mod

Base.:+(a::Mod, b::Mod) = a.mod == b.mod ? Mod(a.val + b.val, a.mod) : error("Moduli mismatch")
Base.:*(a::Mod, b::Mod) = a.mod == b.mod ? Mod(a.val * b.val, a.mod) : error("Moduli mismatch")

end # module
