using Yao
import Base:sparse
import Yao.Blocks: PrimitiveBlock
using Yao.Registers
using Yao.Intrinsics: log2i

################################################
#              Grover Operator Block           #
################################################
# psi and oracle are needed
struct Reflect{N, T} <: PrimitiveBlock{N, T}
    state :: Vector{T}
end
Reflect(state::Vector{T}) where T = Reflect{log2i(length(state)), T}(state)
Reflect(psi::AbstractRegister) = Reflect(statevec(psi))

# NOTE: this should not be matrix multiplication based
import Yao.Blocks: apply!
function apply!(r::AbstractRegister, g::Reflect)
    @views r.state[:,:] .= 2* (g.state'*r.state) .* reshape(g.state, :, 1) - r.state
    r
end
# since julia does not allow call overide on AbstractGate.
(rf::Reflect)(reg::AbstractRegister) = apply!(reg, rf)

import Base: show
function show(io::IO, g::Reflect{N, T}) where {N, T}
    print("Reflect(N = $N")
end
