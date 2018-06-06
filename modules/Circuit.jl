module Circuit

using Compat
using Yao, Yao.Blocks
import Yao.Blocks: dispatch!, blocks, mat, apply!, print_block

export QCBM, initialize!, layer


struct QCBM{N, NL, CT, T} <: CompositeBlock{N, T}
    circuit::CT

    function QCBM{N, NL}(pairs) where {N, NL}
        circuit = chain(N)

        push!(circuit, layer(:first))

        for i = 1:(NL - 1)
            push!(circuit, cache(entangler(pairs)))
            push!(circuit, layer(:mid))
        end

        push!(circuit, cache(entangler(pairs)))
        push!(circuit, layer(:last))

        new{N, NL, typeof(circuit), datatype(circuit)}(circuit)
    end
end

function entangler(pairs)
    chain(
        control([ctrl, ], target=>X) for (ctrl, target) in pairs
    )
end

layer(x::Symbol) = layer(Val(x))
layer(::Val{:first}) = roll(chain(Rx(), Rz()))
layer(::Val{:last}) = roll(chain(Rz(), Rx()))
layer(::Val{:mid}) = roll(chain(Rz(), Rx(), Rz()))


(x::QCBM)(args...) = x.circuit(args...)

function (x::QCBM{N})(nbatch::Int=1) where N
    x(zero_state(N, nbatch))
end

function print_block(io::IO, qcbm::QCBM)
    printstyled(io, "QCBM"; bold=true, color=:red)
end

# forward some composite block's methods
dispatch!(f::Function, qcbm::QCBM, params...) = (dispatch!(f, qcbm.circuit, params...); qcbm)
dispatch!(f::Function, qcbm::QCBM, params::Vector) = (dispatch!(f, qcbm.circuit, params); qcbm)

mat(qcbm::QCBM) = mat(qcbm.circuit)
apply!(r::AbstractRegister, qcbm::QCBM) = apply!(r, qcbm.circuit)
blocks(qcbm::QCBM) = blocks(qcbm.circuit)

function initialize!(qcbm::QCBM)
    params = 2pi * rand(nparameters(qcbm))
    dispatch!(qcbm, params)
end


end