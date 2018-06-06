module Circuit

using Compat
using Yao, Yao.Blocks
using Kernels
import Yao.Blocks: dispatch!, blocks, mat, apply!, print_block
import Base: gradient

export QCBM, initialize!, layer, parameters, entangler, loss


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

function parameters(qcbm::QCBM{N, NL}) where {N, NL}
    params = zeros(real(datatype(qcbm)), nparameters(qcbm))
    idx = 0
    for ilayer = 1:2:(2 * NL + 1)
        idx = parameters!(params, idx, qcbm.circuit[ilayer])
    end
    params
end

function parameters!(params, idx, layer::Roller)
    count = idx
    for each_line in layer
        for each in each_line
            params[count + 1] = each.theta
            count += 1
        end
    end
    count
end

function gradient(qcbm::QCBM{N, NL}, kernel, ptrain) where {N, NL}
    prob = abs2.(statevec(qcbm()))
    grad = zeros(real(datatype(qcbm)), nparameters(qcbm))
    idx = 0
    for ilayer = 1:2:(2 * NL + 1)
        idx = grad_layer!(grad, idx, prob, qcbm, qcbm.circuit[ilayer], kernel, ptrain)
    end
    grad
end

function grad_layer!(grad, idx, prob, qcbm, layer, kernel, ptrain)
    count = idx
    for each_line in layer
        for each in each_line
            gradient!(grad, count+1, prob, qcbm, each, kernel, ptrain)
            count += 1
        end
    end
    count
end

function gradient!(grad, idx, prob, qcbm, gate, kernel, ptrain)
    dispatch!(+, gate, pi / 2)
    prob_pos = abs2.(statevec(qcbm()))

    dispatch!(-, gate, pi)
    prob_neg = abs2.(statevec(qcbm()))

    dispatch!(+, gate, pi / 2) # set back

    grad_pos = Kernels.expect(kernel, prob, prob_pos) - Kernels.expect(kernel, prob, prob_neg)
    grad_neg = Kernels.expect(kernel, ptrain, prob_pos) - Kernels.expect(kernel, ptrain, prob_neg)
    grad[idx] = grad_pos - grad_neg
    grad
end

loss(qcbm::QCBM, kernel, ptrain) = Kernels.loss(abs2.(statevec(qcbm())), kernel, ptrain)

end