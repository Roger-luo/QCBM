using Compat
using Yao, Yao.Blocks

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

entangler(pairs) = chain(control([ctrl, ], target=>X) for (ctrl, target) in pairs)

layer(x::Symbol) = layer(Val(x))
layer(::Val{:first}) = rollrepeat(chain(Rx(0.0), Rz(0.0)))
layer(::Val{:last}) = rollrepeat(chain(Rz(0.0), Rx(0.0)))
layer(::Val{:mid}) = rollrepeat(chain(Rz(0.0), Rx(0.0), Rz(0.0)))

import Yao.Blocks: apply!, blocks
apply!(r::AbstractRegister, blk::QCBM) = apply!(r, blk.circuit)
blocks(blk::QCBM) = blocks(blk.circuit)

function print_block(io::IO, qcbm::QCBM)
    printstyled(io, "QCBM"; bold=true, color=:red)
end

function initialize!(qcbm::QCBM)
    params = 2pi * rand(nparameters(qcbm))
    dispatch!(qcbm, params)
end

function (x::QCBM{N})(nbatch::Int=1) where N
    apply!(zero_state(N, nbatch), x.circuit)
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
    for each_line in blocks(layer)
        for each in blocks(each_line)
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


qcbm = QCBM{4, 5}([1=>2, 2=>3, 3=>4, 4=>1])
nparameters(qcbm)
initialize!(qcbm)
dispatch!(-, qcbm, 1:64)
