__precompile__()

module Circuit

using Compat
using Yao, Yao.Blocks, Yao.LuxurySparse
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
layer(::Val{:first}) = rollrepeat(chain(Rx(), Rz()))
layer(::Val{:last}) = rollrepeat(chain(Rz(), Rx()))
layer(::Val{:mid}) = rollrepeat(chain(Rz(), Rx(), Rz()))


(x::QCBM)(args...) = x.circuit(args...)

function (x::QCBM{N})(nbatch::Int=1) where N
    apply!(zero_state(N, nbatch), x.circuit)
end

function print_block(io::IO, qcbm::QCBM)
    printstyled(io, "QCBM"; bold=true, color=:red)
end

# forward some composite block's methods
dispatch!(f::Function, qcbm::QCBM, params...) = (dispatch!(f, qcbm.circuit, params...); qcbm)
dispatch!(qcbm::QCBM, params...) = (dispatch!(qcbm.circuit, params...); qcbm)

mat(qcbm::QCBM) = mat(qcbm.circuit)
apply!(r::AbstractRegister, qcbm::QCBM) = apply!(r, qcbm.circuit)
blocks(qcbm::QCBM) = blocks(qcbm.circuit)

function initialize!(qcbm::QCBM)
    params = 2pi * rand(nparameters(qcbm))
    dispatch!(qcbm, params)
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

loss(qcbm::QCBM, kernel, ptrain) = Kernels.loss(abs2.(statevec(qcbm())), kernel, ptrain)




############
# RotBasis
############

import Yao.Blocks: _make_rot_mat, nparameters, parameters
export RotBasis, rotbasis

mutable struct RotBasis{T} <: PrimitiveBlock{1, Complex{T}}
    theta::T
    phi::T

    function RotBasis(theta::T, phi::T) where T
        new{T}(theta, phi)
    end
end

rotbasis(::Type{T}) where T = RotBasis(zero(T), zero(T))
rotbasis() = rotbasis(Float64)

copy(x::RotBasis) = RotBasis(x.theta, x.phi)

function mat(x::RotBasis{T}) where T
    R1 = _make_rot_mat(IMatrix{2, Complex{T}}(), mat(Z), -x.phi)
    R2 = _make_rot_mat(IMatrix{2, Complex{T}}(), mat(Y), -x.theta)
    R2 * R1
end

function dispatch!(R::RotBasis{T}, theta, phi) where T
    R.theta = theta
    R.phi = phi
    R
end

parameters(x::RotBasis) = (x.theta, x.phi)
nparameters(::Type{<:RotBasis}) = 2

function print_block(io::IO, x::RotBasis)
    println(io, "rot basis:")
    println(io, "  theta: ", x.theta)
    print(io,   "  phi:   ", x.phi)
end






end
