using Compat
using Compat.Test

using Yao
using Yao.Zoo
using Yao.Blocks

using QCGANS, GradOptim

# TODO using an ideal discriminator!
witness_vec = zeros(1<<4)
witness_vec[[0, 3, 5, 10, 12, 15].+1] = 1
#loss(reg::DefaultRegister) = - witness_vec .* (reg |> probs)
using StatsBase
loss(reg::DefaultRegister) = kldivergence(witness_vec, (reg |> probs))

function vgradient(witness_vec::Vector, circuit::AbstractBlock, reg0::AbstractRegister, rots::Vector{RotationGate})
    loss_func = () -> -sum(witness_vec.*(copy(reg0) |> copy |> circuit |> probs))
    ptb = perturb(loss_func, rots, π/2)
    (ptb[:,1] - ptb[:,2])/2
end

using BenchmarkTools

n = 4
depth = 4
pairs = [1=>2, 2=>3, 3=>1]
generator = diff_circuit(n, depth, pairs)
rots = collect_rotblocks(generator)

T = randn(1<<n,1<<n)
T = T' + T
Trank = ndims(T)
witness_vec = T*(zero_state(4) |> generator |> probs)
gexact = vgradient(witness_vec, generator, zero_state(n), rots)*Trank
T[1:1<<n+1:1<<(2*n)] .= 0

function loss_func()
    p = zero_state(nqubits(witness_vec)) |> generator |> probs
    -sum(p'*T*p)
end

# gnum = num_gradient(loss_func, rots)

function train(generator::AbstractBlock, g_learning_rate::Real, niter::Int)
    rots = collect_rotblocks(generator)
    for i in 1:niter
        ggrad = vgradient(T*(zero_state(n) |> generator |> probs), generator, zero_state(n), rots)*2
        dispatch!(+, generator, -ggrad.*g_learning_rate)
        println("loss = $(loss_func())")
    end
end

train(generator, 0.1, 200)

using PyPlot
bar(0:1<<n-1, apply!(zero_state(n), generator)|>probs)
