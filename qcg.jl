include("INCLUDEME.jl")
using Compat
using Compat.Test

using Yao
using Yao.Zoo
using Yao.Blocks

using QCGANS, GradOptim

"""
gradient of v-statistics.
"""
function vstat_grad(witness::Vector, circuit::AbstractBlock, reg0::AbstractRegister, gates::Vector{RotationGate})
    gradient((rpos, rneg)->((rpos.state - rneg.state)'*witness)[], circuit, reg0, gates)
end

function train(qcg::QCGAN{N}, g_learning_rate::Real, d_learning_rate::Real, niter::Int, ninner::Int) where N
    for i in 1:niter
        ggrad, dgrad = gradient(qcg)
        dispatch!(+, qcg.generator, -ggrad.*g_learning_rate)
        for j = 1:ninner
            ggrad, dgrad = gradient(qcg)
            dispatch!(-, qcg.discriminator, -dgrad.*d_learning_rate)
        end
        print("dist = $(tracedist(qcg))")
        println("loss = $(loss(qcg)())")
    end
end

function train(qcg::QCGAN{N}, optim, optim_inner, niter::Int, ninner::Int) where N
    for i in 1:niter
        ggrad, dgrad = gradient(qcg)
        params = parameters(qcg.generator)
        update!(params, ggrad, optim)
        dispatch!(qcg.generator, params)
        for j = 1:ninner
            ggrad, dgrad = gradient(qcg)
            params = parameters(qcg.discriminator)
            update!(params, -dgrad, optim_inner)
            dispatch!(qcg.discriminator, params)
        end
        print("dist = $(tracedist(qcg)), ")
        println("loss = $(loss(qcg)())")
    end
end

function train!(qcg::QCGAN{N}, optim, niter::Int) where N
    for i in 1:niter
        ggrad, dgrad = gradient(qcg)
        params = parameters(qcg.circuit)
        update!(params, vcat(ggrad, -dgrad), optim)
        dispatch!(qcg.circuit, params)
        print("dist = $(tracedist(qcg)), ")
        println("loss = $(loss(qcg)())")
    end
end

using BenchmarkTools
pairs_gen = [1=>2, 2=>3]
pairs_dis = [1=>2, 2=>3, 3=>4]
qcg = qcgan(uniform_state(3), 5, 5, pairs_gen, pairs_dis)
optim = iRprop(2.0, 0.5);
optim_inner = Adam(lr=0.1);
train(qcg, optim, optim_inner, 200, 1)
train(qcg, 0.0, 0.5, 20, 1)
optim = iRProp(2.0, 0.5)
train!(qcg, optim, 200)

reg = rand_state(3)
@code_warntype apply!(reg, qcg.generator)
