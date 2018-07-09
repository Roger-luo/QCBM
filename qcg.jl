include("INCLUDEME.jl")
using Compat
using Compat.Test

using Yao
using Yao.Zoo
using Yao.Blocks

using QCGANS, GradOptim

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


function train_alter(qcg::QCGAN{N}, g_learning_rate::Real, d_learning_rate::Real, niter::Int) where N
    for i in 1:niter
        ggrad, dgrad = gradient(qcg)
        lossval = loss(qcg)()
        if lossval == 0
            dispatch!(+, qcg.generator, -ggrad.*g_learning_rate)
        else
            ggrad, dgrad = gradient(qcg)
            dispatch!(-, qcg.discriminator, -dgrad.*d_learning_rate)
        end
        #print("dist = $(tracedist(qcg))")
        println("loss = $(lossval)")
    end
end
# TODO using an ideal discriminator!
disc(b) = b in [0, 3, 5, 10, 12, 15]
function train(qcg::QCGAN{N}, g_learning_rate::Real, niter::Int, ninner::Int) where N
    for i in 1:niter
        ggrad, dgrad = gradient(qcg)
        dispatch!(+, qcg.generator, -ggrad.*g_learning_rate)
        print("dist = $(tracedist(qcg))")
        println("loss = $(loss(qcg)())")
    end
end


using BenchmarkTools
pairs_gen = [1=>2, 2=>3, 3=>1]
pairs_dis = [1=>2, 2=>3, 3=>4, 4=>1]
qcg = qcgan(uniform_state(3), 5, 5, pairs_gen, pairs_dis)
qcg.target.state = copy(qcg.reg0) |> qcg.generator |> state
optim = iRProp(2.0, 0.5);
optim_inner = iRProp(2.0, 0.5);
train(qcg, optim, optim_inner, 200, 20)
train(qcg, 0.2, 0.5, 200, 1)
train_alter(qcg, 0.2, 0., 200)
optim = iRProp(2.0, 0.5)
train!(qcg, optim, 200)

reg = rand_state(3)
@code_warntype apply!(reg, qcg.generator)
