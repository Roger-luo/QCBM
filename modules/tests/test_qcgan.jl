using Compat.Test
include("../qcgan.jl")

@testset "quantum circuit gan - opdiff" begin
    N = 3
    target = rand_state(N)
    pairs_gen = [1=>2, 2=>3]
    pairs_dis = [1=>2, 2=>3, 3=>4]
    qcg = qcgan(target, 2, 2, pairs_gen, pairs_dis)
    op_expect = ()->expect(qcg.witness_op, apply!(copy(qcg.reg0), qcg.circuit))
    ggrad = opgrad(op_expect, qcg.grots)
    dgrad = opgrad(op_expect, qcg.drots)
    println(ggrad)
    @test isapprox(ggrad, num_gradient(op_expect, qcg.grots), atol=1e-4)
    @test isapprox(dgrad, num_gradient(op_expect, qcg.drots), atol=1e-4)
end

@testset "quantum circuit gan" begin
    N = 3
    target = rand_state(N)
    pairs_gen = [1=>2, 2=>3]
    pairs_dis = [1=>2, 2=>3, 3=>4]
    qcg = qcgan(target, 2, 2, pairs_gen, pairs_dis)
    ggrad, dgrad = gradient(qcg)
    nggrad= num_gradient(loss(qcg), qcg.grots)
    ndgrad= num_gradient(loss(qcg), qcg.drots)
    println(ndgrad)
    @test isapprox(ggrad, nggrad, atol=1e-3)
    @test isapprox(dgrad, ndgrad, atol=1e-3)
end
