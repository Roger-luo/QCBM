include("../Inference.jl")
using Compat.Test

@testset "Inference" begin
    num_bit = 12
    psi = rand_state(num_bit)
    evidense = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    fidelity = run_inference(psi, evidense)
    println("The final fidelity F = ", fidelity)
    @test isapprox(fidelity, 1, atol=1e-2)
end
