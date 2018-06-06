using Compat.Test
include("../Inference.jl")
include("../wavefunctions/continuous.jl")

@testset "Grover Search" begin
    # alway use sorted CSC format.
    oracle = inference_oracle([2,-1,3])(3)
    v = ones(1<<3)
    v[Int(0b110)+1] *= -1
    @test mat(oracle) ≈ Diagonal(v)

    ####### Construct Grover Search Using Reflection Block
    num_bit = 12
    oracle = inference_oracle(push!(collect(Int, 1:num_bit-1), num_bit))(num_bit)

    psi = GroverSearch(oracle, 12)
    target_state = zeros(1<<num_bit); target_state[end] = 1
    @test isapprox(abs(statevec(psi)'*target_state), 1, atol=1e-3)
end
