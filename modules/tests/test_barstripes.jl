using Compat.Test
include("BarStripes.jl")

@testset "barstripes" begin
    @test size(binary_basis(2,2)) == (2,2,16)

    ba = reshape(bitarray(7, num_bit=9), 3,3)
    @test is_bar(ba) == true
    @test is_stripe(ba) == false

    ba0 = reshape(bitarray(0, num_bit=9), 3,3)
    @test is_bar(ba0) == true
    @test is_stripe(ba0) == true
    @test is_bs(ba) == is_bs(ba0) == true

    @test reshape(bs_configs(2,2), 4, :) |> packbits == findn(barstripe_pdf(2,2))-1
end
