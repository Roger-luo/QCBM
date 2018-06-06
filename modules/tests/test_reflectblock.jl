using Compat.Test
include("../ReflectBlock.jl")

@testset "Reflect" begin
    reg0 = rand_state(3)
    mirror = randn(1<<3)*im; mirror[:]/=norm(mirror)
    rf = Reflect(mirror)
    reg = copy(reg0)
    apply!(reg, rf)

    v0, v1 = vec(reg.state), vec(reg0.state)
    @test rf.state'*v0 ≈ rf.state'*v1
    @test v0-rf.state'*v0*rf.state ≈ -(v1-rf.state'*v1*rf.state)
end
