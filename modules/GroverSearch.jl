include("ReflectBlock.jl")

#grover_step!(psi::AbstractRegister) = rv1(oracle(psi))
num_grover_step(prob::Real) = Int(round(pi/4/sqrt(prob))) - 1

function GroverSearch(oracle, num_bit::Int; psi = uniform_state(num_bit))
    uni_reflect = Reflect(uniform_state(num_bit))
    # solve a real search problem
    num_iter = num_grover_step(1.0/(1<<num_bit))

    for i in 1:num_iter
        psi = uni_reflect(oracle(psi))
    end
    psi
end

function indices_with(num_bit::Int, poss::Vector{Int}, vals::Vector{UInt}, basis::Vector{UInt})
    mask = bmask(poss)
    valmask = bmask(poss[vals.!=0])
    basis[(basis .& mask) .== valmask]
end
bmask(ibit::Vector{Int}) = reduce(+, zero(UInt), [one(UInt) << b for b in (ibit.-1)])
bmask(ibit::Int) = one(UInt) << (ibit-1)

