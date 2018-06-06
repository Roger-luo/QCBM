include("ReflectBlock.jl")

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
