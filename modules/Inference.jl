include("GroverSearch.jl")
using Yao.Intrinsics
# if basis[abs(locs)] == locs>0?1:0, then flip the sign.
inference_oracle(locs) = control(locs[1:end-1], abs(locs[end])=>Z)

################################################
#                Doing Inference               #
################################################
target_space(oracle) = real(Diagonal(mat(oracle)).diag) .< 0

function inference1(psi::AbstractRegister, evidense::Vector{Int}, num_iter::Int)
    oracle = inference_oracle(evidense)(nqubits(psi))
    ts = target_space(oracle)
    rv1 = Reflect(copy(psi))
    grover = chain(oracle, rv1)
    for i in 1:num_iter
        p_target = norm(statevec(psi)[ts])^2
        println("step $i, overlap = $p_target")
        grover(psi)
    end
    psi
end

function run_inference(psi, evidense)
    # the desired subspace
    subinds = indices_with(nqubits(psi), abs.(evidense), Int.(evidense.>0))

    v_desired = statevec(psi)[subinds+1]
    p = norm(v_desired)^2
    v_desired[:] ./= sqrt(p)

    # search the subspace
    num_iter = num_grover_step(p)
    println("Estimated num_step = ", pi/4/sqrt(p))
    inference1(psi, evidense, num_iter)
    fidelity = (psi.state[subinds+1]'*v_desired) |> abs
    fidelity
end

# the second version
