include("GroverSearch.jl")
# if basis[abs(locs)] == locs>0?1:0, then flip the sign.
inference_oracle(locs) = control(locs[1:end-1], abs(locs[end])=>Z)

################################################
#                Doing Inference               #
################################################
target_space(oracle) = real(Diagonal(mat(oracle)).diag) .< 0

function inference(psi::AbstractRegister, evidense::Vector{Int}, num_iter::Int)
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

# the second version

function test_inference()
    # test inference
    num_bit = 12
    psi0 = rand_state(num_bit)
    #psi0 = uniform_state(num_bit)
    evidense = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #evidense = collect(1:num_bit)

    # the desired subspace
    basis = collect(UInt, 0:1<<num_bit-1)
    subinds = indices_with(num_bit, abs.(evidense), UInt.(evidense.>0), basis)

    v_desired = statevec(psi0)[subinds+1]
    p = norm(v_desired)^2
    v_desired[:] ./= sqrt(p)

    # search the subspace
    num_iter = num_grover_step(p)
    println("Estimated num_step = ", pi/4/sqrt(p))
    psi = inference(psi0, evidense, num_iter)
    println((psi.state[subinds+1]'*v_desired) |> abs2)
end

test_inference()
