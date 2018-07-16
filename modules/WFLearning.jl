module WFLearning

using Compat
using Yao, Kernels, Utils, Yao.Zoo, GradOptim, UnicodePlots, JLD2

################################################################################
#                           Define Circuit

using Yao, Yao.Blocks, Yao.LuxurySparse, Yao.Zoo
import Yao.Blocks: blocks, apply!, print_block
import Base: gradient
import Kernels: loss

"""
    There are many born machines, but this one is mine.
"""
struct OhMyBM{N, NL, CT, T, Basis <: AbstractBlock} <: CompositeBlock{N, T}
    circuit::CT
    basis::Basis

    function OhMyBM{N, NL}(pairs) where {N, NL}
        circuit = diff_circuit(N, NL, pairs)
        basis = rot_basis(N)
        new{N, NL, typeof(circuit), datatype(circuit), typeof(basis)}(circuit, basis)
    end
end

# Basic Components
entangler(pairs) = chain(control(ctrl, target=>X) for (ctrl, target) in pairs)
layer(x::Symbol) = layer(Val(x))
layer(::Val{:first}) = rollrepeat(chain(Rx(0.0), Rz(0.0)))
layer(::Val{:last}) = rollrepeat(chain(Rz(0.0), Rx(0.0)))
layer(::Val{:mid}) = rollrepeat(chain(Rz(0.0), Rx(0.0), Rz(0.0)))

# This is just some shortcuts
apply_zero(c) = apply!(zero_state(nqubits(c)), c)
initialize!(c) = dispatch!(c, 2pi * rand(nparameters(c)))
change_basis!(mine) = dispatch!(mine.basis, vec(randpolar(nqubits(mine))))

# Right, let's forward some APIs
function apply!(r::AbstractRegister, mine::OhMyBM)
    apply!(r, mine.circuit)
    apply!(r, mine.basis)
end

blocks(mine::OhMyBM) = blocks(mine.circuit)

function print_block(io::IO, qcbm::OhMyBM)
    printstyled(io, "OhMyBM"; bold=true, color=:red)
end

################################################################################
#                               Gradients
function gradient(c::OhMyBM, rots::Vector, prob, kernel, ptrain)
    map(gate->gradient(c, gate, prob, kernel, ptrain), rots)
end

function gradient(c::OhMyBM, gate, prob, kernel, ptrain)
    dispatch!(+, gate, pi / 2)
    prob_pos = probs(apply_zero(c))

    dispatch!(-, gate, pi)
    prob_neg = probs(apply_zero(c))

    dispatch!(+, gate, pi / 2)

    grad_pos = Kernels.expect(kernel, prob, prob_pos) - Kernels.expect(kernel, prob, prob_neg)
    grad_neg = Kernels.expect(kernel, ptrain, prob_pos) - Kernels.expect(kernel, ptrain, prob_neg)
    grad_pos - grad_neg
end

loss(c::OhMyBM, kernel, ptrain) = Kernels.loss(probs(apply_zero(c)), kernel, ptrain)

################################################################################

function train!(c::OhMyBM, psi, optim; learning_rate=0.1, epochs=200, nbatch=10)
    initialize!(c)
    kernel = Kernels.RBFKernel(nqubits(c), [0.25], false)
    history = Float64[]

    itr = Iterators.filter(x->(hasparameter(x) && isprimitive(x)), BlockTreeIterator(:DFS, c))
    rots = collect(itr)
    batch_grad = zeros(nparameters(c))
    for i = 1:epochs
        fill!(batch_grad, 0)
        curr_loss = 0

        for m = 1:nbatch
            change_basis!(c)
            ptrain = probs(apply!(copy(psi), c.basis))
            prob = probs(apply_zero(c))
            curr_loss += loss(c, kernel, ptrain)
            grad = gradient(c, rots, prob, kernel, ptrain)
            batch_grad += grad
        end
        batch_grad ./= nbatch
        curr_loss /= nbatch
        push!(history, curr_loss)
        println("epoch ", i, " loss = ", curr_loss)

        params = collect(parameters(c))
        update!(params, batch_grad, optim)
        dispatch!(c, params)
    end
    history
end

struct TrainData
    n::Int
    nlayers::Int
    nbatch::Int
    epochs::Int
    learning_rate::Float64
    history::Vector{Float64}
    fidelity::Float64
end

function task(filename, n, nlayers, nbatch, epochs, learning_rate)
    # set up machine
    bm = OhMyBM{n, nlayers}([(i%n + 1)=>((i+1)%n + 1) for i = 1:n])
    initialize!(bm)
    # set up optimizer
    optim = Adam(lr=0.1);

    # set target state
    # r = register(bit"0"^n) + register(bit"1"^n)
    r = rand_state(n)
    normalize!(r)
    his = train!(bm, r, optim, nbatch=nbatch, epochs=epochs, learning_rate=learning_rate)
    fs = fidelity(apply!(zero_state(nqubits(bm)), bm.circuit), r)
    data = TrainData(n, nlayers, nbatch, epochs, learning_rate, his, fs[1])
    jldopen("data/$filename.jld2", "a+") do file
        file["$n/$nlayers/$nbatch/$epochs/$learning_rate"] = data
    end
end

export task

end
