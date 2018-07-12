include("INCLUDEME.jl")

using Compat
using Yao, Kernels, Utils, Yao.Zoo, GradOptim, UnicodePlots, JLD2

################################################################################
#                           Define Circuit

using Yao, Yao.Blocks, Yao.LuxurySparse
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
        circuit = chain(N)

        push!(circuit, layer(:first))

        for i = 1:(NL - 1)
            push!(circuit, cache(entangler(pairs)))
            push!(circuit, layer(:mid))
        end

        push!(circuit, cache(entangler(pairs)))
        push!(circuit, layer(:last))

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

function gradient(c::OhMyBM{N, NL}, kernel, ptrain) where {N, NL}
    grad = zeros(real(datatype(c)), nparameters(c))
    p = probs(apply_zero(c))
    idx = 0
    for ilayer = 1:2:(2 * NL + 1)
        idx = grad_layer!(grad, idx, p, c, c.circuit[ilayer], kernel, ptrain)
    end
    grad
end

function grad_layer!(grad, idx, prob, qcbm, layer, kernel, ptrain)
    count = idx
    for each_line in blocks(layer)
        for each in blocks(each_line)
            gradient!(grad, count+1, prob, qcbm, each, kernel, ptrain)
            count += 1
        end
    end
    count
end

function gradient!(grad, idx, prob, qcbm, gate, kernel, ptrain)
    dispatch!(+, gate, pi / 2)
    prob_pos = probs(apply_zero(qcbm))

    dispatch!(-, gate, pi)
    prob_neg = probs(apply_zero(qcbm))

    dispatch!(+, gate, pi / 2) # set back

    grad_pos = Kernels.expect(kernel, prob, prob_pos) - Kernels.expect(kernel, prob, prob_neg)
    grad_neg = Kernels.expect(kernel, ptrain, prob_pos) - Kernels.expect(kernel, ptrain, prob_neg)
    grad[idx] = grad_pos - grad_neg
    grad
end

loss(c::OhMyBM, kernel, ptrain) = Kernels.loss(probs(apply_zero(c)), kernel, ptrain)

################################################################################

function train!(c::OhMyBM, psi, optim; learning_rate=0.1, maxiter=200, nbatch=10)
    initialize!(c)
    kernel = Kernels.RBFKernel(nqubits(bm), [0.25], false)
    history = Float64[]

    batch_grad = zeros(nparameters(c))
    for i = 1:maxiter
        fill!(batch_grad, 0)
        curr_loss = 0

        for m = 1:nbatch
            change_basis!(c)
            ptrain = probs(apply!(copy(psi), c.basis))
            curr_loss += loss(c, kernel, ptrain)
            grad = gradient(c, kernel, ptrain)
            batch_grad += grad
        end
        batch_grad ./= nbatch
        curr_loss /= nbatch
        push!(history, curr_loss)
        println(i, " step, loss = ", curr_loss)

        params = collect(parameters(c))
        update!(params, batch_grad, optim)
        dispatch!(c, params)
    end
    history
end

function main(n, nlayers)
    # set up machine
    bm = OhMyBM{n, nlayers}([(i%n + 1)=>((i+1)%n + 1) for i = 1:n])
    initialize!(bm)

    # set target state
    # r = register(bit"0"^n) + register(bit"1"^n)
    r = rand_state(n)
    normalize!(r)

    # set up optimizer
    optim = Adam(lr=0.1);

    histories = []
    fidelities = []

    for i = 1:100
        push!(histories, train!(bm, r, optim))
        push!(fidelities, fidelity(apply!(zero_state(nqubits(bm)), bm.circuit), r))
    end

    @save "wflearning.jld2" histories fidelities
end

main(6, 5)
