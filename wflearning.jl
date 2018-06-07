include("INCLUDEME.jl")

using Yao, Circuit, UnicodePlots, GradOptim, Utils, ArgParse
import Kernels

function train!(qcbm::QCBM{N}, target0::AbstractRegister, optim; iter=10, monitor=100) where N
    initialize!(qcbm)
    kernel = Kernels.RBFKernel(nqubits(qcbm), [2.0], false)
    rot = roll(N, rotbasis())
    circuit = chain(qcbm, rot)

    history = Float64[]
    fedility = Float64[]

    for i = 1:iter
        dispatch!(rot, pi * rand(2 * N))
        target = rot(copy(target0))

        ptrain = abs2.(statevec(target))
        grad = gradient(qcbm, kernel, ptrain)

        if i % monitor == 0
            curr_loss = loss(qcbm, kernel, ptrain)
            curr_fedility = abs.(dot(statevec(qcbm()), statevec(target)))
            push!(history, curr_loss)
            push!(fedility, curr_fedility)
            println(i, " step, loss = ", curr_loss)
            println("fedility: ", curr_fedility)
        end
        # Warn: we need a primitive block to enable
        # BLAS here.
        params = parameters(qcbm)
        update!(params, grad, optim)
        dispatch!(qcbm, params)
    end
    history, fedility
end

function train(n, iter, monitor)
    target = register(bit"1"^n) + register(bit"0"^n)

    qcbm = QCBM{n, 10}(get_nn_pairs(n))
    optim = Adam(lr=0.1)
    his, fdl = train!(qcbm, target, optim, iter=iter, monitor=monitor)

    display(lineplot(his, title = "loss"))
    display(lineplot(fdl, title = "fidelity"))
end