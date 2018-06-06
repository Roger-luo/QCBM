__precompile__()

module __main__

include("INCLUDEME.jl")

using Yao, Circuit, UnicodePlots, GradOptim, Utils, ArgParse
import Kernels

function train!(qcbm::QCBM, ptrain, optim; learning_rate=0.1, maxiter=100)
    initialize!(qcbm)
    kernel = Kernels.RBFKernel(nqubits(qcbm), [0.25], false)
    history = Float64[]

    for i = 1:maxiter
        grad = gradient(qcbm, kernel, ptrain)
        curr_loss = loss(qcbm, kernel, ptrain)
        push!(history, curr_loss)
        println(i, " step, loss = ", curr_loss)

        # Warn: we need a primitive block to enable
        # BLAS here.
        params = parameters(qcbm)
        update!(params, grad, optim)
        dispatch!(qcbm, params)
    end
    history
end


function train(n, maxiter)
    pg = gaussian_pdf(n, 2^5-0.5, 2^4)
    fig = lineplot(0:1<<n - 1, pg)
    display(fig)

    qcbm = QCBM{n, 10}(get_nn_pairs(n))
    optim = Adam(lr=0.1)
    his = train!(qcbm, pg, optim, maxiter=maxiter)

    display(lineplot(his, title = "loss"))
    psi = qcbm()
    p = abs2.(statevec(psi))
    p = p / sum(p)
    lineplot!(fig, p, color=:yellow, name="trained")
    display(fig)
end

function main()
    n = parse(ARGS[1])
    if length(ARGS) > 1
        nitr = parse(ARGS[2])
    else
        nitr = 50
    end
    train(n, nitr)
end

end

__main__.main()
