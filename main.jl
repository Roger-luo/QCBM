include("load.jl")

using Yao, UnicodePlots, BenchmarkTools, GradOptim

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
