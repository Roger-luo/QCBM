include("INCLUDEME.jl")

using Yao, Circuit, UnicodePlots, GradOptim, Utils
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

# parameters
n = 6
maxiter = 20

#data to learn
pg = gaussian_pdf(n, 2^5-0.5, 2^4)
fig = lineplot(0:1<<n - 1, pg)
display(fig)

# solver setup
qcbm = QCBM{n, 10}(get_nn_pairs(n))
optim = Adam(lr=0.1);

his = train!(qcbm, pg, optim, maxiter=maxiter);

# analyze result
display(lineplot(his, title = "loss"))
psi = qcbm()
p = statevec(psi) .|> abs2
lineplot!(fig, p, color=:yellow, name="trained")
display(fig)
