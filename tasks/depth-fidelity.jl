# We explore the batch and fidelity here

@everywhere include("../INCLUDEME.jl")

@everywhere using WFLearning

@everywhere function main(n; nbatch=20, epochs=200, learning_rate=0.1)
    pmap(nlayers->task("layers", n, nlayers, nbatch, epochs, learning_rate), 5:15)
end

main(6, nbatch=20, epochs=200, learning_rate=0.1)
