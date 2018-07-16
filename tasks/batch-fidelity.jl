# We explore the batch and fidelity here

@everywhere include("../INCLUDEME.jl")

@everywhere using WFLearning

@everywhere function main(n, nlayers; epochs=200, learning_rate=0.1)
    pmap(nbatch->task("batch", n, nlayers, nbatch, epochs, learning_rate), 10:10:50)
end

main(6, 5, epochs=200, learning_rate=0.1)
