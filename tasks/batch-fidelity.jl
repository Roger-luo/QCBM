# We explore the batch and fidelity here

include("../INCLUDEME.jl")

using WFLearning

function main(n, nlayers; epochs=200, learning_rate=0.1)
    pmap(nbatch->task("batch.jld2", n, nlayers, nbatch, epochs, learning_rate), 10:10:50)
end

main(6, 5, epochs=200, learning_rate=0.1)
