include("INCLUDEME.jl")

using WFLearning

function main(n; nbatch=20, epochs=200, learning_rate=0.1)
    pmap(nlayers->task(n, nlayers, nbatch, epochs, learning_rate), 2:20)
end

# main(6, 5, nbatch=10, nsamples=20, epochs=2, learning_rate=0.1)
