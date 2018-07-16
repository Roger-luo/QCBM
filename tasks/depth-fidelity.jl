# We explore the depth and fidelity here

include("../INCLUDEME.jl")

using WFLearning

function main(n; nbatch=20, epochs=200, learning_rate=0.1)
    pmap(nlayers->task("depth.jld2", n, nlayers, nbatch, epochs, learning_rate), 2:20)
end

main(6, nbatch=10, epochs=200, learning_rate=0.1)
