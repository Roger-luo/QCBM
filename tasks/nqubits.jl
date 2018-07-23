# We explore the batch and fidelity here

@everywhere include("../INCLUDEME.jl")

@everywhere using WFLearning

ptask("nqubits-2", 8, 10, 40, 50, 0.01)

# @everywhere function main(nlayers; nbatch=10, epochs=200, learning_rate=0.1)
#     pmap(n->task("nqubits", n, nlayers, nbatch, epochs, learning_rate), 5:7)
# end
#
# main(5, epochs=200, learning_rate=0.1, nbatch=10)
