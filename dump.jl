include("INCLUDEME.jl")

using Yao, JLD2, Compat
using Yao.Blocks
# using Yao, Circuit, UnicodePlots, GradOptim, Utils, JLD2
import Kernels

# n = 6
# qcbm = QCBM{n, 10}(get_nn_pairs(n))
# initialize!(qcbm)

layer(::Val{:first}) = roll(chain(Rx(), Rz()))
layer(::Val{:last}) = roll(chain(Rz(), Rx()))
layer(::Val{:mid}) = roll(chain(Rz(), Rx(), Rz()))

c = layer(Val(:first))(6)

dispatch!(c, [i for i=1:12])

output = statevec(apply!(register(bit"000000"), c))
@save "data.jld" output
