include("INCLUDEME.jl")

using Yao, Yao.Blocks, JLD2
# using Yao, Circuit, UnicodePlots, GradOptim, Utils, ArgParse, JLD2, FileIO
import Kernels

# n = 6
# qcbm = QCBM{n, 10}(get_nn_pairs(n))

@load "data.jld" output

layer(::Val{:first}) = rollrepeat(chain(Rx(), Rz()))
layer(::Val{:last}) = rollrepeat(chain(Rz(), Rx()))
layer(::Val{:mid}) = rollrepeat(chain(Rz(), Rx(), Rz()))

c = kron(6, i=>chain(Rx(), Rz()) for i = 1:6)

# c = layer(Val(:first))(6)
dispatch!(c, [i for i=1:12])
c

out = statevec(apply!(register(bit"000000"), c))

# dispatch!(qcbm, params)
# out = statevec(qcbm())
@show out ≈ output
out
