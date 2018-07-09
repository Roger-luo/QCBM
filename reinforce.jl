using Compat
using Compat.Test

using Yao
using Yao.Zoo
using Yao.Blocks

witness_vec = zeros(1<<4)
witness_vec[[0, 3, 5, 10, 12, 15].+1] = 1
score_func(out_probs::Vector) = witness_vec'*out_probs

n = 4
depth = 4
pairs = [1=>2, 3=>4, 2=>3, 4=>1]
generator = diff_circuit(n, depth, pairs)

loss_func = () -> -score_func(apply!(zero_state(nqubits(witness_vec)), generator) |> probs)

import Base: gradient
function gradient(rots::Vector{<:RotationGate})
    ptb = perturb(loss_func, rots, π/2)
    (ptb[:,1] - ptb[:,2])/2
end

function train(generator::AbstractBlock, g_learning_rate::Real, niter::Int)
    rots = collect_rotblocks(generator)
    for i in 1:niter
        ggrad = gradient(rots)
        dispatch!(-, generator, ggrad.*g_learning_rate)
        if i%5==1 println("Step $i, loss = $(loss_func())") end
    end
end

train(generator, 0.1, 100)

using PyPlot
pl = apply!(zero_state(n), generator)|>probs
bar(0:1<<n-1, pl)
ylabel("p", fontsize=14)
savefig("/home/leo/jcode/Yao.jl/docs/src/assets/figures/trained22bs.png", dpi=300)

# biggest 3 indices
using PyPlot
using Yao.Intrinsics
configs = (pl |> sortperm)[end-2:end] .- 1 |> bitarray(n)

function show_configs(configs)
    M = size(configs, 2)
    for c in 1:M
        subplot(100+M*10+c)
        imshow(reshape(configs[:,c], 2, 2), vmin=0, vmax=1)
        axis("off")
    end
end

#show_configs(configs)
figure(figsize=(6,2))
show_configs([0, 3, 5, 10, 12, 15] |> bitarray(n))
savefig("/home/leo/jcode/Yao.jl/docs/src/assets/figures/22bs.png", dpi=300)
