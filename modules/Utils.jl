__precompile__()

module Utils

export gaussian_pdf, get_nn_pairs

function gaussian_pdf(n, μ, σ)
    x = collect(1:1<<n)
    pl = @. 1 / sqrt(2pi * σ^2) * exp(-(x - μ)^2 / (2 * σ^2))
    pl / sum(pl)
end

function get_nn_pairs(n)
    pairs = []
    for inth in 1:2
        for i in inth:2:n
            push!(pairs, i=>i % n + 1)
        end
    end
    pairs
end



end