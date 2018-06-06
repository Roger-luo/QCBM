using Yao
using Yao.Intrinsics

"""
    binary_basis(geometry...) -> BitArray

Return bases in reshaped binary form (e.g. binary images)
"""
binary_basis(geometry...) = reshape(bitarray(geometry|>prod|>basis|>collect, num_bit=prod(geometry)), (geometry..., :))

# a sample is a bar or a stripe.
is_bar(mat::AbstractMatrix) = sum(abs.(diff(mat, 1)), (1, 2))[] == 0
is_stripe(mat::AbstractMatrix) = sum(abs.(diff(mat, 2)), (1, 2))[] == 0
is_bs(mat::AbstractMatrix) = is_bar(mat) || is_stripe(mat)

"""
    barstripe_pdf(M::Int, N::Int) -> Vector{Float64}

get bar and stripes PDF
"""
function barstripe_pdf(M::Int, N::Int)
    bss = binary_basis(M, N)
    [bss[:,:,i] |> is_bs for i in indices(bss, 3)] |> normalize
end

"""
    bs_configs(M::Int, N::Int) -> BitArray{3}

the total configuration space for bar and stripes.
"""
function bs_configs(M::Int, N::Int)
    bss = binary_basis(M, N)
    slicedim(bss, 3, [bss[:,:,i] |> is_bs for i in indices(bss, 3)])
end
