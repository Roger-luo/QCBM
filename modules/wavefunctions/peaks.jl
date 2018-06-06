using Yao

"""
GHZ wave function.
"""
function ghz(num_bit::Int; x::DInt=DInt(0))
    v = zeros(DefaultType, 1<<num_bit)
    v[x+1] = 1/sqrt(2)
    v[flip(x, bmask(1:num_bit))+1] = 1/sqrt(2)
    return v
end

"""
onehot wave function.
"""
function onehot(num_bit::Int, x::DInt)
    v = zeros(ComplexF64, 1<<num_bit)
    v[x+1] = 1
    return v
end
