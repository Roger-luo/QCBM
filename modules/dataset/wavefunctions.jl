using Yao

######################### CONTINUOUS ################################
uniform_state(num_bit) = register(ones(Complex128, 1<<num_bit)/sqrt(1<<num_bit))

######################### PEAKS ################################
"""
GHZ wave function.
"""
function ghz(num_bit::Int; x::Int=0)
    v = zeros(DefaultType, 1<<num_bit)
    v[x+1] = 1/sqrt(2)
    v[flip(x, bmask(1:num_bit))+1] = 1/sqrt(2)
    return v
end

"""
onehot wave function.
"""
function onehot(num_bit::Int, x::Int)
    v = zeros(ComplexF64, 1<<num_bit)
    v[x+1] = 1
    return v
end
