using Compat.Test

include("RotBasis.jl")

using Yao, Rot

# translation between polar angle and len-2 complex vector.
function u2polar(vec)
    ratio = slicedim(vec, 1, 2) ./ slicedim(vec, 1, 1)
    @. [atan(abs(ratio))'*2; angle(ratio)']
end

function polar2u(polar)
    theta, phi = slicedim(polar, 1, 1)', slicedim(polar, 1, 2)'
    @. [cos(theta/2) * exp(-im*phi/2); sin(theta/2) * exp(im*phi/2)]
end

# random polar basis, n-> number of basis
randpolar(n) = rand(2, n) * pi

polar = randpolar(10)
@test all(isapprox.(polar |> polar2u |> u2polar, polar))

rb = RotBasis(0.1, 0.3)
angles = randpolar(1)
psi = register(polar2u(angles))
dispatch!(rb, angles)
@test state(rb(psi)) ≈ [1, 0]
