__precompile__()

module GradOptim

using Compat
using Compat.LinearAlgebra.BLAS

export Adam, update!, iRProp

"""
    Adam(;lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8)
    update!(w,g,p::Adam)

Container for parameters of the Adam optimization algorithm used by
[`update!`](@ref).

Adam is one of the methods that compute the adaptive learning rate. It
stores accumulated gradients (first moment) and the sum of the squared
of gradients (second).  It scales the first and second moment as a
function of time. Here is the update formulas:

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g .* g
    mhat = m ./ (1 - beta1 ^ t)
    vhat = v ./ (1 - beta2 ^ t)
    w = w - (lr / (sqrt(vhat) + eps)) * mhat

where `w` is the weight, `g` is the gradient of the objective function
w.r.t `w`, `lr` is the learning rate, `m` is an array with the same
size and type of `w` and holds the accumulated gradients. `v` is an
array with the same size and type of `w` and holds the sum of the
squares of the gradients. `eps` is a small constant to prevent a zero
denominator. `beta1` and `beta2` are the parameters to calculate bias
corrected first and second moments. `t` is the update count.

If `vecnorm(g) > gclip > 0`, `g` is scaled so that its norm is equal
to `gclip`.  If `gclip==0` no scaling takes place.

Reference: [Kingma, D. P., & Ba,
J. L. (2015)](https://arxiv.org/abs/1412.6980). Adam: a Method for
Stochastic Optimization. International Conference on Learning
Representations, 1–13.

"""
mutable struct Adam
    lr::AbstractFloat
    gclip::AbstractFloat
    beta1::AbstractFloat
    beta2::AbstractFloat
    eps::AbstractFloat
    t::Int
    fstm
    scndm
end

Adam(; lr=0.001, gclip=0, beta1=0.9, beta2=0.999, eps=1e-8)=Adam(lr, gclip, beta1, beta2, eps, 0, nothing, nothing)

function update!(w, g, p::Adam)
    gclip!(g, p.gclip)
    if p.fstm===nothing; p.fstm=zeros(w); p.scndm=zeros(w); end
    p.t += 1
    BLAS.scale!(p.beta1, p.fstm)
    BLAS.axpy!(1-p.beta1, g, p.fstm)
    BLAS.scale!(p.beta2, p.scndm)
    BLAS.axpy!(1-p.beta2, g .* g, p.scndm)
    fstm_corrected = p.fstm / (1 - p.beta1 ^ p.t)
    scndm_corrected = p.scndm / (1 - p.beta2 ^ p.t)
    BLAS.axpy!(-p.lr, (fstm_corrected ./ (sqrt.(scndm_corrected) + p.eps)), w)
end

function gclip!(g, gclip)
    if gclip == 0
        g
    else
        gnorm = vecnorm(g)
        if gnorm <= gclip
            g
        else
            BLAS.scale!(gclip/gnorm, g)
        end
    end
end

mutable struct iRProp{T}
    eta_pos::T
    eta_neg::T
    dmax::T
    dmin::T
    d0::T
    gpre::Vector{T}
    Δ::Vector{T}
    iRProp(eta_pos::T, eta_neg::T; d0=1.5e-3π, dmin=1e-6π, dmax=6e-3π) where T = new{T}(eta_pos, eta_neg, dmax, dmin, d0)
end

function update!(w, g, p::iRProp{T}) where T
    N = length(g)
    if !isdefined(p, :gpre)
        p.gpre = zeros(T, N)
        p.Δ = p.d0*ones(T, N)
    end
    sgn = sign.(g.*p.gpre)
    posmask = sgn .> 0
    negmask = sgn .< 0
    p.Δ[posmask] = min.(p.eta_pos.*p.Δ[posmask], p.dmax)
    p.Δ[negmask] = max.(p.eta_neg.*p.Δ[negmask], p.dmin)

    p.gpre[:] = g
    p.gpre[negmask] = 0

    w[:] .-= sign.(p.gpre).*p.Δ
end

end
