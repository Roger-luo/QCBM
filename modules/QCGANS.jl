module QCGANS
using Compat

using Yao
using Yao.Zoo
using Yao.Blocks

import Yao.Registers: tracedist
import Base: gradient

export QCGAN, qcgan, p0g, p0t, loss
"""
Quantum Circuit GAN.

Reference:
    Benedetti, M., Grant, E., Wossnig, L., & Severini, S. (2018). Adversarial quantum circuit learning for pure state approximation, 1–14.
"""
struct QCGAN{N, GT<:MatrixBlock{N}, DT<:MatrixBlock, OT<:MatrixBlock, CT<:AbstractBlock}
    target::DefaultRegister
    generator::GT
    discriminator::DT
    reg0::DefaultRegister
    witness_op::OT

    circuit::CT
    grots::Vector{RotationGate}
    drots::Vector{RotationGate}
end

function QCGAN(target::DefaultRegister, gen::MatrixBlock, dis::MatrixBlock)
    N = nqubits(target)
    c = sequence(gen, addbit(1), dis)
    witness_op = put(N+1, (N+1)=>P0)
    QCGAN{N, typeof(gen), typeof(dis), typeof(witness_op), typeof(c)}(target, gen, dis, zero_state(N), witness_op, c, collect_rotblocks(gen), collect_rotblocks(dis))
end

function qcgan(target::DefaultRegister, depth_gen::Int, depth_disc::Int, pairs1::Vector{<:Pair}, pairs2::Vector{<:Pair})
    n = nqubits(target)
    generator = diff_circuit(n, depth_gen, pairs1)
    discriminator = diff_circuit(n+1, depth_disc, pairs2)
    return QCGAN(target, generator, discriminator)
end

p0g(qcg::QCGAN) = ()->expect(qcg.witness_op, apply!(copy(qcg.reg0), qcg.circuit)) |> real
p0t(qcg::QCGAN) = ()->expect(qcg.witness_op, apply!(copy(qcg.target), qcg.circuit[2:end])) |> real
loss(qcg::QCGAN) = ()->(qcg |> p0t)() - (qcg |> p0g)()

tracedist(qcg::QCGAN) = tracedist(qcg.target, apply!(copy(qcg.reg0), qcg.generator))[]

function gradient(qcg::QCGAN)
    ggrad_g = opgrad(qcg |> p0g, qcg.grots)
    dgrad_g = opgrad(qcg |> p0g, qcg.drots)
    dgrad_t = opgrad(qcg |> p0t, qcg.drots)
    -ggrad_g, dgrad_t - dgrad_g
end

end
