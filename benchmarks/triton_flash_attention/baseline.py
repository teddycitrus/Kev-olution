import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 32
NUM_WARPS = 4
NUM_STAGES = 1


@triton.jit
def _attn_fwd(
    Q, K, V, Out, sm_scale,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_off = off_z * stride_qz + off_h * stride_qh
    k_off = off_z * stride_kz + off_h * stride_kh
    v_off = off_z * stride_vz + off_h * stride_vh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K + k_off + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        qk = tl.dot(q, k) * sm_scale
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + v_off + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        acc += tl.dot(p.to(tl.float16), v)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_off = off_z * stride_oz + off_h * stride_oh
    tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc.to(Out.type.element_ty))


def attention(q, k, v, sm_scale):
    B, H, N, D = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _attn_fwd[grid](
        q, k, v, o, sm_scale,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return o
