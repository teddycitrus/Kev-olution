/**
 * Parametric Triton flash-attention forward kernel.
 *
 * The template mutator edits a KernelConfig and re-renders source from it, so
 * every candidate is well formed by construction: module constants stay as
 * top-level `NAME = <int>` lines (Tier 1 contract), the host entry point stays
 * `attention(q, k, v, sm_scale)` (harness.py contract), and every structural
 * feature lives inside the @triton.jit kernel body.
 *
 * Structural variants and their Triton requirements:
 *   memory "block_ptr": tl.make_block_ptr / tl.advance (Triton 2.1+).
 *   memory "tma":       device-side tensor descriptors, needs contiguous
 *                       inputs and triton.set_allocator (Triton 3.3+).
 *   warpSpecialize:     tl.range(..., warp_specialize=True) (Triton 3.4+).
 *   persistent:         1-D grid of NUM_SMS programs striding over all tiles.
 * Like the baseline, loads are unmasked: N_CTX must divide by BLOCK_M and BLOCK_N.
 */

import { extractFeatures } from "../harness/evaluator";

export type MemoryLayout = "pointer" | "block_ptr" | "tma";

export interface KernelConfig {
  blockM: number;
  blockN: number;
  numWarps: number;
  numStages: number;
  /** exp2 softmax with the log2(e) factor folded into the qk scale. */
  exp2: boolean;
  memory: MemoryLayout;
  warpSpecialize: boolean;
  persistent: boolean;
}

/** Recovers a config from any candidate source (baseline included). */
export function configFromSource(source: string): KernelConfig {
  const f = extractFeatures(source);
  return {
    blockM: f.blockM,
    blockN: f.blockN,
    numWarps: f.numWarps,
    numStages: f.numStages,
    exp2: f.exp2 && f.exp2Rescaled,
    memory: f.tensorDescriptor ? "tma" : f.blockPtr ? "block_ptr" : "pointer",
    warpSpecialize: f.warpSpecialize,
    persistent: f.persistent,
  };
}

/** Compact label, e.g. "128x64 w8 s3 exp2 tma ws persistent". */
export function describeConfig(c: KernelConfig): string {
  const flags = [c.exp2 && "exp2", c.memory !== "pointer" && c.memory, c.warpSpecialize && "ws", c.persistent && "persistent"].filter(Boolean);
  return [`${c.blockM}x${c.blockN}`, `w${c.numWarps}`, `s${c.numStages}`, ...flags].join(" ");
}

export const sameConfig = (a: KernelConfig, b: KernelConfig): boolean => describeConfig(a) === describeConfig(b);

const indent = (lines: string[], spaces: number) => lines.map((l) => (l ? " ".repeat(spaces) + l : l));

function tileBody(c: KernelConfig): string[] {
  const exp = c.exp2 ? "tl.exp2" : "tl.exp";
  const loopArgs = c.warpSpecialize ? "0, N_CTX, BLOCK_N, warp_specialize=True" : "0, N_CTX, BLOCK_N";
  const lines: string[] = [];

  if (c.memory === "tma") {
    lines.push("offset_y = off_hz * N_CTX", "q = desc_q.load([offset_y + start_m * BLOCK_M, 0])");
  } else {
    lines.push(
      "off_z = off_hz // H",
      "off_h = off_hz % H",
      "q_off = off_z * stride_qz + off_h * stride_qh",
      "k_off = off_z * stride_kz + off_h * stride_kh",
      "v_off = off_z * stride_vz + off_h * stride_vh",
      "o_off = off_z * stride_oz + off_h * stride_oh",
    );
    if (c.memory === "block_ptr") {
      lines.push(
        "q_bp = tl.make_block_ptr(base=Q + q_off, shape=(N_CTX, HEAD_DIM), strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))",
        "k_bp = tl.make_block_ptr(base=K + k_off, shape=(HEAD_DIM, N_CTX), strides=(stride_kk, stride_kn), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1))",
        "v_bp = tl.make_block_ptr(base=V + v_off, shape=(N_CTX, HEAD_DIM), strides=(stride_vn, stride_vk), offsets=(0, 0), block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0))",
        "q = tl.load(q_bp)",
      );
    } else {
      lines.push(
        "offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)",
        "offs_n = tl.arange(0, BLOCK_N)",
        "offs_d = tl.arange(0, HEAD_DIM)",
        "q = tl.load(Q + q_off + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)",
      );
    }
  }

  lines.push(
    "",
    'm_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")',
    "l_i = tl.zeros([BLOCK_M], dtype=tl.float32)",
    "acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)",
    c.exp2 ? "qk_scale = sm_scale * 1.44269504  # log2(e): exp2(x * log2e) == exp(x)" : "qk_scale = sm_scale",
    "",
    `for start_n in tl.range(${loopArgs}):`,
  );

  const loadK =
    c.memory === "tma" ? "k = desc_k.load([offset_y + start_n, 0]).T"
    : c.memory === "block_ptr" ? "k = tl.load(k_bp)"
    : "k = tl.load(K + k_off + (start_n + offs_n)[None, :] * stride_kn + offs_d[:, None] * stride_kk)";
  const loadV =
    c.memory === "tma" ? "v = desc_v.load([offset_y + start_n, 0])"
    : c.memory === "block_ptr" ? "v = tl.load(v_bp)"
    : "v = tl.load(V + v_off + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vk)";
  const loop = [
    loadK,
    "qk = tl.dot(q, k) * qk_scale",
    "m_ij = tl.maximum(m_i, tl.max(qk, 1))",
    `p = ${exp}(qk - m_ij[:, None])`,
    `alpha = ${exp}(m_i - m_ij)`,
    "l_i = l_i * alpha + tl.sum(p, 1)",
    "acc = acc * alpha[:, None]",
    loadV,
    "acc += tl.dot(p.to(tl.float16), v)",
    "m_i = m_ij",
  ];
  if (c.memory === "block_ptr") loop.push("k_bp = tl.advance(k_bp, (0, BLOCK_N))", "v_bp = tl.advance(v_bp, (BLOCK_N, 0))");
  lines.push(...indent(loop, 4), "", "acc = acc / l_i[:, None]");

  if (c.memory === "tma") {
    lines.push("desc_o.store([offset_y + start_m * BLOCK_M, 0], acc.to(Out.dtype.element_ty))");
  } else if (c.memory === "block_ptr") {
    lines.push(
      "o_bp = tl.make_block_ptr(base=Out + o_off, shape=(N_CTX, HEAD_DIM), strides=(stride_om, stride_ok), offsets=(start_m * BLOCK_M, 0), block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0))",
      "tl.store(o_bp, acc.to(Out.type.element_ty))",
    );
  } else {
    lines.push("tl.store(Out + o_off + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok, acc.to(Out.type.element_ty))");
  }
  return lines;
}

function kernelBody(c: KernelConfig): string[] {
  const lines: string[] = [];
  if (c.memory === "tma") {
    lines.push(
      "y_dim = Z * H * N_CTX",
      "desc_q = tl.make_tensor_descriptor(Q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])",
      "desc_k = tl.make_tensor_descriptor(K, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])",
      "desc_v = tl.make_tensor_descriptor(V, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])",
      "desc_o = tl.make_tensor_descriptor(Out, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])",
      "",
    );
  }
  if (c.persistent) {
    lines.push(
      "num_m_blocks = tl.cdiv(N_CTX, BLOCK_M)",
      "num_tiles = num_m_blocks * Z * H",
      "for tile_id in range(tl.program_id(0), num_tiles, tl.num_programs(0)):",
      "    start_m = tile_id % num_m_blocks",
      "    off_hz = tile_id // num_m_blocks",
      ...indent(tileBody(c), 4),
    );
  } else {
    lines.push("start_m = tl.program_id(0)", "off_hz = tl.program_id(1)", ...tileBody(c));
  }
  return indent(lines, 4);
}

function hostWrapper(c: KernelConfig): string[] {
  const lines = ["def attention(q, k, v, sm_scale):", "    B, H, N, D = q.shape", "    o = torch.empty_like(q)"];
  if (c.memory === "tma") {
    lines.push(
      "    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()",
      "    triton.set_allocator(lambda size, align, stream: torch.empty(size, device=q.device, dtype=torch.int8))",
    );
  }
  lines.push("    assert N % BLOCK_M == 0 and N % BLOCK_N == 0");
  if (c.persistent) {
    lines.push(
      "    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count",
      "    grid = (min(num_sms, triton.cdiv(N, BLOCK_M) * B * H),)",
    );
  } else {
    lines.push("    grid = (triton.cdiv(N, BLOCK_M), B * H)");
  }
  lines.push(
    "    _attn_fwd[grid](",
    "        q, k, v, o, sm_scale,",
    "        *q.stride(), *k.stride(), *v.stride(), *o.stride(),",
    "        B, H, N,",
    "        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,",
    "        num_warps=NUM_WARPS, num_stages=NUM_STAGES,",
    "    )",
    "    return o",
  );
  return lines;
}

export function renderKernel(c: KernelConfig): string {
  return [
    "import torch",
    "import triton",
    "import triton.language as tl",
    "",
    `BLOCK_M = ${c.blockM}`,
    `BLOCK_N = ${c.blockN}`,
    `NUM_WARPS = ${c.numWarps}`,
    `NUM_STAGES = ${c.numStages}`,
    "",
    "",
    "@triton.jit",
    "def _attn_fwd(",
    "    Q, K, V, Out, sm_scale,",
    "    stride_qz, stride_qh, stride_qm, stride_qk,",
    "    stride_kz, stride_kh, stride_kn, stride_kk,",
    "    stride_vz, stride_vh, stride_vn, stride_vk,",
    "    stride_oz, stride_oh, stride_om, stride_ok,",
    "    Z, H, N_CTX,",
    "    HEAD_DIM: tl.constexpr,",
    "    BLOCK_M: tl.constexpr,",
    "    BLOCK_N: tl.constexpr,",
    "):",
    ...kernelBody(c),
    "",
    "",
    ...hostWrapper(c),
    "",
  ].join("\n");
}
