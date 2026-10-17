/**
 * Prompt templates for LLM-backed code generation (LlmGenerator in
 * src/agents/mutator.ts). The default TemplateGenerator does not use these;
 * they exist so an LLM can be swapped in without touching the loop.
 *
 * Two families:
 *   micro:      small, local edits (tiling, pipelining/unrolling depth, warps,
 *               exp2 softmax). One change at a time, keep the structure.
 *   structural: Paradigm Shift rewrites chosen by kev-27b (memory layout,
 *               TMA, warp specialization, persistent scheduling).
 */

import type { LineageContext } from "../harness/graphStore";

export type PromptFamily = "micro" | "structural";

export interface MutationPromptInput {
  family: PromptFamily;
  strategyLabel: string;
  strategyDescription: string;
  /** e.g. "64x32 w4 s1" from describeConfig(). */
  parentSummary: string;
  /** The config the template generator would produce, as a target hint. */
  targetSummary?: string;
  parentSource: string;
  lineage: LineageContext;
  /** Paradigm Shift directive text from kev-27b, structural family only. */
  directive?: string;
  benchmarkDescription: string;
}

/** Hard constraints every candidate must satisfy (Tier 1 and harness.py contracts). */
export const KERNEL_CONTRACT = [
  "Keep `import torch`, `import triton`, `import triton.language as tl` as plain top-level imports.",
  "Keep module constants BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES as top-level `NAME = <int>` lines and use them at launch.",
  "BLOCK_M and BLOCK_N are powers of two in [16, 256]; NUM_WARPS in {1, 2, 4, 8, 16}; NUM_STAGES in [1, 6].",
  "Keep exactly one @triton.jit kernel and the host entry point `attention(q, k, v, sm_scale)` returning the output tensor.",
  "Output must match the PyTorch reference: keep the online softmax running max, keep an fp32 accumulator, and if you use tl.exp2, fold log2(e) = 1.44269504 into the scale.",
  "No other imports. No file, network, process, eval, or exec access.",
  "Use spaces, not tabs. Return only one ```python code block containing the full file.",
];

const MICRO_TEMPLATE = `You are optimizing a Triton flash-attention forward kernel.
{{benchmark}}

Current kernel config: {{parent}}
Apply exactly one micro-optimization: {{strategy}}
What it means: {{description}}
{{target}}
Keep the kernel structure unchanged. Change only what this optimization requires.

Lineage so far (root first):
{{ancestors}}

Recent dead ends (do not repeat them):
{{deadEnds}}

Rules:
{{rules}}

Current source:
\`\`\`python
{{source}}
\`\`\``;

const STRUCTURAL_TEMPLATE = `You are performing a structural rewrite ("Paradigm Shift") of a Triton flash-attention forward kernel.
Parameter tuning has plateaued; small edits no longer help.
{{benchmark}}

Current kernel config: {{parent}}
Directive: {{strategy}}
What it means: {{description}}
{{directive}}
{{target}}
Rewrite the memory access and scheduling structure as the directive requires. You may retune tile sizes and stages for the new structure.

Lineage so far (root first):
{{ancestors}}

Recent dead ends (do not repeat them):
{{deadEnds}}

Rules:
{{rules}}

Current source:
\`\`\`python
{{source}}
\`\`\``;

function fill(template: string, values: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key: string) => values[key] ?? "");
}

export function renderMutationPrompt(input: MutationPromptInput): string {
  const ancestors =
    input.lineage.ancestors
      .map((a) => `- gen ${a.generation} ${a.mutation.strategy}${a.metrics?.score !== undefined ? `: ${a.metrics.score.toFixed(1)} TFLOPS` : ""}`)
      .join("\n") || "- (none)";
  const deadEnds =
    input.lineage.deadEnds.map((d) => `- ${d.strategy}: ${d.failure?.summary ?? d.pruneReason ?? "rejected"}`).join("\n") || "- (none)";
  return fill(input.family === "micro" ? MICRO_TEMPLATE : STRUCTURAL_TEMPLATE, {
    benchmark: input.benchmarkDescription,
    parent: input.parentSummary,
    strategy: input.strategyLabel,
    description: input.strategyDescription,
    directive: input.directive ? `Why now: ${input.directive}` : "",
    target: input.targetSummary ? `Suggested target config: ${input.targetSummary}` : "",
    ancestors,
    deadEnds,
    rules: KERNEL_CONTRACT.map((r) => `- ${r}`).join("\n"),
    source: input.parentSource,
  });
}

/** Extracts the first fenced python block from an LLM reply. */
export function extractCodeBlock(reply: string): string | undefined {
  const match = /```(?:python|py)?\s*\n([\s\S]*?)```/.exec(reply);
  return match ? `${match[1]!.replace(/\s+$/, "")}\n` : undefined;
}
