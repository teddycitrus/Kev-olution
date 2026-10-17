/**
 * Mutator agent: turns a parent node into candidate children.
 *
 * Strategies edit a KernelConfig (see kernelTemplate.ts). A CodeGenerator turns
 * (parent, strategy, target config) into source:
 *   TemplateGenerator (default): renders the target config. Deterministic,
 *     offline, always satisfies the Tier 1 contract.
 *   LlmGenerator: renders a prompt from src/prompts/mutationPrompts.ts and asks
 *     an injected completion function for the full file.
 *
 * Local (micro) strategies are proposed every cycle. Structural strategies are
 * only used when the Paradigm Shift engine issues a directive.
 */

import type { LineageContext, MutationKind, NewNodeInput, TrajectoryNode } from "../harness/graphStore";
import type { Rng } from "../harness/rng";
import { extractCodeBlock, renderMutationPrompt } from "../prompts/mutationPrompts";
import { configFromSource, describeConfig, renderKernel, sameConfig, type KernelConfig } from "./kernelTemplate";

export interface MutationStrategy {
  id: string;
  kind: Extract<MutationKind, "local" | "paradigm_shift">;
  label: string;
  description: string;
  /** Returns the mutated config, or null when not applicable to `c`. */
  apply(c: KernelConfig): KernelConfig | null;
}

const POW2_TILES = [16, 32, 64, 128, 256];
const WARPS = [1, 2, 4, 8, 16];

function step(values: readonly number[], current: number, dir: 1 | -1): number | null {
  const i = values.indexOf(current);
  const next = i === -1 ? undefined : values[i + dir];
  return next === undefined ? null : next;
}

function tweak(c: KernelConfig, key: "blockM" | "blockN" | "numWarps", values: readonly number[], dir: 1 | -1): KernelConfig | null {
  const v = step(values, c[key], dir);
  return v === null ? null : { ...c, [key]: v };
}

export const MICRO_STRATEGIES: readonly MutationStrategy[] = [
  { id: "tile_m_up", kind: "local", label: "grow BLOCK_M", description: "Double the query tile height for more reuse of K/V per load.", apply: (c) => tweak(c, "blockM", POW2_TILES, 1) },
  { id: "tile_m_down", kind: "local", label: "shrink BLOCK_M", description: "Halve the query tile height to cut registers and shared memory.", apply: (c) => tweak(c, "blockM", POW2_TILES, -1) },
  { id: "tile_n_up", kind: "local", label: "grow BLOCK_N", description: "Double the key/value tile width to amortize loop overhead.", apply: (c) => tweak(c, "blockN", POW2_TILES, 1) },
  { id: "tile_n_down", kind: "local", label: "shrink BLOCK_N", description: "Halve the key/value tile width to fit deeper pipelines.", apply: (c) => tweak(c, "blockN", POW2_TILES, -1) },
  { id: "warps_up", kind: "local", label: "more warps", description: "Double num_warps to raise occupancy of the tensor cores.", apply: (c) => tweak(c, "numWarps", WARPS, 1) },
  { id: "warps_down", kind: "local", label: "fewer warps", description: "Halve num_warps to give each warp more registers.", apply: (c) => tweak(c, "numWarps", WARPS, -1) },
  {
    id: "stages_up", kind: "local", label: "deeper pipelining", description: "Add a software-pipelining stage (unrolls K/V prefetch one tile ahead).",
    apply: (c) => (c.numStages < 6 ? { ...c, numStages: c.numStages + 1 } : null),
  },
  {
    id: "stages_down", kind: "local", label: "shallower pipelining", description: "Drop a pipelining stage to free shared memory.",
    apply: (c) => (c.numStages > 1 && !(c.warpSpecialize && c.numStages <= 2) ? { ...c, numStages: c.numStages - 1 } : null),
  },
  { id: "exp2_softmax", kind: "local", label: "exp2 softmax", description: "Use tl.exp2 with log2(e) folded into the qk scale (one fewer multiply per element).", apply: (c) => (c.exp2 ? null : { ...c, exp2: true }) },
];

/** Priority order doubles as the fallback order when kev-27b is unavailable. */
export const STRUCTURAL_STRATEGIES: readonly MutationStrategy[] = [
  {
    id: "block_ptr_layout", kind: "paradigm_shift", label: "2-D block pointer memory layout",
    description: "Replace scalar pointer arithmetic with tl.make_block_ptr tiles so the compiler emits vectorized, layout-aware loads.",
    apply: (c) => (c.memory === "pointer" ? { ...c, memory: "block_ptr" } : null),
  },
  {
    id: "tma_descriptors", kind: "paradigm_shift", label: "TMA tensor descriptors",
    description: "Move K/V/Q traffic to device-side tensor descriptors (Hopper TMA): asynchronous bulk copies into shared memory.",
    apply: (c) => (c.memory !== "tma" ? { ...c, memory: "tma" } : null),
  },
  {
    id: "warp_specialization", kind: "paradigm_shift", label: "producer/consumer warp specialization",
    description: "Split warps into load producers and MMA consumers so memory and tensor-core work overlap.",
    apply: (c) => (c.warpSpecialize ? null : { ...c, warpSpecialize: true, numStages: Math.max(2, c.numStages) }),
  },
  {
    id: "persistent_kernel", kind: "paradigm_shift", label: "persistent tile scheduling",
    description: "Launch one program per SM that strides over all (query block, head) tiles, removing launch and tail overhead.",
    apply: (c) => (c.persistent ? null : { ...c, persistent: true }),
  },
];

export const ALL_STRATEGIES: readonly MutationStrategy[] = [...MICRO_STRATEGIES, ...STRUCTURAL_STRATEGIES];
export const strategyById = (id: string) => ALL_STRATEGIES.find((s) => s.id === id);

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

export interface GenerationRequest {
  parent: TrajectoryNode;
  parentConfig: KernelConfig;
  strategy: MutationStrategy;
  targetConfig: KernelConfig;
  lineage: LineageContext;
  benchmarkDescription: string;
  directive?: string;
}

export interface CodeGenerator {
  readonly kind: "template" | "llm";
  /** Returns full candidate source, or null to skip this proposal. */
  generate(request: GenerationRequest): Promise<string | null>;
}

export class TemplateGenerator implements CodeGenerator {
  readonly kind = "template" as const;
  async generate(request: GenerationRequest): Promise<string> {
    return renderKernel(request.targetConfig);
  }
}

export type CompletionFn = (prompt: string) => Promise<string>;

export class LlmGenerator implements CodeGenerator {
  readonly kind = "llm" as const;
  constructor(private readonly complete: CompletionFn) {}

  async generate(r: GenerationRequest): Promise<string | null> {
    const prompt = renderMutationPrompt({
      family: r.strategy.kind === "paradigm_shift" ? "structural" : "micro",
      strategyLabel: r.strategy.label,
      strategyDescription: r.strategy.description,
      parentSummary: describeConfig(r.parentConfig),
      targetSummary: describeConfig(r.targetConfig),
      parentSource: r.parent.source,
      lineage: r.lineage,
      directive: r.directive,
      benchmarkDescription: r.benchmarkDescription,
    });
    return extractCodeBlock(await this.complete(prompt)) ?? null;
  }
}

// ---------------------------------------------------------------------------
// Mutator
// ---------------------------------------------------------------------------

export interface MutationProposal {
  input: NewNodeInput;
  strategy: MutationStrategy;
  config: KernelConfig;
}

export interface MutatorOptions {
  generator?: CodeGenerator;
  benchmarkDescription?: string;
  /** Returns true for sources already in the graph; those are skipped. */
  isKnown?: (source: string) => boolean;
}

/** Retunes tried on top of a structural rewrite, since the optimum moves with the structure. */
const RETUNES: ReadonlyArray<(c: KernelConfig) => KernelConfig | null> = [
  (c) => c,
  (c) => (c.blockN < 128 ? { ...c, blockN: 128, numStages: Math.min(c.numStages, 2) } : null),
  (c) => (c.blockN > 64 ? { ...c, blockN: 64, numStages: Math.max(c.numStages, 3) } : null),
  (c) => (c.numWarps < 8 ? { ...c, numWarps: 8 } : null),
  (c) => (c.blockM < 128 ? { ...c, blockM: 128 } : null),
];

export class Mutator {
  readonly generator: CodeGenerator;
  private readonly benchmarkDescription: string;
  private readonly isKnown: (source: string) => boolean;

  constructor(options: MutatorOptions = {}) {
    this.generator = options.generator ?? new TemplateGenerator();
    this.benchmarkDescription = options.benchmarkDescription ?? "";
    this.isKnown = options.isKnown ?? (() => false);
  }

  /** Up to `count` local candidates: single micro steps plus random pairs. */
  async proposeLocal(parent: TrajectoryNode, lineage: LineageContext, count: number, rng: Rng): Promise<MutationProposal[]> {
    const parentConfig = configFromSource(parent.source);
    const plans: Array<{ strategy: MutationStrategy; config: KernelConfig }> = [];
    for (const s of MICRO_STRATEGIES) {
      const c = s.apply(parentConfig);
      if (c) plans.push({ strategy: s, config: c });
    }
    const singles = plans.length;
    for (let i = 0; i < singles; i++) {
      const first = plans[i]!;
      const second = MICRO_STRATEGIES[rng.int(MICRO_STRATEGIES.length)]!;
      const c = second.id === first.strategy.id ? null : second.apply(first.config);
      if (c) {
        plans.push({
          strategy: { ...first.strategy, id: `${first.strategy.id}+${second.id}`, label: `${first.strategy.label} + ${second.label}`, description: `${first.strategy.description} Then: ${second.description}` },
          config: c,
        });
      }
    }
    return this.materialize(parent, parentConfig, lineage, rng.shuffle(plans), count);
  }

  /** Candidates for a Paradigm Shift: the structural rewrite plus retuned variants. */
  async proposeParadigmShift(parent: TrajectoryNode, lineage: LineageContext, strategyId: string, count: number, directive?: string): Promise<MutationProposal[]> {
    const strategy = STRUCTURAL_STRATEGIES.find((s) => s.id === strategyId);
    if (!strategy) throw new Error(`unknown structural strategy ${strategyId}`);
    const parentConfig = configFromSource(parent.source);
    const base = strategy.apply(parentConfig);
    if (!base) return [];
    const plans = RETUNES.map((r) => r(base))
      .filter((c): c is KernelConfig => c !== null)
      .filter((c, i, all) => all.findIndex((x) => sameConfig(x, c)) === i)
      .map((config) => ({ strategy, config }));
    return this.materialize(parent, parentConfig, lineage, plans, count, directive);
  }

  private async materialize(
    parent: TrajectoryNode,
    parentConfig: KernelConfig,
    lineage: LineageContext,
    plans: Array<{ strategy: MutationStrategy; config: KernelConfig }>,
    count: number,
    directive?: string,
  ): Promise<MutationProposal[]> {
    const out: MutationProposal[] = [];
    const seen = new Set<string>();
    for (const plan of plans) {
      if (out.length >= count) break;
      if (sameConfig(plan.config, parentConfig)) continue;
      const source = await this.generator.generate({
        parent, parentConfig, strategy: plan.strategy, targetConfig: plan.config, lineage, directive, benchmarkDescription: this.benchmarkDescription,
      });
      if (!source || source === parent.source || seen.has(source) || this.isKnown(source)) continue;
      seen.add(source);
      out.push({
        strategy: plan.strategy,
        config: plan.config,
        input: {
          parentIds: [parent.id],
          source,
          diff: lineDiff(parent.source, source),
          mutation: {
            kind: plan.strategy.kind,
            strategy: plan.strategy.label,
            rationale: `${plan.strategy.description} ${describeConfig(parentConfig)} -> ${describeConfig(plan.config)}.`,
          },
        },
      });
    }
    return out;
  }
}

/** Minimal line diff ("-"/"+" lines only, no context) via LCS. */
export function lineDiff(before: string, after: string): string {
  const a = before.split("\n");
  const b = after.split("\n");
  const n = a.length;
  const m = b.length;
  const lcs: Uint16Array[] = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i]![j] = a[i] === b[j] ? lcs[i + 1]![j + 1]! + 1 : Math.max(lcs[i + 1]![j]!, lcs[i]![j + 1]!);
    }
  }
  const out: string[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      i++;
      j++;
    } else if (lcs[i + 1]![j]! >= lcs[i]![j + 1]!) out.push(`-${a[i++]}`);
    else out.push(`+${b[j++]}`);
  }
  while (i < n) out.push(`-${a[i++]}`);
  while (j < m) out.push(`+${b[j++]}`);
  return out.join("\n");
}
