/**
 * Multi-Fidelity Sandbox Evaluator.
 *
 * Tier 1: static validation of the kernel source (structure, safety, launch config).
 * Tier 2: mini-workload profile. Checks compile constraints and numerical
 *         correctness, produces a noisy throughput estimate.
 * Tier 3: full benchmark measurement with low noise.
 *
 * No CUDA device is available in this environment, so Tiers 2 and 3 run a
 * deterministic simulated H100 performance model driven by optimization
 * features detected in the source. Results are seeded by a hash of the source,
 * so the same candidate always produces the same numbers.
 */

import type { BenchmarkSpec } from "../types";
import type { ExecutionMetrics, FailureKind, FailureSummary, FidelityTier } from "./graphStore";
import { MAX_FAILURE_SUMMARY_CHARS } from "./graphStore";
import type { KernelRunner, RunOutcome } from "./runners";

export interface KernelFeatures {
  blockM: number;
  blockN: number;
  numWarps: number;
  numStages: number;
  onlineSoftmax: boolean;
  fp32Accumulator: boolean;
  fp8: boolean;
  exp2: boolean;
  exp2Rescaled: boolean;
  blockPtr: boolean;
  tensorDescriptor: boolean;
  warpSpecialize: boolean;
  persistent: boolean;
}

export interface TierReport {
  tier: FidelityTier;
  passed: boolean;
  issues: string[];
  /** Simulated GPU time this tier would consume on real hardware. */
  costGpuSeconds: number;
  wallMs: number;
  score?: number;
  latencyMs?: number;
  maxAbsError?: number;
}

export interface EvalResult {
  /** True when every tier that ran passed and the candidate was not cut for regression. */
  ok: boolean;
  /** Present once Tier 1 passes; tier is the highest tier passed. */
  metrics?: ExecutionMetrics;
  failure?: FailureSummary;
  features: KernelFeatures | null;
  tiers: TierReport[];
  costGpuSeconds: number;
  /** GPU time avoided by stopping before maxTier. */
  savedGpuSeconds: number;
}

export interface EvaluateOptions {
  /** Highest tier to run. Default 3. */
  maxTier?: FidelityTier;
  /** Tier 2 estimate must reach this to be promoted to Tier 3. Default 0. */
  promotionScore?: number;
}

export interface EvaluatorOptions {
  /** Real milliseconds slept per simulated GPU-second, for demo pacing. Default 0. */
  wallMsPerGpuSecond?: number;
  /** Mixed into the noise seed. Default 0. */
  seed?: number;
  /**
   * Execution backend for Tiers 2 and 3 (see runners.ts). When absent the
   * built-in simulated H100 model is used.
   */
  runner?: KernelRunner;
}

export const TIER_COST_GPU_SECONDS: Readonly<Record<FidelityTier, number>> = { 1: 0.002, 2: 1.5, 3: 20 };
const TIER_NOISE: Readonly<Record<2 | 3, number>> = { 2: 0.03, 3: 0.005 };
const MAX_SOURCE_BYTES = 64 * 1024;
const FORBIDDEN_MODULES = ["os", "sys", "subprocess", "socket", "shutil", "ctypes", "requests", "urllib", "pickle", "importlib", "builtins"];
/** Matched against code with comments and strings removed. */
const DYNAMIC_CODE_PATTERNS: ReadonlyArray<readonly [RegExp, string]> = [
  [/\b__import__\b/, "__import__"],
  [/\bimportlib\b/, "importlib"],
  [/(?<![\w.])eval\s*\(/, "eval("],
  [/(?<![\w.])exec\s*\(/, "exec("],
  [/(?<![\w.])compile\s*\(/, "compile("],
  [/(?<![\w.])open\s*\(/, "open("],
  [/\b__builtins__\b|\bbuiltins\b/, "builtins"],
  [/(?<![\w.])globals\s*\(/, "globals("],
];
const STAGE_EFFICIENCY: Readonly<Record<number, number>> = { 1: 0.8, 2: 0.93, 3: 1, 4: 0.98, 5: 0.95, 6: 0.92 };
const VALID_WARPS = new Set([1, 2, 4, 8, 16]);

// ---------------------------------------------------------------------------
// Tier 1: static validation
// ---------------------------------------------------------------------------

function readIntConstant(source: string, name: string): number | undefined {
  const match = new RegExp(`^${name}\\s*=\\s*(\\d+)\\s*(?:#.*)?$`, "m").exec(source);
  return match ? Number(match[1]) : undefined;
}

const isPow2 = (n: number) => n > 0 && (n & (n - 1)) === 0;

function bracketIssues(source: string): string[] {
  const pairs: Record<string, string> = { ")": "(", "]": "[", "}": "{" };
  const stack: string[] = [];
  let quote: string | null = null;
  for (let i = 0; i < source.length; i++) {
    const ch = source[i]!;
    if (quote) {
      if (ch === "\\") i++;
      else if (source.startsWith(quote, i)) {
        i += quote.length - 1;
        quote = null;
      }
      continue;
    }
    if (ch === "#") {
      const nl = source.indexOf("\n", i);
      i = nl === -1 ? source.length : nl;
    } else if (ch === '"' || ch === "'") {
      quote = source.startsWith(ch.repeat(3), i) ? ch.repeat(3) : ch;
      i += quote.length - 1;
    } else if ("([{".includes(ch)) stack.push(ch);
    else if (ch in pairs) {
      if (stack.pop() !== pairs[ch]) return [`unbalanced '${ch}'`];
    }
  }
  if (quote) return ["unterminated string literal"];
  return stack.length ? [`unclosed '${stack[stack.length - 1]}'`] : [];
}

/** Removes comments and string literal contents (quotes and newlines are kept). */
export function stripCode(source: string): string {
  let out = "";
  for (let i = 0; i < source.length; i++) {
    const ch = source[i]!;
    if (ch === "#") {
      const nl = source.indexOf("\n", i);
      i = (nl === -1 ? source.length : nl) - 1;
    } else if (ch === '"' || ch === "'") {
      const quote = source.startsWith(ch.repeat(3), i) ? ch.repeat(3) : ch;
      let j = i + quote.length;
      let newlines = "";
      while (j < source.length && !source.startsWith(quote, j)) {
        if (source[j] === "\\") j++;
        else if (source[j] === "\n") newlines += "\n";
        j++;
      }
      out += quote + newlines + quote;
      i = j + quote.length - 1;
    } else out += ch;
  }
  return out;
}

export interface SourceAnalysis {
  /** Bodies of @triton.jit functions reachable from attention() or module code. */
  kernelText: string;
  /** Module-level code plus every reachable function body. */
  reachableText: string;
  /** @triton.jit functions launched with name[grid](...) from reachable host code. */
  launched: string[];
}

interface TopLevelFunction {
  name: string;
  jit: boolean;
  body: string;
}

/**
 * Splits stripped source into top-level functions and module code, then
 * follows calls (`f(`) and launches (`k[grid](`) from attention() and module
 * code. Code that is never reached contributes nothing to feature detection.
 */
export function analyzeSource(source: string): SourceAnalysis {
  const lines = stripCode(source).split("\n");
  const functions = new Map<string, TopLevelFunction>();
  const moduleLines: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const def = /^def\s+(\w+)\s*\(/.exec(lines[i]!);
    if (!def) {
      moduleLines.push(lines[i]!);
      continue;
    }
    const decorators: string[] = [];
    for (let k = i - 1; k >= 0 && lines[k]!.startsWith("@"); k--) decorators.push(lines[k]!);
    let depth = 0;
    let j = i;
    for (; j < lines.length; j++) {
      for (const ch of lines[j]!) depth += ch === "(" ? 1 : ch === ")" ? -1 : 0;
      if (depth <= 0) break;
    }
    let end = j + 1;
    while (end < lines.length && (!lines[end]!.trim() || /^\s/.test(lines[end]!))) end++;
    functions.set(def[1]!, { name: def[1]!, jit: decorators.some((d) => /^@triton\.jit\b/.test(d)), body: lines.slice(j + 1, end).join("\n") });
    i = end - 1;
  }

  const moduleText = moduleLines.join("\n");
  const reachable = new Set<string>();
  const launched = new Set<string>();
  const visit = (text: string, host: boolean) => {
    for (const fn of functions.values()) {
      const launch = new RegExp(`\\b${fn.name}\\s*\\[[^\\]]*\\]\\s*\\(`).test(text);
      const call = new RegExp(`\\b${fn.name}\\s*\\(`).test(text);
      if (fn.jit && launch && host) launched.add(fn.name);
      if ((launch || call) && !reachable.has(fn.name)) {
        reachable.add(fn.name);
        visit(fn.body, !fn.jit);
      }
    }
  };
  visit(moduleText, true);
  const entry = functions.get("attention");
  if (entry && !reachable.has(entry.name)) {
    reachable.add(entry.name);
    visit(entry.body, !entry.jit);
  }

  const bodies = [...reachable].map((n) => functions.get(n)!);
  return {
    kernelText: bodies.filter((f) => f.jit).map((f) => f.body).join("\n"),
    reachableText: [moduleText, ...bodies.map((f) => f.body)].join("\n"),
    launched: [...launched],
  };
}

/** Returns a list of problems; empty means the source passes Tier 1. */
export function staticCheck(source: string): string[] {
  if (!source.trim()) return ["empty source"];
  if (new TextEncoder().encode(source).length > MAX_SOURCE_BYTES) return [`source exceeds ${MAX_SOURCE_BYTES} bytes`];

  const issues: string[] = [];
  const code = stripCode(source);
  if (!/^import triton\s*$/m.test(code)) issues.push("missing 'import triton'");
  if (!/^import triton\.language as tl\s*$/m.test(code)) issues.push("missing 'import triton.language as tl'");
  if (!/^@triton\.jit\b[^\n]*\n(?:@[^\n]*\n)*def\s+\w+\s*\(/m.test(code)) issues.push("no @triton.jit kernel found");
  else {
    const { launched, kernelText } = analyzeSource(source);
    if (launched.length === 0) issues.push("no @triton.jit kernel is launched from attention() or module code");
    else if (!/tl\.dot\s*\(/.test(kernelText)) issues.push("kernel never calls tl.dot");
  }

  const forbidden = new RegExp(`^\\s*(?:import|from)\\s+(${FORBIDDEN_MODULES.join("|")})\\b`, "m").exec(code);
  if (forbidden) issues.push(`forbidden import '${forbidden[1]}'`);
  for (const [pattern, label] of DYNAMIC_CODE_PATTERNS) if (pattern.test(code)) issues.push(`forbidden dynamic code: ${label}`);
  if (/^[ ]*\t/m.test(source)) issues.push("tab characters in indentation");
  issues.push(...bracketIssues(source));

  const blockM = readIntConstant(source, "BLOCK_M");
  const blockN = readIntConstant(source, "BLOCK_N");
  const warps = readIntConstant(source, "NUM_WARPS");
  const stages = readIntConstant(source, "NUM_STAGES");
  for (const [name, value] of [["BLOCK_M", blockM], ["BLOCK_N", blockN]] as const) {
    if (value === undefined) issues.push(`missing module constant ${name}`);
    else if (!isPow2(value) || value < 16 || value > 256) issues.push(`${name}=${value} must be a power of two in [16, 256]`);
  }
  if (warps === undefined) issues.push("missing module constant NUM_WARPS");
  else if (!VALID_WARPS.has(warps)) issues.push(`NUM_WARPS=${warps} must be one of 1, 2, 4, 8, 16`);
  if (stages === undefined) issues.push("missing module constant NUM_STAGES");
  else if (stages < 1 || stages > 6) issues.push(`NUM_STAGES=${stages} must be in [1, 6]`);
  return issues;
}

/**
 * Detects optimization features. Assumes the source passed staticCheck.
 * Kernel features are read only from reachable @triton.jit bodies with
 * comments and strings removed, so dead code and comments earn nothing.
 */
export function extractFeatures(source: string): KernelFeatures {
  const { kernelText: k, reachableText } = analyzeSource(source);
  return {
    blockM: readIntConstant(source, "BLOCK_M") ?? 0,
    blockN: readIntConstant(source, "BLOCK_N") ?? 0,
    numWarps: readIntConstant(source, "NUM_WARPS") ?? 0,
    numStages: readIntConstant(source, "NUM_STAGES") ?? 0,
    onlineSoftmax: /tl\.maximum\s*\(/.test(k) && /\bm_i\b/.test(k),
    fp32Accumulator: /\bacc\s*=\s*tl\.zeros\([^)]*dtype\s*=\s*tl\.float32/.test(k),
    fp8: /float8/.test(reachableText),
    exp2: /tl\.exp2\s*\(/.test(k),
    exp2Rescaled: /1\.44269504|\blog2e\b|\bLOG2E\b|\bRCP_LN2\b/i.test(k),
    blockPtr: /tl\.make_block_ptr\s*\(/.test(k),
    tensorDescriptor: /make_tensor_descriptor\s*\(/.test(k),
    warpSpecialize: /warp_specialize\s*=\s*True/.test(k),
    persistent: /tl\.num_programs\s*\(/.test(k),
  };
}

// ---------------------------------------------------------------------------
// Simulated hardware model
// ---------------------------------------------------------------------------

export function sharedMemBytes(f: KernelFeatures, headDim: number): number {
  const kvBytes = f.fp8 ? 1 : 2;
  return (f.blockM * headDim * 2 + 2 * f.blockN * headDim * kvBytes) * f.numStages;
}

export function compileIssues(f: KernelFeatures, spec: BenchmarkSpec): string[] {
  const issues: string[] = [];
  const smem = sharedMemBytes(f, spec.problem.headDim);
  if (smem > spec.hardware.sharedMemBytes) {
    issues.push(`OutOfResources: shared memory, required ${smem}, hardware limit ${spec.hardware.sharedMemBytes}`);
  }
  if (f.warpSpecialize && f.numStages < 2) issues.push("warp_specialize requires NUM_STAGES >= 2");
  return issues;
}

/** Max absolute error of the candidate output against the PyTorch reference. */
export function numericalError(f: KernelFeatures): number {
  let err = 0;
  if (!f.onlineSoftmax) err = Math.max(err, Number.POSITIVE_INFINITY);
  if (f.exp2 && !f.exp2Rescaled) err = Math.max(err, 0.5);
  if (f.fp8) err = Math.max(err, 2.4e-2);
  if (!f.fp32Accumulator) err = Math.max(err, 3.1e-3);
  return err;
}

function rawThroughput(f: KernelFeatures): number {
  const tile = Math.pow(Math.min(1, (f.blockM * f.blockN) / (128 * 128)), 0.3);
  const spills = (f.blockM > 128 ? 0.75 : 1) * (f.blockN > 128 ? 0.8 : 1);
  const optimalWarps = Math.min(16, Math.max(1, f.blockM / 16));
  const warps = Math.max(0.5, 1 - 0.12 * Math.abs(Math.log2(f.numWarps / optimalWarps)));
  const stages = STAGE_EFFICIENCY[f.numStages] ?? 0.8;
  let bonus = 1;
  if (f.exp2) bonus *= 1.06;
  // Memory layouts are alternatives: TMA descriptors subsume block-pointer addressing.
  bonus *= f.tensorDescriptor ? 1.2 * 1.15 : f.blockPtr ? 1.2 : 1;
  if (f.warpSpecialize) bonus *= 1.25;
  if (f.persistent) bonus *= 1.08;
  if (f.fp8) bonus *= 1.4;
  return tile * spills * warps * stages * bonus;
}

export function attentionFlops(spec: BenchmarkSpec): number {
  const { batch, heads, seqLen, headDim, causal } = spec.problem;
  return (4 * batch * heads * seqLen * seqLen * headDim) / (causal ? 2 : 1);
}

function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Deterministic value in [-1, 1) via one mulberry32 step. */
function noise(seed: number): number {
  let t = (seed + 0x6d2b79f5) | 0;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
}

const PREFIXED_KINDS: readonly FailureKind[] = ["numerical", "regression", "runtime", "timeout", "compile"];

function failureKindFor(tier: FidelityTier, issue: string): FailureKind {
  if (tier === 1) return "syntax";
  return PREFIXED_KINDS.find((kind) => issue.startsWith(kind)) ?? "compile";
}

function summarize(tier: FidelityTier, issues: string[]): FailureSummary {
  const summary = `T${tier}: ${issues.join("; ")}`;
  return {
    kind: failureKindFor(tier, issues[0] ?? ""),
    tier,
    summary: summary.length > MAX_FAILURE_SUMMARY_CHARS ? `${summary.slice(0, MAX_FAILURE_SUMMARY_CHARS - 3)}...` : summary,
  };
}

// ---------------------------------------------------------------------------
// Evaluator
// ---------------------------------------------------------------------------

export class Evaluator {
  private readonly baselineRaw: number;
  private readonly wallMsPerGpuSecond: number;
  private readonly seed: number;
  private readonly runner?: KernelRunner;

  constructor(readonly spec: BenchmarkSpec, options: EvaluatorOptions = {}) {
    const issues = staticCheck(spec.baselineSource);
    if (issues.length) throw new Error(`Baseline for ${spec.name} fails Tier 1: ${issues.join("; ")}`);
    this.baselineRaw = rawThroughput(extractFeatures(spec.baselineSource));
    this.wallMsPerGpuSecond = options.wallMsPerGpuSecond ?? 0;
    this.seed = options.seed ?? 0;
    this.runner = options.runner;
  }

  /** "simulated" results must be labeled as such in the dashboard. */
  get backend(): KernelRunner["kind"] {
    return this.runner?.kind ?? "simulated";
  }

  /** Noise-free throughput the simulated hardware would deliver. */
  trueTflops(features: KernelFeatures): number {
    return (rawThroughput(features) / this.baselineRaw) * this.spec.baselineTflops;
  }

  async evaluate(source: string, options: EvaluateOptions = {}): Promise<EvalResult> {
    const maxTier = options.maxTier ?? 3;
    const promotion = options.promotionScore ?? 0;
    const tiers: TierReport[] = [];
    const hash = fnv1a(source) ^ this.seed;

    const finish = (failure?: FailureSummary, features: KernelFeatures | null = null): EvalResult => {
      const passed = tiers.filter((t) => t.passed);
      const top = passed[passed.length - 1];
      const ran = tiers.length as number;
      let saved = 0;
      for (let t = ran + 1; t <= maxTier; t++) saved += TIER_COST_GPU_SECONDS[t as FidelityTier];
      return {
        ok: !failure,
        metrics: top && {
          tier: top.tier,
          score: top.score,
          latencyMs: top.latencyMs,
          maxAbsError: top.maxAbsError,
          evalWallMs: tiers.reduce((s, t) => s + t.wallMs, 0),
        },
        failure,
        features,
        tiers,
        costGpuSeconds: tiers.reduce((s, t) => s + t.costGpuSeconds, 0),
        savedGpuSeconds: saved,
      };
    };

    const t1 = await this.runTier(1, () => ({ issues: staticCheck(source) }));
    tiers.push(t1);
    if (!t1.passed) return finish(summarize(1, t1.issues));
    const features = extractFeatures(source);
    if (maxTier === 1) return finish(undefined, features);

    const measure = (tier: 2 | 3): RunOutcome | Promise<RunOutcome> => {
      if (this.runner) return this.runner.run({ tier, source, features, spec: this.spec });
      const issues = compileIssues(features, this.spec);
      if (issues.length) return { issues };
      const maxAbsError = numericalError(features);
      if (maxAbsError > this.spec.errorTolerance) {
        return { issues: [`numerical: max abs error ${maxAbsError} exceeds tolerance ${this.spec.errorTolerance}`], maxAbsError };
      }
      const score = this.trueTflops(features) * (1 + TIER_NOISE[tier] * noise(hash + tier));
      return { issues: [], score, maxAbsError, latencyMs: (attentionFlops(this.spec) / (score * 1e12)) * 1e3 };
    };

    const t2 = await this.runTier(2, () => measure(2));
    tiers.push(t2);
    if (!t2.passed) return finish(summarize(2, t2.issues), features);
    if (t2.score! < promotion) {
      return finish(summarize(2, [`regression: estimate ${t2.score!.toFixed(1)} TFLOPS below promotion bar ${promotion.toFixed(1)}`]), features);
    }
    if (maxTier === 2) return finish(undefined, features);

    const t3 = await this.runTier(3, () => measure(3));
    tiers.push(t3);
    return finish(t3.passed ? undefined : summarize(3, t3.issues), features);
  }

  private async runTier(tier: FidelityTier, run: () => RunOutcome | Promise<RunOutcome>): Promise<TierReport> {
    const started = performance.now();
    const { costGpuSeconds: measuredCost, ...outcome } = await run();
    const cost = measuredCost ?? TIER_COST_GPU_SECONDS[tier];
    // Pacing only applies to the simulation; real runs already take real time.
    if (this.wallMsPerGpuSecond > 0 && measuredCost === undefined) await Bun.sleep(cost * this.wallMsPerGpuSecond);
    return { tier, passed: outcome.issues.length === 0, costGpuSeconds: cost, wallMs: performance.now() - started, ...outcome };
  }
}
