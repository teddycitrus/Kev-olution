/**
 * Reflector agent: compresses a failed evaluation into a FailureSummary short
 * enough (< 280 chars) to sit in the DAG and be replayed into prompts and Kev
 * states via lineageContext(). The full trace goes to disk, never the graph.
 *
 * Summary format:  "T<tier> <kind>: <core> | cfg <config> | fix: <hint>"
 * e.g. "T2 compile: smem 245760B > 232448B | cfg 128x128 w8 s3 | fix: shrink BLOCK_N or NUM_STAGES"
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { EvalResult } from "../harness/evaluator";
import { MAX_FAILURE_SUMMARY_CHARS, type FailureKind, type FailureSummary, type FidelityTier } from "../harness/graphStore";
import { configFromSource, describeConfig } from "./kernelTemplate";

/** Strictly below the store cap, per the PRD's "< 280 char" requirement. */
export const REFLECTOR_MAX_CHARS = MAX_FAILURE_SUMMARY_CHARS - 1;

const HINTS: Readonly<Record<FailureKind, string>> = {
  syntax: "keep Tier 1 contract (imports, constants, @triton.jit)",
  compile: "shrink BLOCK_N or NUM_STAGES",
  runtime: "check launch grid and descriptor allocation",
  numerical: "keep fp32 acc and running max; fold log2e into exp2",
  timeout: "reduce tile work or pipeline depth",
  regression: "local basin exhausted; try a structural change",
};

export interface ReflectInput {
  nodeId: string;
  source: string;
  result: EvalResult;
}

export interface ReflectorOptions {
  /** When set, the full tier trace is written to <logDir>/<nodeId>.log. */
  logDir?: string;
  /** Task label for the "cfg" part of summaries. Defaults to the kernel config label. */
  describe?: (source: string) => string;
}

/** Shortens the evaluator's verbose issue strings. */
export function compressIssue(issue: string): string {
  const smem = /shared memory, required (\d+), hardware limit (\d+)/.exec(issue);
  if (smem) return `smem ${smem[1]}B > ${smem[2]}B`;
  const num = /max abs error (\S+) exceeds tolerance (\S+)/.exec(issue);
  if (num) return `max|err| ${Number(num[1]).toPrecision(3)} > tol ${num[2]}`;
  const reg = /estimate ([\d.]+) TFLOPS below promotion bar ([\d.]+)/.exec(issue);
  if (reg) return `est ${reg[1]} < bar ${reg[2]} TFLOPS`;
  return issue.replace(/^(compile|runtime|numerical|timeout|regression):\s*/, "").replace(/\s+/g, " ").trim();
}

function fit(parts: string[], max: number): string {
  let text = parts.join(" | ");
  if (text.length <= max) return text;
  // Drop the hint first, then truncate the core.
  text = parts.slice(0, 2).join(" | ");
  return text.length <= max ? text : `${text.slice(0, max - 3)}...`;
}

export class Reflector {
  constructor(private readonly options: ReflectorOptions = {}) {}

  reflect({ nodeId, source, result }: ReflectInput): FailureSummary {
    const failing = [...result.tiers].reverse().find((t) => !t.passed) ?? result.tiers[result.tiers.length - 1];
    const tier: FidelityTier = result.failure?.tier ?? failing?.tier ?? 1;
    const kind: FailureKind = result.failure?.kind ?? "runtime";
    const issues = failing?.issues.length ? failing.issues : [result.failure?.summary ?? "unknown failure"];
    const core = issues.slice(0, 2).map(compressIssue).join("; ");
    const cfg = tier === 1 ? undefined : (this.options.describe ?? ((src: string) => describeConfig(configFromSource(src))))(source);

    const summary = fit([`T${tier} ${kind}: ${core}`, ...(cfg ? [`cfg ${cfg}`] : []), `fix: ${HINTS[kind]}`], REFLECTOR_MAX_CHARS);
    const logPath = this.writeLog(nodeId, result);
    return { kind, tier, summary, ...(logPath && { logPath }) };
  }

  private writeLog(nodeId: string, result: EvalResult): string | undefined {
    if (!this.options.logDir) return undefined;
    const safeId = nodeId.replace(/[^\w-]/g, "_");
    mkdirSync(this.options.logDir, { recursive: true });
    const path = join(this.options.logDir, `${safeId}.log`);
    writeFileSync(path, JSON.stringify({ failure: result.failure, tiers: result.tiers }, null, 2));
    return path;
  }
}
