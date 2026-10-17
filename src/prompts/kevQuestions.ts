/**
 * Kev SystemOne question builders. Kev is a classifier: the state is plain
 * text describing the search, and each question is a typed decision about it.
 *
 * kev-9b  -> one "noul" question per candidate: will it beat its parent?
 * kev-27b -> one "choice" question on plateau: which structural rewrite?
 *
 * States must be deterministic (no timestamps, no ids that depend on wall
 * time) so demo-mode rehearsal snapshots replay as exact cache hits.
 */

import type { LineageContext } from "../harness/graphStore";
import type { ChoiceQuestion, NoulQuestion } from "../harness/kevClient";
import type { StructuralOption } from "../tasks/types";

const MAX_DIFF_CHARS = 3000;

/** Kernel-task wording. Kept verbatim: recorded demo caches are keyed on the exact text. */
export const KERNEL_OBJECTIVE = "Task: optimize a Triton flash-attention forward kernel for throughput with zero numerical error.";
export const KERNEL_SHIFT_INSTRUCTIONS =
  "Parameter tuning has stopped improving throughput. Which structural rewrite of the kernel is most likely to break through the plateau on this hardware?";

export interface SearchStateInput {
  benchmarkDescription: string;
  baselineScore: number;
  targetScore: number;
  parentSummary: string;
  parentScore?: number;
  championSummary?: string;
  championScore?: number;
  lineage: LineageContext;
  /** Best TFLOPS per recent cycle, oldest first. */
  recentBest?: number[];
  /** Structural features already applied to the champion. */
  appliedStructural?: string[];
  /** Task wording; defaults reproduce the kernel task exactly. */
  objective?: string;
  noun?: string;
  unit?: string;
  metricName?: string;
}

export function renderSearchState(s: SearchStateInput): string {
  const unit = s.unit ?? "TFLOPS";
  const noun = s.noun ?? "kernel";
  const fmt = (x: number | undefined) => (x === undefined ? "unmeasured" : `${x.toFixed(1)} ${unit}`);
  const lines = [
    s.objective ?? KERNEL_OBJECTIVE,
    s.benchmarkDescription,
    `Baseline: ${fmt(s.baselineScore)}. Target: ${fmt(s.targetScore)}.`,
    `Parent ${noun}: ${s.parentSummary} at ${fmt(s.parentScore)}.`,
  ];
  if (s.championSummary) lines.push(`Best ${noun} so far: ${s.championSummary} at ${fmt(s.championScore)}.`);
  if (s.appliedStructural?.length) lines.push(`Structural changes already applied: ${s.appliedStructural.join(", ")}.`);
  if (s.recentBest?.length) lines.push(`Best ${s.metricName ?? "throughput"} over recent cycles: ${s.recentBest.map((x) => x.toFixed(1)).join(" -> ")}.`);
  const path = s.lineage.ancestors.map((a) => `${a.mutation.strategy}${a.metrics?.score !== undefined ? ` (${a.metrics.score.toFixed(1)})` : ""}`);
  if (path.length) lines.push(`Lineage: ${path.join(" -> ")}.`);
  if (s.lineage.deadEnds.length) {
    lines.push("Recent dead ends:", ...s.lineage.deadEnds.map((d) => `- ${d.strategy}: ${d.failure?.summary ?? d.pruneReason ?? "rejected"}`));
  }
  return lines.join("\n");
}

export function pruneQuestion(label: string, rationale: string, diff: string | undefined): NoulQuestion {
  const clipped = diff && diff.length > MAX_DIFF_CHARS ? `${diff.slice(0, MAX_DIFF_CHARS)}\n...` : diff ?? "";
  return {
    type: "noul",
    instructions:
      `Candidate mutation: ${label}. ${rationale}\n` +
      (clipped ? `Diff against the parent kernel:\n${clipped}\n` : "") +
      "Will this candidate compile, match the reference output exactly, and run faster than its parent kernel?",
  };
}

export type { StructuralOption };

export function paradigmQuestion(options: readonly StructuralOption[], instructions: string = KERNEL_SHIFT_INSTRUCTIONS): ChoiceQuestion {
  const criteria: Record<string, string> = {};
  for (const o of options) criteria[o.id] = `${o.label}: ${o.description}`;
  return {
    type: "choice",
    instructions,
    criteria,
  };
}
