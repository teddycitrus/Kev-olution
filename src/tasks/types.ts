/**
 * ResearchTask: everything the evolution loop needs to know about a problem.
 *
 * The loop, DAG store, Kev routing, plateau detector, paradigm-shift engine,
 * and dashboards are task-agnostic. A task supplies the four task-specific
 * pieces:
 *   1. a score     (evaluate: candidate source -> metrics or a failure)
 *   2. a free gate (staticCheck: cheap validity checks before any Kev call)
 *   3. mutations   (proposeLocal for small edits, structuralOptions +
 *                   proposeShift for the rewrites kev-27b chooses between)
 *   4. Kev wording (objective line and per-candidate question text)
 *
 * Candidates are plain strings (kernel source, a JSON config, a prompt, ...).
 * Higher scores are better. See src/tasks/kernelTask.ts and
 * src/tasks/binPacking.ts for the two implementations.
 */

import type { EvalResult } from "../harness/evaluator";
import type { FidelityTier, LineageContext, NewNodeInput, TrajectoryNode } from "../harness/graphStore";
import type { Rng } from "../harness/rng";

export interface TaskMetric {
  /** Lower-case noun used in Kev states, e.g. "throughput" or "packing efficiency". */
  name: string;
  /** Unit shown next to scores, e.g. "TFLOPS" or "% fill". */
  unit: string;
}

/** One candidate produced by a task's mutator. */
export interface TaskProposal {
  input: NewNodeInput;
  strategy: { id: string; label: string };
}

/** A structural rewrite kev-27b can choose on a plateau. */
export interface StructuralOption {
  id: string;
  label: string;
  description: string;
}

/** How scores were produced. "exact" means computed directly, not modeled. */
export type TaskBackend = "simulated" | "gpu" | "exact";

export interface ResearchTask {
  id: string;
  name: string;
  /** One or two sentences for Kev states and dashboards. */
  description: string;
  /** First line of every Kev state, e.g. "Task: optimize ... with zero numerical error." */
  objective: string;
  /** Noun for one candidate in Kev states and labels ("kernel", "heuristic"). */
  noun: string;
  metric: TaskMetric;
  baselineScore: number;
  targetScore: number;
  baselineSource: string;
  /** Rationale recorded on the seed node. */
  seedRationale: string;
  backend: TaskBackend;
  /** Short badge text for dashboards, e.g. "SIMULATED H100". */
  backendLabel: string;
  /** Cost of each evaluation tier in compute-seconds, for savings estimates. */
  tierCosts: Readonly<Record<FidelityTier, number>>;

  /** Short human label for a candidate, e.g. "128x64 w8 s3 exp2". */
  describe(source: string): string;
  /** Free Tier 1 checks; an empty list means the candidate may proceed. */
  staticCheck(source: string): string[];
  /** Tiers 1-3. Must stop before Tier 3 when the Tier 2 score is below promotionScore. */
  evaluate(source: string, options: { promotionScore: number }): Promise<EvalResult>;

  proposeLocal(parent: TrajectoryNode, lineage: LineageContext, count: number, rng: Rng): Promise<TaskProposal[]>;
  /** Rewrites that still apply to `source`, in fallback priority order. */
  structuralOptions(source: string): StructuralOption[];
  /** Labels of rewrites already applied to `source` (context for kev-27b). */
  appliedStructural(source: string): string[];
  proposeShift(parent: TrajectoryNode, lineage: LineageContext, optionId: string, count: number, rationale?: string): Promise<TaskProposal[]>;

  kev: {
    /** Instructions for the kev-9b "noul" question about one candidate. */
    pruneInstructions(label: string, rationale: string, diff: string | undefined): string;
    /** Instructions for the kev-27b "choice" question on a plateau. */
    shiftInstructions: string;
  };
}
