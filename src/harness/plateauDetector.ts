/**
 * Plateau Detector and Paradigm Shift Engine.
 *
 * PlateauDetector watches the champion throughput once per cycle. When the
 * best value has not improved by more than `minRelativeGain` for `window`
 * consecutive cycles (default 3), it fires a PlateauSignal and resets.
 *
 * ParadigmShiftEngine reacts to a signal:
 *   1. Purges the stalled local branch: every descendant of the champion
 *      (the champion and its ancestors survive).
 *   2. Asks kev-27b one choice question over the structural rewrites not yet
 *      applied to the champion. Kev is a classifier, so the "directive" is its
 *      choice among the task's structural options; the task's mutator writes the candidate.
 *   3. Falls back to the fixed priority order when kev-27b is unavailable
 *      (no client, quota exhausted, API error) and says so in `decidedBy`.
 */

import { paradigmQuestion, renderSearchState } from "../prompts/kevQuestions";
import { createKernelTask } from "../tasks/kernelTask";
import type { ResearchTask, StructuralOption } from "../tasks/types";
import type { BenchmarkSpec } from "../types";
import type { GraphStore, NodeId } from "./graphStore";
import type { KevClient, KevResultSource } from "./kevClient";

export interface PlateauConfig {
  /** Consecutive non-improving cycles before firing. Default 3. */
  window?: number;
  /** Improvement below this fraction of the previous best counts as stalled. Default 0.01. */
  minRelativeGain?: number;
}

export interface PlateauSignal {
  cycle: number;
  stalledCycles: number;
  bestScore: number;
  /** Best value per observed cycle in the stalled window, oldest first. */
  history: number[];
}

export class PlateauDetector {
  readonly window: number;
  readonly minRelativeGain: number;
  private best = Number.NEGATIVE_INFINITY;
  private stalled = 0;
  private recent: number[] = [];

  constructor(config: PlateauConfig = {}) {
    this.window = config.window ?? 3;
    this.minRelativeGain = config.minRelativeGain ?? 0.01;
    if (this.window < 1) throw new Error("plateau window must be >= 1");
  }

  /** Records one cycle's best throughput. Returns a signal when a plateau is detected. */
  observe(cycle: number, bestScore: number): PlateauSignal | null {
    const improved = !Number.isFinite(this.best) || bestScore > this.best * (1 + this.minRelativeGain);
    if (improved) {
      this.best = bestScore;
      this.stalled = 0;
      this.recent = [bestScore];
      return null;
    }
    this.best = Math.max(this.best, bestScore);
    this.stalled += 1;
    this.recent.push(bestScore);
    if (this.stalled < this.window) return null;
    const signal: PlateauSignal = { cycle, stalledCycles: this.stalled, bestScore: this.best, history: this.recent.slice(-(this.window + 1)) };
    this.reset();
    return signal;
  }

  /** Clears the stall counter but keeps the best value (call after a shift). */
  reset(): void {
    this.stalled = 0;
    this.recent = Number.isFinite(this.best) ? [this.best] : [];
  }

  get stalledCycles(): number {
    return this.stalled;
  }
}

export interface ParadigmShiftDirective {
  strategy: StructuralOption;
  /** Who made the call. "fallback" means kev-27b was not consulted successfully. */
  decidedBy: "kev-27b" | "fallback";
  probabilities: Record<string, number>;
  kevLatencyMs?: number;
  kevSource?: KevResultSource;
  fallbackReason?: string;
  rootId: NodeId;
  purgedIds: NodeId[];
  /** One line for the dashboard and the structural prompt. */
  rationale: string;
}

export type ShiftOutcome = { kind: "directive"; directive: ParadigmShiftDirective } | { kind: "exhausted"; rootId: NodeId; reason: string };

export interface ParadigmShiftEngineOptions {
  store: GraphStore;
  /** The task whose structural menu and wording to use. */
  task?: ResearchTask;
  /** Kernel shorthand: used to build the kernel task when `task` is omitted. */
  benchmark?: BenchmarkSpec;
  /** null runs offline: always the fallback order. */
  kev: KevClient | null;
}

export class ParadigmShiftEngine {
  /** "<championId>:<strategyId>" pairs already issued; a rewrite is never retried from the same champion. */
  private readonly tried = new Set<string>();

  private readonly task: ResearchTask;

  constructor(private readonly options: ParadigmShiftEngineOptions) {
    if (options.task) this.task = options.task;
    else if (options.benchmark) this.task = createKernelTask(options.benchmark);
    else throw new Error("ParadigmShiftEngine needs a task or a benchmark");
  }

  async shift(signal: PlateauSignal): Promise<ShiftOutcome> {
    const { store, kev } = this.options;
    const task = this.task;
    const champion = store.champion();
    if (!champion) throw new Error("paradigm shift requires a champion");
    const summary = task.describe(champion.source);
    const options = task.structuralOptions(champion.source).filter((s) => !this.tried.has(`${champion.id}:${s.id}`));
    if (options.length === 0) {
      return { kind: "exhausted", rootId: champion.id, reason: `no untried structural rewrites left for ${summary}` };
    }

    const purgedIds = store.purgeBranch(champion.id, `plateau: ${signal.stalledCycles} stalled cycles at ${signal.bestScore.toFixed(1)} ${task.metric.unit}`);
    const base = { rootId: champion.id, purgedIds };

    if (kev) {
      try {
        const state = renderSearchState({
          objective: task.objective,
          noun: task.noun,
          unit: task.metric.unit,
          metricName: task.metric.name,
          benchmarkDescription: task.description,
          baselineScore: task.baselineScore,
          targetScore: task.targetScore,
          parentSummary: summary,
          parentScore: champion.metrics?.score,
          championSummary: summary,
          championScore: champion.metrics?.score,
          lineage: store.lineageContext(champion.id),
          recentBest: signal.history,
          appliedStructural: task.appliedStructural(champion.source),
        });
        const { choice, probabilities, result } = await kev.choose("paradigm_shift", state, paradigmQuestion(options, task.kev.shiftInstructions));
        const strategy = options.find((o) => o.id === choice)!;
        this.tried.add(`${champion.id}:${strategy.id}`);
        return {
          kind: "directive",
          directive: {
            ...base, strategy, decidedBy: "kev-27b", probabilities, kevLatencyMs: result.roundTripMs, kevSource: result.source,
            rationale: `kev-27b chose ${strategy.label} after ${signal.stalledCycles} stalled cycles at ${signal.bestScore.toFixed(1)} ${task.metric.unit}`,
          },
        };
      } catch (err) {
        return { kind: "directive", directive: this.fallback(options[0]!, base, signal, err instanceof Error ? err.message : String(err)) };
      }
    }
    return { kind: "directive", directive: this.fallback(options[0]!, base, signal, "offline (no Kev client)") };
  }

  private fallback(strategy: StructuralOption, base: { rootId: NodeId; purgedIds: NodeId[] }, signal: PlateauSignal, reason: string): ParadigmShiftDirective {
    this.tried.add(`${base.rootId}:${strategy.id}`);
    return {
      ...base, strategy, decidedBy: "fallback", probabilities: {}, fallbackReason: reason,
      rationale: `fallback order picked ${strategy.label} after ${signal.stalledCycles} stalled cycles at ${signal.bestScore.toFixed(1)} ${this.task.metric.unit}`,
    };
  }
}
