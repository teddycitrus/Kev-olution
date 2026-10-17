/**
 * Main evolutionary loop: evolve().
 *
 * One cycle:
 *   1. Select a parent (champion most of the time, else a frontier node).
 *   2. Mutator proposes local candidates; each becomes a pending DAG node.
 *   3. AST filter: Tier 1 static checks run locally (free) before any Kev call.
 *   4. Kev routing (kev-9b, one batched call): a "noul" score per survivor.
 *      Low scores and anything outside the top K are pruned before the sandbox.
 *   5. Survivors run through the 3-tier evaluator. Tier 2 estimates below
 *      promotionRatio x champion are cut before Tier 3.
 *   6. Plateau check. On a signal, the Paradigm Shift engine purges the stalled
 *      branch, kev-27b picks a structural rewrite, and its candidates are
 *      evaluated in the same cycle.
 *
 * Without a Kev client (offline) routing falls back to mutator order and every
 * event says so. Not separate from "simulated": the dashboard must show both
 * the evaluator backend and the Kev decision source.
 */

import type { Mutator } from "../agents/mutator";
import type { Reflector } from "../agents/reflector";
import { renderSearchState } from "../prompts/kevQuestions";
import { createKernelTask } from "../tasks/kernelTask";
import type { ResearchTask, TaskBackend, TaskProposal } from "../tasks/types";
import type { BenchmarkSpec } from "../types";
import type { Evaluator } from "./evaluator";
import type { GraphStore, NodeId, TrajectoryNode } from "./graphStore";
import type { KevClient, KevModel, KevResultSource } from "./kevClient";
import type { ParadigmShiftDirective, ParadigmShiftEngine, PlateauDetector, PlateauSignal } from "./plateauDetector";
import { createRng } from "./rng";

export interface EvolveDeps {
  /** The research task. When omitted, the kernel task is built from benchmark + evaluator + mutator. */
  task?: ResearchTask;
  benchmark?: BenchmarkSpec;
  evaluator?: Evaluator;
  mutator?: Mutator;
  store: GraphStore;
  reflector: Reflector;
  plateau: PlateauDetector;
  shiftEngine: ParadigmShiftEngine;
  /** null = offline routing. */
  kev: KevClient | null;
}

export interface EvolveOptions {
  maxCycles?: number;
  /** Wall-clock budget; the current cycle finishes before stopping. */
  durationMs?: number;
  seed?: number;
  candidatesPerCycle?: number;
  /** Max candidates per cycle that reach the sandbox. */
  keepTopK?: number;
  /** kev-9b scores below this are pruned (the best candidate is always kept). */
  minKevScore?: number;
  /** Tier 2 must reach this fraction of the champion to be promoted to Tier 3. */
  promotionRatio?: number;
  /** Probability of exploring a non-champion frontier node as parent. */
  exploreRate?: number;
  /** Candidates generated per Paradigm Shift directive. */
  shiftCandidates?: number;
  signal?: AbortSignal;
  onEvent?: (event: EvolveEvent) => void;
}

// ---------------------------------------------------------------------------
// Telemetry (dashboard contract)
// ---------------------------------------------------------------------------

export type RoutingSource = KevResultSource | "heuristic";

export type EvolveEvent =
  | {
      type: "started"; benchmark: string; backend: TaskBackend; routing: "kev" | "offline"; baselineScore: number; targetScore: number;
      /** Task metadata for labels and units (absent in older recordings: kernel defaults apply). */
      task?: { id: string; name: string; description: string; noun: string; metricName: string; unit: string; backendLabel: string };
    }
  | { type: "cycle_started"; cycle: number; parentId: NodeId; parentSummary: string }
  | { type: "candidates_proposed"; cycle: number; ids: NodeId[]; kind: "local" | "paradigm_shift"; labels?: Record<NodeId, string> }
  | { type: "ast_filtered"; cycle: number; rejected: NodeId[] }
  | {
      type: "kev_decision"; cycle: number; model: KevModel | null; source: RoutingSource; latencyMs: number;
      /** Kev-reported model time (recorded value on cache hits). */
      modelLatencyMs?: number;
      kept: NodeId[]; pruned: NodeId[]; scores: Record<NodeId, number>; error?: string;
    }
  /** tier: the failing tier when !ok, else the highest tier passed. */
  | { type: "evaluated"; cycle: number; id: NodeId; ok: boolean; tier: number; score?: number; summary: string; costGpuSeconds: number; savedGpuSeconds: number }
  | { type: "champion"; cycle: number; id: NodeId; score: number; summary: string }
  | { type: "plateau"; cycle: number; signal: PlateauSignal }
  | { type: "paradigm_shift"; cycle: number; directive: Omit<ParadigmShiftDirective, "strategy"> & { strategyId: string; strategyLabel: string } }
  | { type: "converged"; cycle: number; reason: string }
  | { type: "cycle_finished"; cycle: number; bestScore: number; stats: EvolveStats }
  | { type: "finished"; summary: EvolveSummary };

export interface EvolveStats {
  candidates: number;
  astRejected: number;
  kevPruned: number;
  evaluated: number;
  gpuSecondsSpent: number;
  /** Skipped tiers inside evaluations (early exit). Measured from the cost table. */
  gpuSecondsSavedEarlyExit: number;
  /** Estimated: each pruned candidate would have cost at least Tier 2 + Tier 3. */
  gpuSecondsSavedByPruningEstimate: number;
  kevCalls: Record<KevModel, number>;
  kevLatencyMs: number[];
  paradigmShifts: number;
}

export interface EvolveSummary {
  cycles: number;
  stopReason: "max_cycles" | "duration" | "aborted" | "converged";
  championId?: NodeId;
  championSummary?: string;
  bestScore: number;
  baselineScore: number;
  speedup: number;
  targetReached: boolean;
  backend: TaskBackend;
  stats: EvolveStats;
}

// ---------------------------------------------------------------------------
// Loop
// ---------------------------------------------------------------------------

export async function evolve(deps: EvolveDeps, options: EvolveOptions = {}): Promise<EvolveSummary> {
  const { store, reflector, plateau, shiftEngine, kev } = deps;
  const task = resolveTask(deps);
  const o = {
    maxCycles: options.maxCycles ?? 40,
    durationMs: options.durationMs ?? Number.POSITIVE_INFINITY,
    candidatesPerCycle: options.candidatesPerCycle ?? 6,
    keepTopK: options.keepTopK ?? 3,
    minKevScore: options.minKevScore ?? 0.35,
    promotionRatio: options.promotionRatio ?? 0.95,
    exploreRate: options.exploreRate ?? 0.25,
    shiftCandidates: options.shiftCandidates ?? 4,
  };
  const rng = createRng(options.seed ?? 0);
  const emit = (e: EvolveEvent) => {
    try {
      options.onEvent?.(e);
    } catch {
      // Telemetry must never stop the search.
    }
  };
  const stats: EvolveStats = {
    candidates: 0, astRejected: 0, kevPruned: 0, evaluated: 0, gpuSecondsSpent: 0, gpuSecondsSavedEarlyExit: 0,
    gpuSecondsSavedByPruningEstimate: 0, kevCalls: { "kev-9b": 0, "kev-27b": 0 }, kevLatencyMs: [], paradigmShifts: 0,
  };
  const started = performance.now();
  const bestScore = () => store.champion()?.metrics?.score ?? 0;

  emit({
    type: "started", benchmark: task.id, backend: task.backend, routing: kev ? "kev" : "offline", baselineScore: task.baselineScore, targetScore: task.targetScore,
    task: { id: task.id, name: task.name, description: task.description, noun: task.noun, metricName: task.metric.name, unit: task.metric.unit, backendLabel: task.backendLabel },
  });

  // Seed: the baseline is generation 0 and must evaluate cleanly.
  const seed = store.addNode({ parentIds: [], source: task.baselineSource, mutation: { kind: "seed", strategy: "baseline", rationale: task.seedRationale } });
  await evaluateNode(seed, 0, false);
  if (!store.champion()) throw new Error(`baseline for ${task.id} did not produce a champion`);

  let cycle = 0;
  let stopReason: EvolveSummary["stopReason"] = "max_cycles";
  let exhausted = false;

  while (cycle < o.maxCycles) {
    if (options.signal?.aborted) { stopReason = "aborted"; break; }
    if (performance.now() - started >= o.durationMs) { stopReason = "duration"; break; }
    cycle += 1;

    const parent = selectParent();
    emit({ type: "cycle_started", cycle, parentId: parent.id, parentSummary: task.describe(parent.source) });
    const proposals = await task.proposeLocal(parent, store.lineageContext(parent.id), o.candidatesPerCycle, rng);
    const nodes = addProposals(proposals, cycle, "local");
    const survivors = astFilter(nodes, cycle);
    const kept = await route(parent, survivors, proposals, cycle);
    for (const node of kept) await evaluateNode(node, cycle, true);

    const signal = plateau.observe(cycle, bestScore());
    if (signal) {
      emit({ type: "plateau", cycle, signal });
      const outcome = await shiftEngine.shift(signal);
      if (outcome.kind === "exhausted") {
        if (exhausted) { stopReason = "converged"; emit({ type: "converged", cycle, reason: outcome.reason }); emitCycle(); break; }
        exhausted = true;
        emit({ type: "converged", cycle, reason: `${outcome.reason}; continuing local search one more window` });
      } else {
        await runShift(outcome.directive, cycle);
      }
    }
    emitCycle();
  }

  const champion = store.champion();
  const summary: EvolveSummary = {
    cycles: cycle,
    stopReason,
    championId: champion?.id,
    championSummary: champion && task.describe(champion.source),
    bestScore: bestScore(),
    baselineScore: task.baselineScore,
    speedup: bestScore() / task.baselineScore,
    targetReached: bestScore() >= task.targetScore,
    backend: task.backend,
    stats,
  };
  emit({ type: "finished", summary });
  return summary;

  // -------------------------------------------------------------------------

  function emitCycle() {
    emit({ type: "cycle_finished", cycle, bestScore: bestScore(), stats: { ...stats, kevCalls: { ...stats.kevCalls }, kevLatencyMs: [...stats.kevLatencyMs] } });
  }

  function selectParent(): TrajectoryNode {
    const champion = store.champion()!;
    if (rng.next() >= o.exploreRate) return champion;
    const others = store.frontier({ limit: 4 }).filter((n) => n.id !== champion.id);
    return others.length ? others[rng.int(others.length)]! : champion;
  }

  function addProposals(proposals: TaskProposal[], c: number, kind: "local" | "paradigm_shift"): TrajectoryNode[] {
    const nodes = proposals.map((p) => store.addNode(p.input));
    stats.candidates += nodes.length;
    if (nodes.length) {
      const labels: Record<NodeId, string> = {};
      for (const n of nodes) labels[n.id] = task.describe(n.source);
      emit({ type: "candidates_proposed", cycle: c, ids: nodes.map((n) => n.id), kind, labels });
    }
    return nodes;
  }

  function astFilter(nodes: TrajectoryNode[], c: number): TrajectoryNode[] {
    const rejected: NodeId[] = [];
    const ok = nodes.filter((n) => {
      const issues = task.staticCheck(n.source);
      if (!issues.length) return true;
      store.recordFailure(n.id, reflector.reflect({ nodeId: n.id, source: n.source, result: tier1Result(issues, task.tierCosts) }));
      rejected.push(n.id);
      return false;
    });
    stats.astRejected += rejected.length;
    if (rejected.length) emit({ type: "ast_filtered", cycle: c, rejected });
    return ok;
  }

  async function route(parent: TrajectoryNode, nodes: TrajectoryNode[], proposals: TaskProposal[], c: number): Promise<TrajectoryNode[]> {
    if (nodes.length === 0) return [];
    const byId = new Map(nodes.map((n) => [n.id, n]));
    let ranked: NodeId[] = nodes.map((n) => n.id);
    const scores: Record<NodeId, number> = {};
    let source: RoutingSource = "heuristic";
    let latencyMs = 0;
    let modelLatencyMs: number | undefined;
    let model: KevModel | null = null;
    let error: string | undefined;

    if (kev) {
      model = "kev-9b";
      try {
        const champion = store.champion();
        const state = renderSearchState({
          objective: task.objective,
          noun: task.noun,
          unit: task.metric.unit,
          metricName: task.metric.name,
          benchmarkDescription: task.description,
          baselineScore: task.baselineScore,
          targetScore: task.targetScore,
          parentSummary: task.describe(parent.source),
          parentScore: parent.metrics?.score,
          championSummary: champion && task.describe(champion.source),
          championScore: champion?.metrics?.score,
          lineage: store.lineageContext(parent.id),
        });
        const candidates = nodes.map((n) => {
          const p = proposals.find((x) => x.input.source === n.source)!;
          return { id: n.id, instructions: task.kev.pruneInstructions(p.strategy.label, p.input.mutation.rationale, n.diff) };
        });
        const { scores: s, result } = await kev.scoreCandidates(state, candidates, "prune");
        source = result.source;
        latencyMs = result.roundTripMs;
        // Kev-reported model time; on cache hits this is the time recorded when Kev first answered.
        modelLatencyMs = result.response.latency_ms;
        if (result.source === "network") stats.kevCalls["kev-9b"] += 1;
        stats.kevLatencyMs.push(latencyMs);
        for (const n of nodes) {
          scores[n.id] = s[n.id]!;
          store.recordVerdict(n.id, { model: "kev-9b", score: s[n.id]!, latencyMs });
        }
        ranked = [...ranked].sort((a, b) => scores[b]! - scores[a]!);
      } catch (err) {
        error = err instanceof Error ? err.message : String(err);
      }
    }

    const kept = ranked.filter((id, i) => i < o.keepTopK && (i === 0 || scores[id] === undefined || scores[id]! >= o.minKevScore));
    const pruned = ranked.filter((id) => !kept.includes(id));
    for (const id of pruned) {
      const why = scores[id] === undefined ? `outside top ${o.keepTopK} (heuristic order)` : scores[id]! < o.minKevScore ? `kev-9b ${scores[id]!.toFixed(2)} < ${o.minKevScore}` : `kev-9b rank > top ${o.keepTopK}`;
      store.markPruned(id, why);
    }
    stats.kevPruned += pruned.length;
    stats.gpuSecondsSavedByPruningEstimate += pruned.length * (task.tierCosts[2] + task.tierCosts[3]);
    emit({ type: "kev_decision", cycle: c, model, source, latencyMs, ...(modelLatencyMs !== undefined && { modelLatencyMs }), kept, pruned, scores, ...(error && { error }) });
    return kept.map((id) => byId.get(id)!);
  }

  async function evaluateNode(node: TrajectoryNode, c: number, promote: boolean): Promise<void> {
    const previous = store.champion()?.id;
    const bar = promote ? bestScore() * o.promotionRatio : 0;
    const result = await task.evaluate(node.source, { promotionScore: bar });
    stats.evaluated += 1;
    stats.gpuSecondsSpent += result.costGpuSeconds;
    stats.gpuSecondsSavedEarlyExit += result.savedGpuSeconds;
    let summary: string;
    if (result.ok && result.metrics) {
      store.recordEvaluation(node.id, result.metrics);
      summary = `${task.describe(node.source)} ${result.metrics.score?.toFixed(1) ?? "?"} ${task.metric.unit}`;
    } else {
      const failure = reflector.reflect({ nodeId: node.id, source: node.source, result });
      store.recordFailure(node.id, failure);
      summary = failure.summary;
    }
    emit({
      type: "evaluated", cycle: c, id: node.id, ok: result.ok, tier: result.failure?.tier ?? result.metrics?.tier ?? 1, score: result.metrics?.score,
      summary, costGpuSeconds: result.costGpuSeconds, savedGpuSeconds: result.savedGpuSeconds,
    });
    const champion = store.champion();
    if (champion && champion.id !== previous) {
      emit({ type: "champion", cycle: c, id: champion.id, score: champion.metrics!.score!, summary: task.describe(champion.source) });
    }
  }

  async function runShift(directive: ParadigmShiftDirective, c: number): Promise<void> {
    stats.paradigmShifts += 1;
    if (directive.decidedBy === "kev-27b" && directive.kevSource === "network") stats.kevCalls["kev-27b"] += 1;
    const { strategy, ...rest } = directive;
    emit({ type: "paradigm_shift", cycle: c, directive: { ...rest, strategyId: strategy.id, strategyLabel: strategy.label } });
    const champion = store.champion()!;
    const proposals = await task.proposeShift(champion, store.lineageContext(champion.id), strategy.id, o.shiftCandidates, directive.rationale);
    const nodes = astFilter(addProposals(proposals, c, "paradigm_shift"), c);
    // kev-27b already chose the direction; every structural candidate is evaluated.
    for (const node of nodes) await evaluateNode(node, c, false);
  }
}

function resolveTask(deps: EvolveDeps): ResearchTask {
  if (deps.task) return deps.task;
  if (deps.benchmark && deps.evaluator && deps.mutator) return createKernelTask(deps.benchmark, deps.evaluator, deps.mutator);
  throw new Error("evolve() needs a task, or benchmark + evaluator + mutator for the kernel task");
}

function tier1Result(issues: string[], costs: ResearchTask["tierCosts"]) {
  return {
    ok: false,
    failure: { kind: "syntax" as const, tier: 1 as const, summary: issues.join("; ") },
    features: null,
    tiers: [{ tier: 1 as const, passed: false, issues, costGpuSeconds: costs[1], wallMs: 0 }],
    costGpuSeconds: costs[1],
    savedGpuSeconds: costs[2] + costs[3],
  };
}
