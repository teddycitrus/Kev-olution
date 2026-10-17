/**
 * Dashboard state: a pure reducer over the two event streams.
 *   EvolveEvent (runner.ts)    -> cycles, champion, Kev decisions, costs, log.
 *   GraphEvent  (graphStore.ts) -> the search DAG, incrementally (never snapshot()).
 * `now` is passed in so reducers stay deterministic under test.
 */

import { configFromSource, describeConfig } from "../agents/kernelTemplate";
import type { GraphEvent, MutationKind, NodeId, NodeStatus } from "../harness/graphStore";
import type { EvolveEvent, EvolveStats, EvolveSummary, RoutingSource } from "../harness/runner";

export interface DagNode {
  id: NodeId;
  parentId?: NodeId;
  generation: number;
  kind: MutationKind;
  strategy: string;
  config: string;
  status: NodeStatus;
  score?: number;
  /** Max abs error vs the reference, from the evaluator (drives the verification badge). */
  maxAbsError?: number;
  kevScore?: number;
}

export interface ShiftMarker {
  cycle: number;
  label: string;
  decidedBy: "kev-27b" | "fallback";
}

export interface LogLine {
  cycle: number;
  kind: "champion" | "plateau" | "shift" | "fail" | "kev" | "info";
  text: string;
}

export interface DashboardState {
  benchmark: string;
  backend: "simulated" | "gpu";
  routing: "kev" | "offline";
  baselineScore: number;
  targetScore: number;
  startedAt?: number;
  now: number;
  cycle: number;
  /** Best TFLOPS at the end of each cycle; cycle 0 is the baseline. */
  series: Array<{ cycle: number; score: number }>;
  shifts: ShiftMarker[];
  champion?: { id: NodeId; score: number; summary: string };
  nodes: Map<NodeId, DagNode>;
  order: NodeId[];
  /** Latest cycle's candidate ids, for the "current generation" panel. */
  lastCandidates: NodeId[];
  kevLatency: Record<RoutingSource, number[]>;
  /** Server-reported model time for network answers. */
  modelLatency: number[];
  /** Candidates routed (Kev or heuristic) and total routing time, for decisions/s. */
  decisions: number;
  decisionMs: number;
  /** Consecutive cycles without a >1% gain (mirrors the PlateauDetector default). */
  stalledCycles: number;
  /** Cycle of the most recent Paradigm Shift; drives the banner. */
  lastShiftCycle?: number;
  kevErrors: number;
  stats?: EvolveStats;
  log: LogLine[];
  summary?: EvolveSummary;
}

const MAX_LOG = 200;

export function initialState(now = 0): DashboardState {
  return {
    benchmark: "", backend: "simulated", routing: "offline", baselineScore: 0, targetScore: 0,
    now, cycle: 0, series: [], shifts: [], nodes: new Map(), order: [], lastCandidates: [],
    kevLatency: { network: [], cache: [], inflight: [], heuristic: [] }, modelLatency: [], decisions: 0, decisionMs: 0, stalledCycles: 0, kevErrors: 0, log: [],
  };
}

function pushLog(s: DashboardState, line: LogLine): void {
  s.log.push(line);
  if (s.log.length > MAX_LOG) s.log.splice(0, s.log.length - MAX_LOG);
}

/** Applies one EvolveEvent in place. Returns the same state for chaining. */
export function reduceEvolve(s: DashboardState, e: EvolveEvent, now: number): DashboardState {
  s.now = now;
  switch (e.type) {
    case "started":
      Object.assign(s, { benchmark: e.benchmark, backend: e.backend, routing: e.routing, baselineScore: e.baselineScore, targetScore: e.targetScore, startedAt: now });
      break;
    case "cycle_started":
      s.cycle = e.cycle;
      s.lastCandidates = [];
      break;
    case "candidates_proposed":
      s.lastCandidates.push(...e.ids);
      break;
    case "kev_decision":
      s.kevLatency[e.source].push(e.latencyMs);
      if (e.modelLatencyMs !== undefined) s.modelLatency.push(e.modelLatencyMs);
      s.decisions += e.kept.length + e.pruned.length;
      s.decisionMs += e.latencyMs;
      if (e.error) {
        s.kevErrors += 1;
        pushLog(s, { cycle: e.cycle, kind: "kev", text: `kev-9b unavailable, heuristic routing (${e.error.slice(0, 60)})` });
      }
      break;
    case "evaluated":
      if (!e.ok) pushLog(s, { cycle: e.cycle, kind: "fail", text: `${e.id} ${e.summary}` });
      break;
    case "champion":
      s.champion = { id: e.id, score: e.score, summary: e.summary };
      if (e.cycle === 0) s.series = [{ cycle: 0, score: e.score }];
      pushLog(s, { cycle: e.cycle, kind: "champion", text: `champion ${e.id} ${e.score.toFixed(1)} TFLOPS (${e.summary})` });
      break;
    case "plateau":
      pushLog(s, { cycle: e.cycle, kind: "plateau", text: `plateau: ${e.signal.stalledCycles} stalled cycles at ${e.signal.bestScore.toFixed(1)} TFLOPS` });
      break;
    case "paradigm_shift": {
      const d = e.directive;
      s.shifts.push({ cycle: e.cycle, label: d.strategyLabel, decidedBy: d.decidedBy });
      s.lastShiftCycle = e.cycle;
      s.stalledCycles = 0;
      if (d.kevLatencyMs !== undefined && d.kevSource) s.kevLatency[d.kevSource].push(d.kevLatencyMs);
      pushLog(s, { cycle: e.cycle, kind: "shift", text: `PARADIGM SHIFT -> ${d.strategyLabel} [${d.decidedBy}] purged ${d.purgedIds.length}` });
      break;
    }
    case "converged":
      pushLog(s, { cycle: e.cycle, kind: "info", text: e.reason });
      break;
    case "cycle_finished": {
      s.stats = e.stats;
      const last = s.series[s.series.length - 1];
      if (s.lastShiftCycle !== e.cycle) {
        const prev = last?.cycle === e.cycle ? s.series[s.series.length - 2] : last;
        s.stalledCycles = prev && e.bestScore <= prev.score * 1.01 ? s.stalledCycles + 1 : 0;
      }
      if (last?.cycle === e.cycle) last.score = e.bestScore;
      else s.series.push({ cycle: e.cycle, score: e.bestScore });
      break;
    }
    case "finished":
      s.summary = e.summary;
      s.stats = e.summary.stats;
      break;
    case "ast_filtered":
      break;
  }
  return s;
}

/** Applies one GraphEvent in place (the DAG panel). */
export function reduceGraph(s: DashboardState, e: GraphEvent): DashboardState {
  const node = (id: NodeId) => s.nodes.get(id);
  switch (e.type) {
    case "node_added": {
      const n = e.node;
      s.nodes.set(n.id, {
        id: n.id, parentId: n.parentIds[0], generation: n.generation, kind: n.mutation.kind, strategy: n.mutation.strategy,
        config: safeDescribe(n.source), status: n.status,
      });
      s.order.push(n.id);
      break;
    }
    case "node_scored": {
      const n = node(e.id);
      if (n) n.kevScore = e.verdict.score;
      break;
    }
    case "node_pruned":
    case "node_failed":
    case "node_evaluated": {
      const n = node(e.id);
      if (!n) break;
      if (e.type === "node_evaluated") {
        n.score = e.metrics.score;
        n.maxAbsError = e.metrics.maxAbsError;
      }
      // Purged is terminal, matching the store.
      if (n.status !== "purged") n.status = e.type === "node_pruned" ? "pruned" : e.type === "node_failed" ? "failed" : "evaluated";
      break;
    }
    case "branch_purged":
      for (const id of e.purgedIds) {
        const n = node(id);
        if (n) n.status = "purged";
      }
      break;
    case "champion_changed":
      break;
  }
  return s;
}

function safeDescribe(source: string): string {
  try {
    return describeConfig(configFromSource(source));
  } catch {
    return "?";
  }
}

export function percentile(values: readonly number[], p: number): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))];
}
