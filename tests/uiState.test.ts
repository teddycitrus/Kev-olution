import { describe, expect, test } from "bun:test";
import { renderKernel } from "../src/agents/kernelTemplate";
import { Mutator } from "../src/agents/mutator";
import { Reflector } from "../src/agents/reflector";
import { loadBenchmark } from "../src/harness/benchmark";
import { Evaluator } from "../src/harness/evaluator";
import type { GraphEvent, TrajectoryNode } from "../src/harness/graphStore";
import { InMemoryGraphStore } from "../src/harness/inMemoryGraphStore";
import { ParadigmShiftEngine, PlateauDetector } from "../src/harness/plateauDetector";
import { evolve, type EvolveEvent, type EvolveStats, type EvolveSummary } from "../src/harness/runner";
import { initialState, percentile, reduceEvolve, reduceGraph } from "../src/ui/state";

const spec = await loadBenchmark("triton_flash_attention");

const STATS: EvolveStats = {
  candidates: 6, astRejected: 0, kevPruned: 3, evaluated: 3, gpuSecondsSpent: 40, gpuSecondsSavedEarlyExit: 20,
  gpuSecondsSavedByPruningEstimate: 64.5, kevCalls: { "kev-9b": 1, "kev-27b": 0 }, kevLatencyMs: [12], paradigmShifts: 0,
};

const started: EvolveEvent = { type: "started", benchmark: "triton_flash_attention", backend: "simulated", routing: "kev", baselineScore: 41.2, targetScore: 140 };
const champion = (cycle: number, score: number, id = `n${cycle}`): EvolveEvent => ({ type: "champion", cycle, id, score, summary: "64x32 w4 s1" });
const cycleFinished = (cycle: number, bestScore: number): EvolveEvent => ({ type: "cycle_finished", cycle, bestScore, stats: STATS });
const shift = (cycle: number, extra: Partial<Extract<EvolveEvent, { type: "paradigm_shift" }>["directive"]> = {}): EvolveEvent => ({
  type: "paradigm_shift",
  cycle,
  directive: { strategyId: "tma_descriptors", strategyLabel: "TMA tensor descriptors", decidedBy: "kev-27b", probabilities: {}, rootId: "n1", purgedIds: ["n2", "n3"], rationale: "r", ...extra },
});

function node(id: string, source: string, parentIds: string[] = [], kind: TrajectoryNode["mutation"]["kind"] = parentIds.length ? "local" : "seed"): TrajectoryNode {
  return { id, parentIds, generation: parentIds.length, createdAt: 0, source, mutation: { kind, strategy: `s-${id}`, rationale: "" }, status: "pending" };
}
let seq = 0;
type EventBody<T extends GraphEvent["type"]> = Omit<Extract<GraphEvent, { type: T }>, "seq" | "at">;
const ge = <T extends GraphEvent["type"]>(e: EventBody<T>): GraphEvent => Object.assign({ seq: ++seq, at: 0 }, e) as Extract<GraphEvent, { type: T }>;

describe("reduceEvolve", () => {
  test("started sets meta and startedAt from the injected clock", () => {
    const s = reduceEvolve(initialState(5), started, 1000);
    expect(s).toMatchObject({ benchmark: "triton_flash_attention", backend: "simulated", routing: "kev", baselineScore: 41.2, targetScore: 140, startedAt: 1000, now: 1000 });
  });

  test("the cycle-0 champion seeds the series; cycle_finished appends or updates a repeated cycle", () => {
    const s = initialState();
    reduceEvolve(s, started, 0);
    reduceEvolve(s, champion(0, 41.2), 1);
    expect(s.series).toEqual([{ cycle: 0, score: 41.2 }]);
    expect(s.champion).toEqual({ id: "n0", score: 41.2, summary: "64x32 w4 s1" });
    reduceEvolve(s, cycleFinished(1, 55), 2);
    reduceEvolve(s, cycleFinished(2, 60), 3);
    reduceEvolve(s, cycleFinished(2, 61), 4);
    expect(s.series).toEqual([{ cycle: 0, score: 41.2 }, { cycle: 1, score: 55 }, { cycle: 2, score: 61 }]);
    expect(s.stats).toBe(STATS);
    reduceEvolve(s, champion(3, 70), 5);
    expect(s.series).toHaveLength(3);
    expect(s.champion?.score).toBe(70);
  });

  test("cycle_started resets lastCandidates; candidates_proposed accumulates", () => {
    const s = initialState();
    reduceEvolve(s, { type: "cycle_started", cycle: 1, parentId: "n0", parentSummary: "x" }, 0);
    reduceEvolve(s, { type: "candidates_proposed", cycle: 1, ids: ["n1", "n2"], kind: "local" }, 0);
    reduceEvolve(s, { type: "candidates_proposed", cycle: 1, ids: ["n3"], kind: "paradigm_shift" }, 0);
    expect(s.cycle).toBe(1);
    expect(s.lastCandidates).toEqual(["n1", "n2", "n3"]);
    reduceEvolve(s, { type: "cycle_started", cycle: 2, parentId: "n1", parentSummary: "x" }, 0);
    expect(s.lastCandidates).toEqual([]);
  });

  test("paradigm_shift adds a marker and records kev-27b latency under its source", () => {
    const s = initialState();
    reduceEvolve(s, shift(4, { kevLatencyMs: 17, kevSource: "network" }), 0);
    reduceEvolve(s, shift(9, { decidedBy: "fallback", fallbackReason: "offline" }), 0);
    reduceEvolve(s, shift(12, { kevLatencyMs: 0.2, kevSource: "cache" }), 0);
    expect(s.shifts).toEqual([
      { cycle: 4, label: "TMA tensor descriptors", decidedBy: "kev-27b" },
      { cycle: 9, label: "TMA tensor descriptors", decidedBy: "fallback" },
      { cycle: 12, label: "TMA tensor descriptors", decidedBy: "kev-27b" },
    ]);
    expect(s.kevLatency.network).toEqual([17]);
    expect(s.kevLatency.cache).toEqual([0.2]);
    expect(s.log.filter((l) => l.kind === "shift").map((l) => l.text)).toContain("PARADIGM SHIFT -> TMA tensor descriptors [kev-27b] purged 2");
  });

  test("kev_decision records latency per source; errors increment kevErrors and log", () => {
    const s = initialState();
    const decision = (source: "network" | "heuristic", latencyMs: number, error?: string): EvolveEvent => ({
      type: "kev_decision", cycle: 1, model: "kev-9b", source, latencyMs, kept: [], pruned: [], scores: {}, ...(error && { error }),
    });
    reduceEvolve(s, decision("network", 12), 0);
    reduceEvolve(s, decision("heuristic", 0, `Kev API error 503: ${"x".repeat(100)}`), 0);
    expect(s.kevLatency.network).toEqual([12]);
    expect(s.kevLatency.heuristic).toEqual([0]);
    expect(s.kevErrors).toBe(1);
    const line = s.log.at(-1)!;
    expect(line.kind).toBe("kev");
    expect(line.text).toStartWith("kev-9b unavailable, heuristic routing (Kev API error 503");
    expect(line.text.length).toBeLessThan(110);
  });

  test("failed evaluations, plateaus, and convergence are logged; successes are not", () => {
    const s = initialState();
    const evaluated = (ok: boolean): EvolveEvent => ({ type: "evaluated", cycle: 2, id: "n5", ok, tier: 2, summary: "T2 compile: smem", costGpuSeconds: 1, savedGpuSeconds: 20 });
    reduceEvolve(s, evaluated(true), 0);
    expect(s.log).toHaveLength(0);
    reduceEvolve(s, evaluated(false), 0);
    reduceEvolve(s, { type: "plateau", cycle: 3, signal: { cycle: 3, stalledCycles: 3, bestScore: 89, history: [] } }, 0);
    reduceEvolve(s, { type: "converged", cycle: 4, reason: "nothing left" }, 0);
    reduceEvolve(s, { type: "ast_filtered", cycle: 4, rejected: ["n9"] }, 0);
    expect(s.log.map((l) => [l.kind, l.text])).toEqual([
      ["fail", "n5 T2 compile: smem"],
      ["plateau", "plateau: 3 stalled cycles at 89.0 TFLOPS"],
      ["info", "nothing left"],
    ]);
  });

  test("the log is capped at 200 lines, keeping the newest", () => {
    const s = initialState();
    for (let i = 0; i < 250; i++) reduceEvolve(s, { type: "converged", cycle: i, reason: `r${i}` }, 0);
    expect(s.log).toHaveLength(200);
    expect(s.log[0]!.text).toBe("r50");
    expect(s.log.at(-1)!.text).toBe("r249");
  });

  test("finished stores the summary and its stats", () => {
    const s = initialState();
    const summary: EvolveSummary = { cycles: 3, stopReason: "converged", bestScore: 150, baselineScore: 41.2, speedup: 3.64, targetReached: true, backend: "simulated", stats: STATS };
    reduceEvolve(s, { type: "finished", summary }, 0);
    expect(s.summary).toBe(summary);
    expect(s.stats).toBe(STATS);
  });
});

describe("reduceGraph", () => {
  test("node_added stores the config label, parent, kind, and order", () => {
    const s = initialState();
    reduceGraph(s, ge<"node_added">({ type: "node_added", node: node("n0", spec.baselineSource) }));
    reduceGraph(s, ge<"node_added">({ type: "node_added", node: node("n1", renderKernel({ blockM: 128, blockN: 64, numWarps: 8, numStages: 3, exp2: true, memory: "tma", warpSpecialize: true, persistent: false }), ["n0"], "paradigm_shift") }));
    reduceGraph(s, ge<"node_added">({ type: "node_added", node: node("n2", "not a kernel", ["n0"]) }));
    expect(s.order).toEqual(["n0", "n1", "n2"]);
    expect(s.nodes.get("n0")).toMatchObject({ id: "n0", config: "64x32 w4 s1", status: "pending", kind: "seed", generation: 0 });
    expect(s.nodes.get("n0")!.parentId).toBeUndefined();
    expect(s.nodes.get("n1")).toMatchObject({ parentId: "n0", kind: "paradigm_shift", config: "128x64 w8 s3 exp2 tma ws", strategy: "s-n1" });
    expect(typeof s.nodes.get("n2")!.config).toBe("string");
  });

  test("scored, pruned, failed, and evaluated transitions apply", () => {
    const s = initialState();
    for (const id of ["a", "b", "c"]) reduceGraph(s, ge<"node_added">({ type: "node_added", node: node(id, spec.baselineSource) }));
    reduceGraph(s, ge<"node_scored">({ type: "node_scored", id: "a", verdict: { model: "kev-9b", score: 0.12, latencyMs: 10 } }));
    reduceGraph(s, ge<"node_pruned">({ type: "node_pruned", id: "a", reason: "kev-9b 0.12 < 0.35" }));
    reduceGraph(s, ge<"node_failed">({ type: "node_failed", id: "b", failure: { kind: "compile", tier: 2, summary: "smem" } }));
    reduceGraph(s, ge<"node_evaluated">({ type: "node_evaluated", id: "c", metrics: { tier: 3, score: 88.5, maxAbsError: 0, evalWallMs: 1 } }));
    expect(s.nodes.get("a")).toMatchObject({ status: "pruned", kevScore: 0.12 });
    expect(s.nodes.get("b")!.status).toBe("failed");
    expect(s.nodes.get("c")).toMatchObject({ status: "evaluated", score: 88.5 });
  });

  test("branch_purged is terminal: a later evaluation keeps purged but records score", () => {
    const s = initialState();
    for (const id of ["a", "b"]) reduceGraph(s, ge<"node_added">({ type: "node_added", node: node(id, spec.baselineSource) }));
    reduceGraph(s, ge<"branch_purged">({ type: "branch_purged", rootId: "root", purgedIds: ["a", "b", "ghost"], reason: "plateau" }));
    reduceGraph(s, ge<"node_evaluated">({ type: "node_evaluated", id: "a", metrics: { tier: 3, score: 500, maxAbsError: 0, evalWallMs: 1 } }));
    reduceGraph(s, ge<"node_failed">({ type: "node_failed", id: "b", failure: { kind: "runtime", tier: 3, summary: "late" } }));
    reduceGraph(s, ge<"node_pruned">({ type: "node_pruned", id: "b", reason: "late" }));
    expect(s.nodes.get("a")).toMatchObject({ status: "purged", score: 500 });
    expect(s.nodes.get("b")!.status).toBe("purged");
    expect(s.nodes.has("ghost")).toBe(false);
  });

  test("events for unknown nodes and champion_changed are ignored", () => {
    const s = initialState();
    reduceGraph(s, ge<"node_scored">({ type: "node_scored", id: "x", verdict: { model: "kev-9b", score: 1, latencyMs: 1 } }));
    reduceGraph(s, ge<"node_evaluated">({ type: "node_evaluated", id: "x", metrics: { tier: 1, evalWallMs: 0 } }));
    reduceGraph(s, ge<"champion_changed">({ type: "champion_changed", id: "x", score: 1 }));
    expect(s.nodes.size).toBe(0);
  });
});

describe("recorded offline run", () => {
  test("replaying evolve and graph events reproduces the run's series, shifts, and DAG", async () => {
    const store = new InMemoryGraphStore({ now: () => 0 });
    const known = new Set<string>();
    const s = initialState(0);
    let clock = 0;
    store.subscribe((e) => {
      if (e.type === "node_added") known.add(e.node.source);
      reduceGraph(s, e);
    });
    const events: EvolveEvent[] = [];
    const summary = await evolve(
      {
        benchmark: spec, store, evaluator: new Evaluator(spec), reflector: new Reflector(),
        mutator: new Mutator({ isKnown: (src) => known.has(src) }), plateau: new PlateauDetector(),
        shiftEngine: new ParadigmShiftEngine({ store, benchmark: spec, kev: null }), kev: null,
      },
      { seed: 0, maxCycles: 40, onEvent: (e) => { events.push(e); reduceEvolve(s, e, (clock += 10)); } },
    );
    expect(s.summary).toBe(summary);
    expect(s.startedAt).toBe(10);
    const baselineChampion = events.find((e): e is Extract<EvolveEvent, { type: "champion" }> => e.type === "champion")!;
    expect(baselineChampion.cycle).toBe(0);
    expect(s.series[0]).toEqual({ cycle: 0, score: baselineChampion.score });
    expect(Math.abs(s.series[0]!.score - spec.baselineTflops) / spec.baselineTflops).toBeLessThanOrEqual(0.005);
    expect(s.series.at(-1)!.score).toBe(summary.bestScore);
    expect(s.series.map((p) => p.cycle)).toEqual(Array.from({ length: summary.cycles + 1 }, (_, i) => i));
    for (let i = 1; i < s.series.length; i++) expect(s.series[i]!.score).toBeGreaterThanOrEqual(s.series[i - 1]!.score);
    expect(s.shifts).toHaveLength(summary.stats.paradigmShifts);
    expect(s.champion?.id).toBe(summary.championId!);
    const snap = store.snapshot();
    expect(s.order).toEqual(snap.nodes.map((n) => n.id));
    for (const n of snap.nodes) expect(s.nodes.get(n.id)!.status).toBe(n.status);
    expect(s.kevLatency.heuristic.length).toBe(events.filter((e) => e.type === "kev_decision").length);
  });
});

describe("percentile", () => {
  test("empty list is undefined", () => {
    expect(percentile([], 50)).toBeUndefined();
  });

  test("one value is returned for any p", () => {
    for (const p of [0, 50, 95, 100]) expect(percentile([7], p)).toBe(7);
  });

  test("nearest-rank p50 and p95 of 1..100, independent of input order", () => {
    const values = Array.from({ length: 100 }, (_, i) => 100 - i);
    expect(percentile(values, 50)).toBe(50);
    expect(percentile(values, 95)).toBe(95);
    expect(percentile(values, 100)).toBe(100);
    expect(percentile(values, 0)).toBe(1);
    expect(values[0]).toBe(100);
  });
});

describe("telemetry fields", () => {
  const decision = (o: { source?: "network" | "cache" | "heuristic"; latencyMs: number; kept: number; pruned: number; modelLatencyMs?: number }): EvolveEvent => ({
    type: "kev_decision", cycle: 1, model: "kev-9b", source: o.source ?? "network", latencyMs: o.latencyMs,
    kept: Array.from({ length: o.kept }, (_, i) => `k${i}`), pruned: Array.from({ length: o.pruned }, (_, i) => `p${i}`), scores: {},
    ...(o.modelLatencyMs !== undefined && { modelLatencyMs: o.modelLatencyMs }),
  });

  test("kev_decision accumulates decisions, decisionMs, and modelLatency only when present", () => {
    const s = initialState();
    reduceEvolve(s, decision({ latencyMs: 200, kept: 3, pruned: 3, modelLatencyMs: 120 }), 0);
    reduceEvolve(s, decision({ source: "cache", latencyMs: 0.5, kept: 2, pruned: 1, modelLatencyMs: 118 }), 0);
    reduceEvolve(s, decision({ source: "heuristic", latencyMs: 0, kept: 3, pruned: 0 }), 0);
    expect(s.decisions).toBe(12);
    expect(s.decisionMs).toBe(200.5);
    expect(s.modelLatency).toEqual([120, 118]);
  });

  test("stalledCycles counts cycles with <= 1% gain and resets on an improvement", () => {
    const s = initialState();
    reduceEvolve(s, champion(0, 100), 0);
    reduceEvolve(s, cycleFinished(1, 100.5), 0);
    expect(s.stalledCycles).toBe(1);
    reduceEvolve(s, cycleFinished(2, 101.5), 0);
    expect(s.stalledCycles).toBe(2);
    reduceEvolve(s, cycleFinished(3, 105), 0);
    expect(s.stalledCycles).toBe(0);
    reduceEvolve(s, cycleFinished(4, 105), 0);
    reduceEvolve(s, cycleFinished(4, 105.2), 0);
    expect(s.stalledCycles).toBe(2);
  });

  test("paradigm_shift sets lastShiftCycle and resets stalledCycles; the shift cycle itself is not counted", () => {
    const s = initialState();
    reduceEvolve(s, champion(0, 100), 0);
    reduceEvolve(s, cycleFinished(1, 100), 0);
    reduceEvolve(s, cycleFinished(2, 100), 0);
    expect(s.stalledCycles).toBe(2);
    reduceEvolve(s, shift(3), 0);
    expect(s.lastShiftCycle).toBe(3);
    expect(s.stalledCycles).toBe(0);
    reduceEvolve(s, cycleFinished(3, 100), 0);
    expect(s.stalledCycles).toBe(0);
    reduceEvolve(s, cycleFinished(4, 100), 0);
    expect(s.stalledCycles).toBe(1);
    reduceEvolve(s, shift(9), 0);
    expect(s.lastShiftCycle).toBe(9);
  });
});
