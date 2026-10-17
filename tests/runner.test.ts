import { describe, expect, test } from "bun:test";
import { Mutator } from "../src/agents/mutator";
import { Reflector } from "../src/agents/reflector";
import { loadBenchmark } from "../src/harness/benchmark";
import { Evaluator } from "../src/harness/evaluator";
import { InMemoryGraphStore } from "../src/harness/inMemoryGraphStore";
import { KevClient, MemoryQuotaStore, QuotaTracker } from "../src/harness/kevClient";
import { ParadigmShiftEngine, PlateauDetector } from "../src/harness/plateauDetector";
import { evolve, type EvolveEvent, type EvolveOptions } from "../src/harness/runner";

const spec = await loadBenchmark("triton_flash_attention");

type Ev<T extends EvolveEvent["type"]> = Extract<EvolveEvent, { type: T }>;

function setup(kev: KevClient | null = null, seed = 0) {
  const store = new InMemoryGraphStore({ now: () => 0 });
  const sources = new Set<string>();
  store.subscribe((e) => {
    if (e.type === "node_added") sources.add(e.node.source);
  });
  const deps = {
    benchmark: spec,
    store,
    evaluator: new Evaluator(spec, { seed }),
    mutator: new Mutator({ benchmarkDescription: spec.description, isKnown: (s) => sources.has(s) }),
    reflector: new Reflector(),
    plateau: new PlateauDetector({ window: 3 }),
    shiftEngine: new ParadigmShiftEngine({ store, benchmark: spec, kev }),
    kev,
  };
  return { deps, store };
}

async function run(kev: KevClient | null, options: EvolveOptions = {}) {
  const { deps, store } = setup(kev, options.seed ?? 0);
  const events: EvolveEvent[] = [];
  const summary = await evolve(deps, { ...options, onEvent: (e) => events.push(e) });
  const of = <T extends EvolveEvent["type"]>(type: T) => events.filter((e): e is Ev<T> => e.type === type);
  return { summary, events, of, store };
}

/** Mock gateway: noul scores by candidate order, and a choice that picks the first offered option. */
function mockKev(opts: { scores?: number[]; fail?: boolean } = {}) {
  const bodies: any[] = [];
  const fetchImpl = (async (_u: string | URL | Request, init: RequestInit = {}) => {
    const body = JSON.parse(String(init.body));
    bodies.push(body);
    if (opts.fail) return new Response(JSON.stringify({ detail: "kev is down" }), { status: 503 });
    const ids = Object.keys(body.questions);
    const answers: Record<string, unknown> = {};
    if (body.questions.decision?.type === "choice") {
      const choice = Object.keys(body.questions.decision.criteria)[0]!;
      answers.decision = { type: "choice", choice, confidence: 0.9, probabilities: { [choice]: 0.9 } };
    } else {
      ids.forEach((id, i) => (answers[id] = { type: "noul", noul: opts.scores?.[i] ?? 0.1 }));
    }
    return new Response(JSON.stringify({ model: body.model, answers, usage: {}, latency_ms: 10 }));
  }) as typeof fetch;
  const kev = new KevClient({ apiKey: "k", fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore()) });
  return { kev, bodies };
}

describe("offline evolve", () => {
  test(
    "seed 0 reaches the target and converges",
    async () => {
      const { summary, events, of, store } = await run(null, { seed: 0, maxCycles: 40 });
      expect(summary.bestScore).toBeGreaterThanOrEqual(140);
      expect(summary.targetReached).toBe(true);
      expect(summary.stopReason).toBe("converged");
      expect(summary.backend).toBe("simulated");
      expect(summary.speedup).toBeCloseTo(summary.bestScore / spec.baselineTflops, 6);
      expect(events[0]).toMatchObject({ type: "started", benchmark: spec.name, backend: "simulated", routing: "offline" });
      expect(events.at(-1)!.type).toBe("finished");
      expect(of("finished")).toHaveLength(1);
      expect(of("paradigm_shift").length).toBeGreaterThanOrEqual(1);
      for (const shift of of("paradigm_shift")) expect(shift.directive.decidedBy).toBe("fallback");
      for (const d of of("kev_decision")) expect(d).toMatchObject({ model: null, source: "heuristic" });
      expect(summary.stats.kevCalls).toEqual({ "kev-9b": 0, "kev-27b": 0 });
      expect(store.champion()!.id).toBe(summary.championId!);
      expect(of("cycle_finished").map((e) => e.cycle)).toEqual(Array.from({ length: summary.cycles }, (_, i) => i + 1));
      const champs = of("champion").map((e) => e.score);
      expect([...champs].sort((a, b) => a - b)).toEqual(champs);
      expect(summary.stats.gpuSecondsSavedEarlyExit).toBeGreaterThan(0);
      expect(summary.stats.evaluated).toBeGreaterThan(summary.cycles);
    },
    60_000,
  );

  test("is deterministic for a fixed seed", async () => {
    const a = await run(null, { seed: 3, maxCycles: 8 });
    const b = await run(null, { seed: 3, maxCycles: 8 });
    expect(b.summary.bestScore).toBe(a.summary.bestScore);
    expect(b.store.snapshot().nodes.map((n) => n.source)).toEqual(a.store.snapshot().nodes.map((n) => n.source));
  });
});

describe("Kev routing", () => {
  test("low kev-9b scores are pruned with a kev-9b reason; the shift is decided by kev-27b", async () => {
    const { kev, bodies } = mockKev({ scores: [0.9, 0.6, 0.1, 0.05, 0.2, 0.3] });
    const { summary, of, store } = await run(kev, { seed: 0, maxCycles: 6 });
    const decisions = of("kev_decision");
    expect(decisions.length).toBeGreaterThan(0);
    for (const d of decisions) {
      expect(d).toMatchObject({ model: "kev-9b", source: "network" });
      expect(d.error).toBeUndefined();
      for (const id of d.pruned) expect(d.scores[id]).toBeDefined();
      expect(d.kept.length).toBeLessThanOrEqual(3);
    }
    const reasons = store.events().filter((e) => e.type === "node_pruned").map((e) => (e as { reason: string }).reason);
    expect(reasons.length).toBeGreaterThan(0);
    for (const r of reasons) expect(r).toContain("kev-9b");
    expect(reasons.some((r) => /kev-9b 0\.\d\d < 0\.35/.test(r))).toBe(true);
    const prunedIds = new Set(decisions.flatMap((d) => d.pruned));
    for (const id of prunedIds) expect(store.getNode(id)!.kev?.model).toBe("kev-9b");
    for (const shift of of("paradigm_shift")) expect(shift.directive).toMatchObject({ decidedBy: "kev-27b", kevSource: "network" });
    expect(summary.stats.kevCalls["kev-9b"]).toBe(decisions.length);
    expect(summary.stats.kevCalls["kev-27b"]).toBe(of("paradigm_shift").length);
    expect(bodies.filter((b) => b.model === "kev-27b")).toHaveLength(of("paradigm_shift").length);
    expect(summary.stats.kevPruned).toBe(reasons.length);
  });

  test("a failing Kev degrades to heuristic routing with the error recorded, and the run completes", async () => {
    const { kev } = mockKev({ fail: true });
    const { summary, of, events } = await run(kev, { seed: 0, maxCycles: 6 });
    const decisions = of("kev_decision");
    expect(decisions.length).toBeGreaterThan(0);
    for (const d of decisions) {
      expect(d.source).toBe("heuristic");
      expect(d.error).toContain("503");
      expect(d.scores).toEqual({});
    }
    for (const shift of of("paradigm_shift")) expect(shift.directive.decidedBy).toBe("fallback");
    expect(events.at(-1)!.type).toBe("finished");
    expect(summary.cycles).toBe(6);
    expect(summary.bestScore).toBeGreaterThan(spec.baselineTflops);
  });
});

describe("stopping", () => {
  test("an already-aborted signal stops before cycle 1", async () => {
    const ac = new AbortController();
    ac.abort();
    const { summary, of } = await run(null, { signal: ac.signal });
    expect(summary).toMatchObject({ stopReason: "aborted", cycles: 0 });
    expect(of("cycle_started")).toHaveLength(0);
    expect(summary.championId).toBeDefined();
  });

  test("aborting mid-run stops after the current cycle", async () => {
    const ac = new AbortController();
    const { deps } = setup();
    const events: EvolveEvent[] = [];
    const summary = await evolve(deps, {
      signal: ac.signal,
      onEvent: (e) => {
        events.push(e);
        if (e.type === "cycle_finished" && e.cycle === 2) ac.abort();
      },
    });
    expect(summary).toMatchObject({ stopReason: "aborted", cycles: 2 });
  });

  test("durationMs 0 stops before cycle 1", async () => {
    const { summary } = await run(null, { durationMs: 0 });
    expect(summary).toMatchObject({ stopReason: "duration", cycles: 0 });
  });

  test("maxCycles is honored", async () => {
    const { summary } = await run(null, { maxCycles: 2 });
    expect(summary).toMatchObject({ stopReason: "max_cycles", cycles: 2 });
  });

  test("onEvent exceptions never stop the search", async () => {
    const { deps } = setup();
    const summary = await evolve(deps, {
      maxCycles: 2,
      onEvent: () => {
        throw new Error("dashboard crashed");
      },
    });
    expect(summary.cycles).toBe(2);
  });
});

describe("kev_decision.modelLatencyMs", () => {
  test("carries Kev's reported latency_ms for network answers and for demo-cache hits", async () => {
    const fetchImpl = (async (_u: string | URL | Request, init: RequestInit = {}) => {
      const body = JSON.parse(String(init.body));
      const answers: Record<string, unknown> = {};
      if (body.questions.decision) {
        const choice = Object.keys(body.questions.decision.criteria)[0]!;
        answers.decision = { type: "choice", choice, probabilities: { [choice]: 1 } };
      } else for (const id of Object.keys(body.questions)) answers[id] = { type: "noul", noul: 0.8 };
      return new Response(JSON.stringify({ model: body.model, answers, usage: {}, latency_ms: 12 }));
    }) as typeof fetch;
    const kev = new KevClient({ apiKey: "k", fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore()), demo: true });

    const first = await run(kev, { seed: 0, maxCycles: 3 });
    const networkDecisions = first.of("kev_decision");
    expect(networkDecisions.length).toBeGreaterThan(0);
    for (const d of networkDecisions) expect(d).toMatchObject({ source: "network", modelLatencyMs: 12 });

    const second = await run(kev, { seed: 0, maxCycles: 3 });
    const cached = second.of("kev_decision");
    expect(cached.length).toBe(networkDecisions.length);
    for (const d of cached) expect(d).toMatchObject({ source: "cache", modelLatencyMs: 12 });
  });

  test("heuristic decisions have no modelLatencyMs", async () => {
    const { of } = await run(null, { seed: 0, maxCycles: 2 });
    for (const d of of("kev_decision")) expect("modelLatencyMs" in d).toBe(false);
  });
});
