import { describe, expect, test } from "bun:test";
import { renderKernel, type KernelConfig } from "../src/agents/kernelTemplate";
import { loadBenchmark } from "../src/harness/benchmark";
import type { ExecutionMetrics, MutationInfo } from "../src/harness/graphStore";
import { InMemoryGraphStore } from "../src/harness/inMemoryGraphStore";
import { KevClient, MemoryQuotaStore, QuotaTracker } from "../src/harness/kevClient";
import { ParadigmShiftEngine, PlateauDetector, type PlateauSignal } from "../src/harness/plateauDetector";

const spec = await loadBenchmark("triton_flash_attention");
const BASE: KernelConfig = { blockM: 128, blockN: 64, numWarps: 8, numStages: 3, exp2: true, memory: "pointer", warpSpecialize: false, persistent: false };
const local = (s: string): MutationInfo => ({ kind: "local", strategy: s, rationale: "" });
const metrics = (score: number): ExecutionMetrics => ({ tier: 3, score, maxAbsError: 0, evalWallMs: 1 });
const SIGNAL: PlateauSignal = { cycle: 5, stalledCycles: 3, bestScore: 89, history: [89, 89, 89, 89] };

describe("PlateauDetector", () => {
  test("fires on exactly the 3rd non-improving cycle and resets", () => {
    const d = new PlateauDetector();
    expect(d.observe(1, 80)).toBeNull();
    expect(d.observe(2, 80.4)).toBeNull();
    expect(d.stalledCycles).toBe(1);
    expect(d.observe(3, 80.4)).toBeNull();
    const signal = d.observe(4, 80.7);
    expect(signal).toEqual({ cycle: 4, stalledCycles: 3, bestScore: 80.7, history: [80, 80.4, 80.4, 80.7] });
    expect(d.stalledCycles).toBe(0);
    expect(d.observe(5, 80.7)).toBeNull();
    expect(d.observe(6, 80.7)).toBeNull();
    expect(d.observe(7, 80.7)?.cycle).toBe(7);
  });

  test("a +0.5% gain counts as a stall; more than +1% resets the window", () => {
    const d = new PlateauDetector();
    d.observe(1, 100);
    expect(d.observe(2, 100.5)).toBeNull();
    expect(d.observe(3, 101.0)).toBeNull();
    expect(d.stalledCycles).toBe(2);
    expect(d.observe(4, 102.2)).toBeNull();
    expect(d.stalledCycles).toBe(0);
  });

  test("custom window and threshold", () => {
    const d = new PlateauDetector({ window: 1, minRelativeGain: 0.1 });
    d.observe(1, 50);
    expect(d.observe(2, 54)?.stalledCycles).toBe(1);
    expect(() => new PlateauDetector({ window: 0 })).toThrow("window");
  });

  test("best never decreases when throughput dips", () => {
    const d = new PlateauDetector();
    d.observe(1, 90);
    d.observe(2, 70);
    d.observe(3, 60);
    expect(d.observe(4, 50)?.bestScore).toBe(90);
  });
});

/** root -> champ (evaluated champion) -> [c1 -> c2], root -> sibling. */
function scenario(championConfig: KernelConfig = BASE) {
  const store = new InMemoryGraphStore({ now: () => 0 });
  const root = store.addNode({ parentIds: [], source: spec.baselineSource, mutation: { kind: "seed", strategy: "baseline", rationale: "" } });
  store.recordEvaluation(root.id, metrics(41.2));
  const champ = store.addNode({ parentIds: [root.id], source: renderKernel(championConfig), mutation: local("tune") });
  store.recordEvaluation(champ.id, metrics(89));
  const c1 = store.addNode({ parentIds: [champ.id], source: "c1", mutation: local("c1") });
  const c2 = store.addNode({ parentIds: [c1.id], source: "c2", mutation: local("c2") });
  const sibling = store.addNode({ parentIds: [root.id], source: "s", mutation: local("s") });
  return { store, root, champ, c1, c2, sibling };
}

function mockKev(responder: (body: any) => Response | Promise<Response>) {
  const calls: any[] = [];
  const fetchImpl = (async (_u: string | URL | Request, init: RequestInit = {}) => {
    const body = JSON.parse(String(init.body));
    calls.push(body);
    return responder(body);
  }) as typeof fetch;
  const kev = new KevClient({ apiKey: "k", fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore()) });
  return { kev, calls };
}

const choiceReply = (choice: string, probabilities: Record<string, number> = { [choice]: 0.8 }) =>
  new Response(JSON.stringify({ model: "kev-27b", answers: { decision: { type: "choice", choice, confidence: 0.8, probabilities } }, usage: {}, latency_ms: 17 }));

describe("ParadigmShiftEngine", () => {
  test("kev-27b decides and only the champion's descendants are purged", async () => {
    const { store, champ, c1, c2, sibling, root } = scenario();
    const { kev, calls } = mockKev(() => choiceReply("tma_descriptors", { tma_descriptors: 0.8, block_ptr_layout: 0.1, warp_specialization: 0.05, persistent_kernel: 0.05 }));
    const outcome = await new ParadigmShiftEngine({ store, benchmark: spec, kev }).shift(SIGNAL);
    expect(outcome.kind).toBe("directive");
    if (outcome.kind !== "directive") return;
    const d = outcome.directive;
    expect(d).toMatchObject({ decidedBy: "kev-27b", rootId: champ.id, kevSource: "network" });
    expect(d.strategy.id).toBe("tma_descriptors");
    expect(d.probabilities.tma_descriptors).toBe(0.8);
    expect(d.rationale).toContain("kev-27b chose TMA tensor descriptors");
    expect(new Set(d.purgedIds)).toEqual(new Set([c1.id, c2.id]));
    expect(store.getNode(sibling.id)!.status).toBe("pending");
    expect(store.getNode(root.id)!.status).toBe("evaluated");
    expect(store.getNode(champ.id)!.status).toBe("evaluated");
    expect(calls).toHaveLength(1);
    expect(calls[0].model).toBe("kev-27b");
    expect(Object.keys(calls[0].questions.decision.criteria)).toEqual(["block_ptr_layout", "tma_descriptors", "warp_specialization", "persistent_kernel"]);
    expect(calls[0].state).toContain("Best throughput over recent cycles: 89.0 -> 89.0 -> 89.0 -> 89.0.");
  });

  test("an API error falls back to priority order with a reason", async () => {
    const { store } = scenario();
    const { kev } = mockKev(() => new Response(JSON.stringify({ detail: "gpu on fire" }), { status: 500 }));
    const outcome = await new ParadigmShiftEngine({ store, benchmark: spec, kev }).shift(SIGNAL);
    if (outcome.kind !== "directive") throw new Error("expected a directive");
    expect(outcome.directive).toMatchObject({ decidedBy: "fallback", probabilities: {} });
    expect(outcome.directive.strategy.id).toBe("block_ptr_layout");
    expect(outcome.directive.fallbackReason).toContain("500");
    expect(outcome.directive.fallbackReason).toContain("gpu on fire");
  });

  test("a choice outside the offered options falls back", async () => {
    const { store } = scenario();
    const { kev } = mockKev(() => choiceReply("rewrite_in_rust"));
    const outcome = await new ParadigmShiftEngine({ store, benchmark: spec, kev }).shift(SIGNAL);
    if (outcome.kind !== "directive") throw new Error("expected a directive");
    expect(outcome.directive.decidedBy).toBe("fallback");
    expect(outcome.directive.fallbackReason).toContain("not one of");
  });

  test("offline engine uses the fallback order and says so", async () => {
    const { store } = scenario();
    const outcome = await new ParadigmShiftEngine({ store, benchmark: spec, kev: null }).shift(SIGNAL);
    if (outcome.kind !== "directive") throw new Error("expected a directive");
    expect(outcome.directive.fallbackReason).toBe("offline (no Kev client)");
  });

  test("repeated shifts never repeat a (champion, strategy) pair, then report exhausted", async () => {
    const { store, champ } = scenario();
    const { kev, calls } = mockKev(() => choiceReply("tma_descriptors"));
    const engine = new ParadigmShiftEngine({ store, benchmark: spec, kev });
    const picked: string[] = [];
    for (let i = 0; i < 4; i++) {
      const o = await engine.shift(SIGNAL);
      if (o.kind !== "directive") throw new Error(`shift ${i} exhausted early`);
      expect(o.directive.rootId).toBe(champ.id);
      picked.push(o.directive.strategy.id);
    }
    expect(picked[0]).toBe("tma_descriptors");
    expect(new Set(picked).size).toBe(4);
    for (const body of calls.slice(1)) expect(Object.keys(body.questions.decision.criteria)).not.toContain("tma_descriptors");
    const done = await engine.shift(SIGNAL);
    expect(done).toMatchObject({ kind: "exhausted", rootId: champ.id });
  });

  test("an exhausted champion returns exhausted without purging or calling Kev", async () => {
    const { store, c1 } = scenario({ ...BASE, memory: "tma", warpSpecialize: true, persistent: true });
    const { kev, calls } = mockKev(() => choiceReply("tma_descriptors"));
    const before = store.events().length;
    const outcome = await new ParadigmShiftEngine({ store, benchmark: spec, kev }).shift(SIGNAL);
    expect(outcome.kind).toBe("exhausted");
    if (outcome.kind === "exhausted") expect(outcome.reason).toContain("128x64 w8 s3 exp2 tma ws persistent");
    expect(store.events().length).toBe(before);
    expect(store.getNode(c1.id)!.status).toBe("pending");
    expect(calls).toHaveLength(0);
  });

  test("throws without a champion", async () => {
    const store = new InMemoryGraphStore();
    await expect(new ParadigmShiftEngine({ store, benchmark: spec, kev: null }).shift(SIGNAL)).rejects.toThrow("requires a champion");
  });
});
