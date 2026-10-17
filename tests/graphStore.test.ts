import { describe, expect, test } from "bun:test";
import { MAX_FAILURE_SUMMARY_CHARS, type ExecutionMetrics, type GraphEvent, type MutationInfo } from "../src/harness/graphStore";
import { InMemoryGraphStore, replayGraphStore } from "../src/harness/inMemoryGraphStore";

const seed: MutationInfo = { kind: "seed", strategy: "baseline", rationale: "reference kernel" };
const local = (strategy: string): MutationInfo => ({ kind: "local", strategy, rationale: "tune" });
const metrics = (score: number, maxAbsError = 0): ExecutionMetrics => ({ tier: 3, score, maxAbsError, evalWallMs: 5 });

function makeStore() {
  let t = 1000;
  return new InMemoryGraphStore({ now: () => t++ });
}

function seeded() {
  const store = makeStore();
  const root = store.addNode({ parentIds: [], source: "root", mutation: seed });
  return { store, root };
}

describe("addNode", () => {
  test("assigns ids, generations, and timestamps", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", diff: "@@", mutation: local("tile") });
    const b = store.addNode({ parentIds: [a.id], source: "b", mutation: local("warps") });
    const c = store.addNode({ parentIds: [root.id], source: "c", mutation: local("stages") });
    const x = store.addNode({ parentIds: [b.id, c.id], source: "x", mutation: { kind: "crossover", strategy: "mix", rationale: "" } });
    expect(root.generation).toBe(0);
    expect(a.generation).toBe(1);
    expect(b.generation).toBe(2);
    expect(x.generation).toBe(3);
    expect(new Set([root.id, a.id, b.id, c.id, x.id]).size).toBe(5);
    expect(a.status).toBe("pending");
    expect(a.diff).toBe("@@");
    expect(b.createdAt).toBeGreaterThan(a.createdAt);
    expect(store.parents(x.id).map((n) => n.id)).toEqual([b.id, c.id]);
    expect(store.children(root.id).map((n) => n.id)).toEqual([a.id, c.id]);
  });

  test("rejects unknown parents without mutating the store", () => {
    const { store } = seeded();
    const before = store.events().length;
    expect(() => store.addNode({ parentIds: ["ghost"], source: "", mutation: local("x") })).toThrow("unknown node ghost");
    expect(store.events().length).toBe(before);
  });

  test("rejects parent lists that would break the DAG or the mutation contract", () => {
    const { store, root } = seeded();
    expect(() => store.addNode({ parentIds: [root.id, root.id], source: "", mutation: { kind: "crossover", strategy: "", rationale: "" } })).toThrow("duplicate");
    expect(() => store.addNode({ parentIds: [root.id], source: "", mutation: seed })).toThrow("seed");
    expect(() => store.addNode({ parentIds: [], source: "", mutation: local("x") })).toThrow("local");
    expect(() => store.addNode({ parentIds: [root.id], source: "", mutation: { kind: "crossover", strategy: "", rationale: "" } })).toThrow("crossover");
  });

  test("cycles are impossible: nodes only reference existing parents and ids are never reused", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("x") });
    expect(store.getNode(root.id)!.parentIds).toEqual([]);
    expect(() => store.addNode({ parentIds: [`n${2}`], source: "", mutation: local("self") })).toThrow("unknown node");
    expect(store.lineage(a.id).map((n) => n.id)).toEqual([root.id, a.id]);
  });
});

describe("events", () => {
  test("every write appends one event with monotonic seq and notifies subscribers", () => {
    const { store, root } = seeded();
    const seen: GraphEvent[] = [];
    const unsubscribe = store.subscribe((e) => seen.push(e));
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("x") });
    store.recordVerdict(a.id, { model: "kev-9b", score: 0.7, latencyMs: 12 });
    store.markPruned(a.id, "low score");
    const b = store.addNode({ parentIds: [root.id], source: "b", mutation: local("y") });
    store.recordFailure(b.id, { kind: "compile", tier: 2, summary: "smem" });
    store.purgeBranch(root.id, "plateau");
    expect(seen.map((e) => e.type)).toEqual(["node_added", "node_scored", "node_pruned", "node_added", "node_failed", "branch_purged"]);
    const seqs = store.events().map((e) => e.seq);
    expect(seqs).toEqual(seqs.map((_, i) => i + 1));
    expect(store.events(3).map((e) => e.seq)).toEqual([4, 5, 6, 7]);
    unsubscribe();
    store.addNode({ parentIds: [root.id], source: "c", mutation: local("z") });
    expect(seen).toHaveLength(6);
  });

  test("returned events are copies", () => {
    const { store } = seeded();
    const [first] = store.events();
    (first as Extract<GraphEvent, { type: "node_added" }>).node.status = "purged";
    expect(store.snapshot().nodes[0]!.status).toBe("pending");
  });

  test("writes against unknown nodes throw", () => {
    const store = makeStore();
    expect(() => store.recordVerdict("x", { model: "kev-9b", score: 0.5, latencyMs: 1 })).toThrow("unknown node");
    expect(() => store.markPruned("x", "")).toThrow("unknown node");
    expect(() => store.recordEvaluation("x", metrics(1))).toThrow("unknown node");
    expect(() => store.recordFailure("x", { kind: "syntax", tier: 1, summary: "" })).toThrow("unknown node");
    expect(() => store.purgeBranch("x", "")).toThrow("unknown node");
  });

  test("rejects Kev scores outside [0, 1]", () => {
    const { store, root } = seeded();
    expect(() => store.recordVerdict(root.id, { model: "kev-9b", score: 1.5, latencyMs: 1 })).toThrow("outside");
    expect(() => store.recordVerdict(root.id, { model: "kev-9b", score: Number.NaN, latencyMs: 1 })).toThrow("outside");
  });
});

describe("champion", () => {
  test("tracks the highest score with zero error and emits champion_changed only on improvement", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const b = store.addNode({ parentIds: [root.id], source: "b", mutation: local("b") });
    const c = store.addNode({ parentIds: [root.id], source: "c", mutation: local("c") });
    const d = store.addNode({ parentIds: [root.id], source: "d", mutation: local("d") });

    store.recordEvaluation(root.id, metrics(41));
    store.recordEvaluation(a.id, metrics(60));
    store.recordEvaluation(b.id, metrics(200, 3e-3));
    store.recordEvaluation(c.id, metrics(55));
    store.recordEvaluation(d.id, { tier: 1, evalWallMs: 1 });

    const changes = store.events().filter((e) => e.type === "champion_changed");
    expect(changes).toHaveLength(2);
    expect(changes[0]).toMatchObject({ id: root.id, score: 41 });
    expect(changes[0]).not.toHaveProperty("previousId");
    expect(changes[1]).toMatchObject({ id: a.id, previousId: root.id, score: 60 });
    expect(store.champion()?.id).toBe(a.id);
    expect(store.getNode(b.id)!.status).toBe("evaluated");
  });

  test("equal score does not replace the champion", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    store.recordEvaluation(root.id, metrics(50));
    store.recordEvaluation(a.id, metrics(50));
    expect(store.champion()?.id).toBe(root.id);
  });

  test("no champion until an error-free evaluation exists", () => {
    const { store, root } = seeded();
    expect(store.champion()).toBeUndefined();
    store.recordEvaluation(root.id, metrics(80, 1e-4));
    expect(store.champion()).toBeUndefined();
  });
});

describe("failures and pruning", () => {
  test("recordFailure truncates the summary and sets status", () => {
    const { store, root } = seeded();
    store.recordFailure(root.id, { kind: "runtime", tier: 2, summary: "x".repeat(1000), logPath: "/tmp/log" });
    const node = store.getNode(root.id)!;
    expect(node.status).toBe("failed");
    expect(node.failure!.summary.length).toBe(MAX_FAILURE_SUMMARY_CHARS);
    expect(node.failure!.logPath).toBe("/tmp/log");
    const event = store.events().at(-1) as Extract<GraphEvent, { type: "node_failed" }>;
    expect(event.failure.summary.length).toBe(MAX_FAILURE_SUMMARY_CHARS);
  });

  test("short summaries are stored unchanged", () => {
    const { store, root } = seeded();
    store.recordFailure(root.id, { kind: "numerical", tier: 2, summary: "nan" });
    expect(store.getNode(root.id)!.failure!.summary).toBe("nan");
  });

  test("markPruned and recordVerdict update the node", () => {
    const { store, root } = seeded();
    store.recordVerdict(root.id, { model: "kev-9b", score: 0.1, latencyMs: 14 });
    store.markPruned(root.id, "kev score 0.10");
    expect(store.getNode(root.id)).toMatchObject({ status: "pruned", kev: { model: "kev-9b", score: 0.1, latencyMs: 14 } });
  });
});

describe("purgeBranch", () => {
  test("marks all descendants, not the root, and returns their ids", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const b = store.addNode({ parentIds: [a.id], source: "b", mutation: local("b") });
    const c = store.addNode({ parentIds: [a.id], source: "c", mutation: local("c") });
    const other = store.addNode({ parentIds: [root.id], source: "o", mutation: local("o") });
    const merge = store.addNode({ parentIds: [b.id, c.id], source: "m", mutation: { kind: "crossover", strategy: "m", rationale: "" } });

    const purged = store.purgeBranch(a.id, "plateau for 5 generations");
    expect(new Set(purged)).toEqual(new Set([b.id, c.id, merge.id]));
    expect(purged).toHaveLength(3);
    expect(store.getNode(a.id)!.status).toBe("pending");
    expect(store.getNode(other.id)!.status).toBe("pending");
    for (const id of purged) expect(store.getNode(id)!.status).toBe("purged");
    const event = store.events().at(-1)!;
    expect(event).toMatchObject({ type: "branch_purged", rootId: a.id, reason: "plateau for 5 generations" });
  });

  test("skips the champion and all of its ancestors, but purges their other descendants", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const side = store.addNode({ parentIds: [root.id], source: "s", mutation: local("s") });
    const sideChild = store.addNode({ parentIds: [side.id], source: "sc", mutation: local("sc") });
    const champ = store.addNode({ parentIds: [a.id, sideChild.id], source: "c", mutation: { kind: "crossover", strategy: "c", rationale: "" } });
    const champChild = store.addNode({ parentIds: [champ.id], source: "cc", mutation: local("cc") });
    const dead = store.addNode({ parentIds: [a.id], source: "d", mutation: local("d") });
    store.recordEvaluation(a.id, metrics(60));
    store.recordFailure(dead.id, { kind: "compile", tier: 2, summary: "smem" });
    store.markPruned(champChild.id, "kev 0.1");
    store.recordEvaluation(champ.id, metrics(90));
    expect(store.champion()?.id).toBe(champ.id);

    const purged = store.purgeBranch(root.id, "plateau");
    expect(new Set(purged)).toEqual(new Set([dead.id, champChild.id]));
    expect(store.getNode(champ.id)!.status).toBe("evaluated");
    expect(store.getNode(a.id)!.status).toBe("evaluated");
    expect(store.getNode(side.id)!.status).toBe("pending");
    expect(store.getNode(sideChild.id)!.status).toBe("pending");
    expect(store.champion()?.id).toBe(champ.id);
    expect(store.getNode(dead.id)).toMatchObject({ status: "purged", failure: { kind: "compile", summary: "smem" } });
    expect(store.lineageContext(champ.id).deadEnds).toEqual([]);
    expect(store.frontier({ limit: 10, includePending: true }).map((n) => n.id)).toContain(champ.id);
  });

  test("already purged nodes are not reported again", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    expect(store.purgeBranch(root.id, "first")).toEqual([a.id]);
    expect(store.purgeBranch(root.id, "second")).toEqual([]);
  });

  test("purging a leaf emits an event with no ids", () => {
    const { store, root } = seeded();
    expect(store.purgeBranch(root.id, "noop")).toEqual([]);
    expect(store.events().at(-1)!.type).toBe("branch_purged");
  });
});

describe("purged is terminal", () => {
  function purgedNode() {
    const { store, root } = seeded();
    store.recordEvaluation(root.id, metrics(41));
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    store.addNode({ parentIds: [root.id], source: "keep", mutation: local("keep") });
    store.purgeBranch(root.id, "plateau");
    expect(store.getNode(a.id)!.status).toBe("purged");
    return { store, root, a };
  }

  test("a late evaluation is stored but keeps status purged and never changes the champion", () => {
    const { store, root, a } = purgedNode();
    const before = store.events().length;
    store.recordEvaluation(a.id, metrics(500));
    const node = store.getNode(a.id)!;
    expect(node.status).toBe("purged");
    expect(node.metrics?.score).toBe(500);
    expect(store.champion()?.id).toBe(root.id);
    expect(store.events().slice(before).map((e) => e.type)).toEqual(["node_evaluated"]);
    expect(store.bestByGeneration().at(-1)!.score).toBe(41);
    expect(store.frontier({ limit: 10, includePending: true }).map((n) => n.id)).not.toContain(a.id);
  });

  test("late failure, verdict, and prune keep status purged and store their data", () => {
    const { store, a } = purgedNode();
    store.recordFailure(a.id, { kind: "timeout", tier: 3, summary: "late" });
    store.recordVerdict(a.id, { model: "kev-9b", score: 0.4, latencyMs: 9 });
    store.markPruned(a.id, "late prune");
    expect(store.getNode(a.id)).toMatchObject({ status: "purged", failure: { summary: "late" }, kev: { score: 0.4 } });
    expect(store.snapshot().counts.purged).toBe(2);
  });

  test("replay of late writes on purged nodes is equivalent", () => {
    const { store, a } = purgedNode();
    store.recordEvaluation(a.id, metrics(500));
    store.recordFailure(a.id, { kind: "runtime", tier: 3, summary: "x" });
    const replayed = replayGraphStore(store.events());
    expect(replayed.snapshot()).toEqual(store.snapshot());
    expect(replayed.bestByGeneration()).toEqual(store.bestByGeneration());
  });
});

describe("frontier", () => {
  test("excludes pruned, purged, and failed nodes and pending by default", () => {
    const { store, root, a, b } = frontierStoreWithoutPurge();
    expect(store.frontier({ limit: 10 }).map((n) => n.id)).toEqual([b, a, root]);
  });

  test("includePending ranks by score then Kev score", () => {
    const { store, root, a, b, c } = frontierStoreWithoutPurge();
    expect(store.frontier({ limit: 10, includePending: true }).map((n) => n.id)).toEqual([b, a, root, c]);
    expect(store.frontier({ limit: 2, includePending: true }).map((n) => n.id)).toEqual([b, a]);
  });

  test("custom ranker", () => {
    const { store, root, a, b } = frontierStoreWithoutPurge();
    expect(store.frontier({ limit: 10, rankBy: (n) => -(n.metrics?.score ?? 0) }).map((n) => n.id)).toEqual([root, a, b]);
  });

  test("purged nodes leave the frontier; the champion stays", () => {
    const { store, root, b } = frontierStoreWithoutPurge();
    expect(store.champion()?.id).toBe(b);
    store.purgeBranch(root, "plateau");
    expect(store.frontier({ limit: 10, includePending: true }).map((n) => n.id)).toEqual([b, root]);
  });

  function frontierStoreWithoutPurge() {
    const { store, root } = seeded();
    const [a, b, c, d, e] = ["a", "b", "c", "d", "e"].map((s) => store.addNode({ parentIds: [root.id], source: s, mutation: local(s) }).id) as [
      string,
      string,
      string,
      string,
      string,
    ];
    store.recordEvaluation(root.id, metrics(41));
    store.recordEvaluation(a, metrics(70));
    store.recordEvaluation(b, metrics(90));
    store.recordVerdict(c, { model: "kev-9b", score: 0.9, latencyMs: 10 });
    store.markPruned(d, "low");
    store.recordFailure(e, { kind: "compile", tier: 2, summary: "smem" });
    return { store, root: root.id, a, b, c };
  }
});

describe("lineage and context", () => {
  test("lineage follows first parents root to target", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const side = store.addNode({ parentIds: [root.id], source: "s", mutation: local("s") });
    const x = store.addNode({ parentIds: [a.id, side.id], source: "x", mutation: { kind: "crossover", strategy: "x", rationale: "" } });
    expect(store.lineage(x.id).map((n) => n.id)).toEqual([root.id, a.id, x.id]);
    expect(() => store.lineage("nope")).toThrow("unknown node");
  });

  test("lineageContext lists failed and pruned children of ancestors, newest first", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("tile 128") });
    const deadRoot = store.addNode({ parentIds: [root.id], source: "d1", mutation: local("fp8") });
    const deadA = store.addNode({ parentIds: [a.id], source: "d2", mutation: local("stages 6") });
    const okSibling = store.addNode({ parentIds: [a.id], source: "ok", mutation: local("warps") });
    const target = store.addNode({ parentIds: [a.id], source: "t", mutation: local("exp2") });
    store.recordEvaluation(a.id, metrics(70));
    store.recordFailure(deadRoot.id, { kind: "numerical", tier: 2, summary: "err 2.4e-2" });
    store.markPruned(deadA.id, "kev score 0.05");
    store.recordEvaluation(okSibling.id, metrics(72));

    const ctx = store.lineageContext(target.id);
    expect(ctx.target).toBe(target.id);
    expect(ctx.ancestors.map((n) => n.id)).toEqual([root.id, a.id, target.id]);
    expect(ctx.ancestors[1]!.metrics?.score).toBe(70);
    expect(ctx.deadEnds).toEqual([
      { id: deadA.id, strategy: "stages 6", pruneReason: "kev score 0.05" },
      { id: deadRoot.id, strategy: "fp8", failure: { kind: "numerical", tier: 2, summary: "err 2.4e-2" } },
    ]);
    expect(ctx.approxTokens).toBe(Math.ceil(JSON.stringify({ ancestors: ctx.ancestors, deadEnds: ctx.deadEnds }).length / 4));
    expect(store.lineageContext(target.id, 1).deadEnds.map((d) => d.id)).toEqual([deadA.id]);
    expect(store.lineageContext(target.id, 0).deadEnds).toEqual([]);
  });
});

describe("bestByGeneration and snapshot", () => {
  test("bestByGeneration is the running best error-free score per generation", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const b = store.addNode({ parentIds: [a.id], source: "b", mutation: local("b") });
    const c = store.addNode({ parentIds: [b.id], source: "c", mutation: local("c") });
    store.recordEvaluation(root.id, metrics(41));
    store.recordEvaluation(a.id, metrics(39));
    store.recordEvaluation(b.id, metrics(300, 1e-2));
    store.recordEvaluation(c.id, metrics(88));
    expect(store.bestByGeneration()).toEqual([
      { generation: 0, score: 41 },
      { generation: 1, score: 41 },
      { generation: 2, score: 41 },
      { generation: 3, score: 88 },
    ]);
  });

  test("snapshot has nodes, edges, counts, and champion", () => {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const b = store.addNode({ parentIds: [root.id], source: "b", mutation: local("b") });
    const x = store.addNode({ parentIds: [a.id, b.id], source: "x", mutation: { kind: "crossover", strategy: "x", rationale: "" } });
    store.recordEvaluation(a.id, metrics(50));
    store.markPruned(b.id, "low");
    const snap = store.snapshot();
    expect(snap.nodes.map((n) => n.id)).toEqual([root.id, a.id, b.id, x.id]);
    expect(snap.edges).toEqual([
      { from: root.id, to: a.id, kind: "local" },
      { from: root.id, to: b.id, kind: "local" },
      { from: a.id, to: x.id, kind: "crossover" },
      { from: b.id, to: x.id, kind: "crossover" },
    ]);
    expect(snap.counts).toEqual({ pending: 2, pruned: 1, evaluated: 1, failed: 0, purged: 0 });
    expect(snap.championId).toBe(a.id);
    expect(snap.generation).toBe(2);
    expect(snap.lastSeq).toBe(store.events().length);
    snap.nodes[0]!.status = "purged";
    expect(store.getNode(root.id)!.status).toBe("pending");
  });

  test("empty store snapshot", () => {
    expect(makeStore().snapshot()).toEqual({
      nodes: [],
      edges: [],
      generation: 0,
      counts: { pending: 0, pruned: 0, evaluated: 0, failed: 0, purged: 0 },
      lastSeq: 0,
    });
  });
});

describe("replay", () => {
  function busyStore() {
    const { store, root } = seeded();
    const a = store.addNode({ parentIds: [root.id], source: "a", mutation: local("a") });
    const b = store.addNode({ parentIds: [a.id], source: "b", mutation: local("b") });
    const c = store.addNode({ parentIds: [a.id], source: "c", mutation: local("c") });
    const x = store.addNode({ parentIds: [b.id, c.id], source: "x", mutation: { kind: "crossover", strategy: "x", rationale: "" } });
    store.recordVerdict(b.id, { model: "kev-9b", score: 0.8, latencyMs: 11 });
    store.recordEvaluation(root.id, metrics(41));
    store.recordEvaluation(a.id, metrics(70));
    store.recordFailure(c.id, { kind: "numerical", tier: 2, summary: "y".repeat(400) });
    store.markPruned(x.id, "low");
    store.purgeBranch(a.id, "plateau");
    const p = store.addNode({ parentIds: [a.id], source: "p", mutation: { kind: "paradigm_shift", strategy: "warp specialization", rationale: "escape basin" } });
    store.recordVerdict(p.id, { model: "kev-27b", score: 0.95, latencyMs: 19 });
    store.recordEvaluation(p.id, metrics(150));
    return store;
  }

  test("replaying events yields an identical snapshot and derived views", () => {
    const original = busyStore();
    const replayed = replayGraphStore(original.events());
    expect(replayed.snapshot()).toEqual(original.snapshot());
    expect(replayed.champion()?.id).toBe(original.champion()!.id);
    expect(replayed.bestByGeneration()).toEqual(original.bestByGeneration());
    expect(replayed.events()).toEqual(original.events());
    const last = original.snapshot().nodes.at(-1)!.id;
    expect(replayed.lineageContext(last)).toEqual(original.lineageContext(last));
  });

  test("replay is deterministic and a replayed store accepts new writes", () => {
    const events = busyStore().events();
    const r1 = replayGraphStore(events);
    const r2 = replayGraphStore(events);
    expect(r1.snapshot()).toEqual(r2.snapshot());
    const n = r1.addNode({ parentIds: [r1.champion()!.id], source: "next", mutation: local("next") });
    expect(r1.getNode(n.id)).toBeDefined();
    expect(r1.events().at(-1)!.seq).toBe(events.length + 1);
  });

  test("replay of a prefix reconstructs the earlier state", () => {
    const original = busyStore();
    const events = original.events();
    const partial = replayGraphStore(events.slice(0, 5));
    expect(partial.snapshot().nodes).toHaveLength(5);
    expect(partial.snapshot().lastSeq).toBe(5);
  });

  test("replay rejects out-of-order events", () => {
    const events = busyStore().events();
    expect(() => replayGraphStore([events[1]!, events[0]!])).toThrow();
  });

  test("replay does not notify subscribers of history", () => {
    const store = new InMemoryGraphStore();
    const seen: GraphEvent[] = [];
    store.subscribe((e) => seen.push(e));
    for (const e of busyStore().events()) store.replay(e);
    expect(seen).toHaveLength(0);
  });
});
