import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { cacheKey, KevResponseCache } from "../src/harness/kevCache";
import { KevClient, MemoryQuotaStore, QuotaTracker, type KevCallEvent, type KevModel, type SystemOneResponse } from "../src/harness/kevClient";

const Q = { q: { type: "noul" as const, instructions: "Is this mutation promising?" } };

const response = (model: KevModel, noul = 0.7): SystemOneResponse => ({
  model,
  answers: { q: { type: "noul", noul } },
  usage: { input_tokens: 10, output_tokens: 1 },
  latency_ms: 11,
});

interface Gate {
  release: () => void;
}

/** Fetch mock that can hold responses open to test in-flight sharing. */
function mockFetch(opts: { hold?: boolean; status?: number | (() => number) } = {}) {
  const calls: Array<{ body: any }> = [];
  const gates: Gate[] = [];
  const fetchImpl = (async (_url: string | URL | Request, init: RequestInit = {}) => {
    const body = JSON.parse(String(init.body));
    calls.push({ body });
    if (opts.hold) await new Promise<void>((resolve) => gates.push({ release: resolve }));
    const status = typeof opts.status === "function" ? opts.status() : opts.status ?? 200;
    const payload = status === 200 ? response(body.model, calls.length / 100) : { detail: "boom" };
    return new Response(JSON.stringify(payload), { status });
  }) as typeof fetch;
  const releaseAll = () => gates.splice(0).forEach((g) => g.release());
  return { calls, fetchImpl, releaseAll, pending: () => gates.length };
}

function client(opts: { demo?: boolean | ConstructorParameters<typeof KevResponseCache>[0]; hold?: boolean; status?: number | (() => number); limits?: Record<KevModel, number>; onCall?: (e: KevCallEvent) => void } = {}) {
  const mock = mockFetch({ hold: opts.hold, status: opts.status });
  const quota = new QuotaTracker(new MemoryQuotaStore(), opts.limits ?? { "kev-9b": 1000, "kev-27b": 200 });
  const kev = new KevClient({ apiKey: "k", fetchImpl: mock.fetchImpl, quota, demo: opts.demo ?? true, onCall: opts.onCall });
  return { kev, quota, ...mock };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

describe("cacheKey", () => {
  test("ignores object key order and undefined values", () => {
    const a = cacheKey("kev-9b", { b: 1, a: { y: 2, x: [1, { d: 1, c: 2 }] } }, { q: { type: "noul", instructions: "x" } });
    const b = cacheKey("kev-9b", { a: { x: [1, { c: 2, d: 1 }], y: 2 }, b: 1, z: undefined }, { q: { instructions: "x", type: "noul" } });
    expect(a).toBe(b);
    expect(a).toMatch(/^[0-9a-f]{40}$/);
  });

  test("differs on model, state, questions, and array order", () => {
    const base = cacheKey("kev-9b", { s: 1 }, Q);
    expect(cacheKey("kev-27b", { s: 1 }, Q)).not.toBe(base);
    expect(cacheKey("kev-9b", { s: 2 }, Q)).not.toBe(base);
    expect(cacheKey("kev-9b", { s: 1 }, { q: { type: "noul", instructions: "other" } })).not.toBe(base);
    expect(cacheKey("kev-9b", { s: [1, 2] }, Q)).not.toBe(cacheKey("kev-9b", { s: [2, 1] }, Q));
    expect(cacheKey("kev-9b", "text state", Q)).not.toBe(cacheKey("kev-9b", { "text state": true }, Q));
  });
});

describe("demo mode off", () => {
  test("no cache, every call hits the network, speculate is a no-op", async () => {
    const { kev, calls, quota } = client({ demo: false });
    expect(kev.cache).toBeUndefined();
    const a = await kev.ask("prune", { s: 1 }, Q);
    const b = await kev.ask("prune", { s: 1 }, Q);
    expect([a.source, b.source]).toEqual(["network", "network"]);
    expect(calls).toHaveLength(2);
    expect(quota.snapshot().used["kev-9b"]).toBe(2);
    expect(kev.speculate("prune", { s: 2 }, Q)).toBe(false);
    expect(calls).toHaveLength(2);
  });
});

describe("ask() resolution order", () => {
  test("network on first ask, then cache hits with no quota and zero server latency", async () => {
    const events: KevCallEvent[] = [];
    const { kev, calls, quota } = client({ onCall: (e) => events.push(e) });
    const first = await kev.ask("prune", { s: 1 }, Q);
    const second = await kev.ask("prune", { s: 1 }, Q);
    expect(first.source).toBe("network");
    expect(second.source).toBe("cache");
    expect(second.response).toEqual(first.response);
    expect(calls).toHaveLength(1);
    expect(quota.snapshot().used["kev-9b"]).toBe(1);
    expect(second.quota.used["kev-9b"]).toBe(1);
    expect(events.map((e) => [e.source, e.serverLatencyMs, e.speculative])).toEqual([
      ["network", 11, false],
      ["cache", 0, false],
    ]);
    expect(kev.cache!.stats()).toMatchObject({ hits: 1, misses: 1, inflightHits: 0, entries: 1 });
  });

  test("concurrent identical asks share one in-flight request and one quota unit", async () => {
    const { kev, calls, quota, releaseAll, pending } = client({ hold: true });
    const a = kev.ask("prune", { s: 1 }, Q);
    await tick();
    const b = kev.ask("prune", { s: 1 }, Q);
    const c = kev.ask("prune", { s: 1 }, Q);
    await tick();
    expect(pending()).toBe(1);
    releaseAll();
    const results = await Promise.all([a, b, c]);
    expect(results.map((r) => r.source)).toEqual(["network", "inflight", "inflight"]);
    expect(new Set(results.map((r) => JSON.stringify(r.response))).size).toBe(1);
    expect(calls).toHaveLength(1);
    expect(quota.snapshot().used["kev-9b"]).toBe(1);
    expect(kev.cache!.stats()).toMatchObject({ inflightHits: 2, misses: 1, entries: 1 });
    expect((await kev.ask("prune", { s: 1 }, Q)).source).toBe("cache");
  });

  test("different models do not share entries", async () => {
    const { kev, calls } = client();
    await kev.ask("prune", { s: 1 }, Q);
    const shift = await kev.ask("paradigm_shift", { s: 1 }, Q);
    expect(shift.source).toBe("network");
    expect(shift.response.model).toBe("kev-27b");
    expect(calls).toHaveLength(2);
  });

  test("failed requests are not cached and the next ask retries the network", async () => {
    let status = 500;
    const { kev, calls, quota } = client({ status: () => status });
    await expect(kev.ask("prune", { s: 1 }, Q)).rejects.toMatchObject({ status: 500 });
    expect(kev.cache!.stats().entries).toBe(0);
    status = 200;
    const retry = await kev.ask("prune", { s: 1 }, Q);
    expect(retry.source).toBe("network");
    expect(calls).toHaveLength(2);
    expect(quota.snapshot().used["kev-9b"]).toBe(2);
  });

  test("a failed in-flight request rejects every waiter and leaves nothing cached", async () => {
    const { kev, releaseAll } = client({ hold: true, status: 503 });
    const a = kev.ask("prune", { s: 1 }, Q);
    await tick();
    const b = kev.ask("prune", { s: 1 }, Q);
    await tick();
    releaseAll();
    const settled = await Promise.allSettled([a, b]);
    expect(settled.map((s) => s.status)).toEqual(["rejected", "rejected"]);
    for (const s of settled) expect((s as PromiseRejectedResult).reason).toMatchObject({ status: 503 });
    await tick();
    expect(kev.cache!.has(cacheKey("kev-9b", { s: 1 }, Q))).toBe(false);
  });

  test("onCall exceptions are swallowed", async () => {
    const { kev } = client({
      onCall: () => {
        throw new Error("telemetry exploded");
      },
    });
    expect((await kev.ask("prune", { s: 1 }, Q)).source).toBe("network");
    expect((await kev.ask("prune", { s: 1 }, Q)).source).toBe("cache");
  });

  test("100 cache hits each answer in under 5 ms", async () => {
    const { kev } = client();
    await kev.ask("prune", { s: "warm" }, Q);
    const times: number[] = [];
    for (let i = 0; i < 100; i++) {
      const r = await kev.ask("prune", { s: "warm" }, Q);
      expect(r.source).toBe("cache");
      times.push(r.roundTripMs);
    }
    times.sort((a, b) => a - b);
    expect(times[98]!).toBeLessThan(5);
    expect(times[49]!).toBeLessThan(1);
  });
});

describe("speculate()", () => {
  test("prefetches in the background; a later ask is served from cache", async () => {
    const events: KevCallEvent[] = [];
    const { kev, calls, quota } = client({ onCall: (e) => events.push(e) });
    expect(kev.speculate("score_trajectory", { s: 1 }, Q)).toBe(true);
    await tick();
    await tick();
    expect(calls).toHaveLength(1);
    expect(quota.snapshot().used["kev-9b"]).toBe(1);
    const r = await kev.ask("score_trajectory", { s: 1 }, Q);
    expect(r.source).toBe("cache");
    expect(calls).toHaveLength(1);
    expect(events[0]).toMatchObject({ speculative: true, source: "network", ok: true });
    expect(kev.cache!.stats().speculative).toBe(1);
  });

  test("an ask while speculation is in flight joins it", async () => {
    const { kev, calls, releaseAll } = client({ hold: true });
    kev.speculate("prune", { s: 1 }, Q);
    await tick();
    const ask = kev.ask("prune", { s: 1 }, Q);
    releaseAll();
    expect((await ask).source).toBe("inflight");
    expect(calls).toHaveLength(1);
  });

  test("returns false when already cached or already in flight", async () => {
    const { kev, calls, releaseAll } = client({ hold: true });
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(true);
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(false);
    await tick();
    releaseAll();
    await tick();
    await tick();
    expect(kev.cache!.has(cacheKey("kev-9b", { s: 1 }, Q))).toBe(true);
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(false);
    expect(calls).toHaveLength(1);
  });

  test("respects maxSpeculative and frees slots when requests settle", async () => {
    const { kev, calls, releaseAll } = client({ hold: true, demo: { maxSpeculative: 2 } });
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(true);
    expect(kev.speculate("prune", { s: 2 }, Q)).toBe(true);
    expect(kev.speculate("prune", { s: 3 }, Q)).toBe(false);
    await tick();
    expect(calls).toHaveLength(2);
    releaseAll();
    await tick();
    await tick();
    expect(kev.speculate("prune", { s: 3 }, Q)).toBe(true);
  });

  test("normal asks do not consume speculation slots", async () => {
    const { kev, releaseAll } = client({ hold: true, demo: { maxSpeculative: 1 } });
    const pendingAsk = kev.ask("prune", { s: "ask" }, Q);
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(true);
    await tick();
    releaseAll();
    await pendingAsk;
  });

  test("returns false when quota is exhausted", async () => {
    const { kev, calls } = client({ limits: { "kev-9b": 1, "kev-27b": 1 } });
    await kev.ask("prune", { s: 1 }, Q);
    expect(kev.speculate("prune", { s: 2 }, Q)).toBe(false);
    expect(calls).toHaveLength(1);
  });

  test("failed speculation is not cached and frees its slot", async () => {
    let status = 500;
    const { kev, calls } = client({ status: () => status, demo: { maxSpeculative: 1 } });
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(true);
    await tick();
    await tick();
    await tick();
    expect(kev.cache!.stats().entries).toBe(0);
    status = 200;
    expect(kev.speculate("prune", { s: 1 }, Q)).toBe(true);
    await tick();
    await tick();
    expect(calls).toHaveLength(2);
    expect((await kev.ask("prune", { s: 1 }, Q)).source).toBe("cache");
  });

  test("kev-27b can never be speculated (compile-time guard)", () => {
    const { kev } = client();
    const typeCheckOnly = () => {
      // @ts-expect-error paradigm_shift is excluded from speculate()
      kev.speculate("paradigm_shift", { s: 1 }, Q);
    };
    expect(typeof typeCheckOnly).toBe("function");
  });
});

describe("KevResponseCache LRU", () => {
  test("evicts the least recently used entry; get() refreshes recency", () => {
    const cache = new KevResponseCache({ maxEntries: 2 });
    cache.set("a", response("kev-9b", 0.1));
    cache.set("b", response("kev-9b", 0.2));
    expect(cache.get("a")).toBeDefined();
    cache.set("c", response("kev-9b", 0.3));
    expect(cache.has("a")).toBe(true);
    expect(cache.has("b")).toBe(false);
    expect(cache.has("c")).toBe(true);
    expect(cache.stats().entries).toBe(2);
  });

  test("re-setting a key refreshes it without growing the cache", () => {
    const cache = new KevResponseCache({ maxEntries: 2 });
    cache.set("a", response("kev-9b", 0.1));
    cache.set("b", response("kev-9b", 0.2));
    cache.set("a", response("kev-9b", 0.9));
    cache.set("c", response("kev-9b", 0.3));
    expect(cache.has("b")).toBe(false);
    expect((cache.get("a")!.answers.q as { noul: number }).noul).toBe(0.9);
  });

  test("client LRU through ask(): an evicted entry goes back to the network", async () => {
    const { kev, calls } = client({ demo: { maxEntries: 1 } });
    await kev.ask("prune", { s: 1 }, Q);
    await kev.ask("prune", { s: 2 }, Q);
    expect((await kev.ask("prune", { s: 1 }, Q)).source).toBe("network");
    expect(calls).toHaveLength(3);
  });
});

describe("rehearsal snapshots", () => {
  const dir = mkdtempSync(join(tmpdir(), "kev-cache-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test("a rehearsal run replays with zero network calls and zero quota", async () => {
    const path = join(dir, "nested", "rehearsal.json");
    const rehearsal = client({ demo: { snapshotPath: path } });
    for (let i = 0; i < 5; i++) await rehearsal.kev.ask("prune", { gen: i }, Q);
    await rehearsal.kev.ask("paradigm_shift", { gen: 5 }, Q);
    rehearsal.kev.cache!.saveSnapshot();

    const live = client({ demo: { snapshotPath: path } });
    expect(live.kev.cache!.stats().entries).toBe(6);
    for (let i = 0; i < 5; i++) {
      const r = await live.kev.ask("prune", { gen: i }, Q);
      expect(r.source).toBe("cache");
      expect(r.response).toEqual((await rehearsal.kev.ask("prune", { gen: i }, Q)).response);
    }
    expect((await live.kev.ask("paradigm_shift", { gen: 5 }, Q)).source).toBe("cache");
    expect(live.calls).toHaveLength(0);
    expect(live.quota.snapshot().used).toEqual({ "kev-9b": 0, "kev-27b": 0 });
  });

  test("saveSnapshot accepts an explicit path and requires one", () => {
    const cache = new KevResponseCache();
    cache.set("k", response("kev-9b"));
    expect(() => cache.saveSnapshot()).toThrow("requires a path");
    const path = join(dir, "explicit.json");
    cache.saveSnapshot(path);
    expect(new KevResponseCache({ snapshotPath: path }).get("k")).toEqual(response("kev-9b"));
  });

  test.each([
    ["corrupt JSON", "{not json"],
    ["wrong version", JSON.stringify({ version: 2, entries: [["k", response("kev-9b")]] })],
    ["entries not an array", JSON.stringify({ version: 1, entries: {} })],
  ])("ignores a snapshot with %s", (_label, content) => {
    const path = join(dir, `bad-${_label.replace(/\W/g, "")}.json`);
    writeFileSync(path, content);
    expect(new KevResponseCache({ snapshotPath: path }).stats().entries).toBe(0);
  });

  test("skips malformed entries but keeps valid ones", () => {
    const path = join(dir, "mixed.json");
    writeFileSync(path, JSON.stringify({ version: 1, entries: [["ok", response("kev-9b")], [42, response("kev-9b")], ["no-response", null], "junk"] }));
    const cache = new KevResponseCache({ snapshotPath: path });
    expect(cache.stats().entries).toBe(1);
    expect(cache.has("ok")).toBe(true);
  });

  test("a missing snapshot file is fine", () => {
    expect(new KevResponseCache({ snapshotPath: join(dir, "does-not-exist.json") }).stats().entries).toBe(0);
  });
});
