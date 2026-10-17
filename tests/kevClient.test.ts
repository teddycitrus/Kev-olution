import { afterAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  FileQuotaStore,
  KEV_DAILY_LIMITS,
  KEV_DEFAULT_BASE_URL,
  KevApiError,
  KevClient,
  KevQuotaExceededError,
  KevRoutingError,
  MemoryQuotaStore,
  QuotaTracker,
  parseServerUsage,
  parseSystemOneResponse,
  type KevCallEvent,
  type KevModel,
  type KevPurpose,
} from "../src/harness/kevClient";

interface Call {
  url: string;
  init: RequestInit;
  body: any;
}

const okBody = (model: KevModel, answers: Record<string, unknown> = { q: { type: "noul", noul: 0.5 } }) => ({
  model,
  answers,
  usage: { input_tokens: 10, output_tokens: 2 },
  latency_ms: 12,
});

function mockFetch(responder: (call: Call) => Response | Promise<Response>) {
  const calls: Call[] = [];
  const fetchImpl = (async (input: string | URL | Request, init: RequestInit = {}) => {
    const call = { url: String(input), init, body: init.body ? JSON.parse(String(init.body)) : undefined };
    calls.push(call);
    return responder(call);
  }) as typeof fetch;
  return { calls, fetchImpl };
}

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const echoModel = (c: Call) => json(200, okBody(c.body.model));

function client(opts: { responder?: (call: Call) => Response | Promise<Response>; limits?: Record<KevModel, number>; now?: () => Date; timeoutMs?: number; onCall?: (e: KevCallEvent) => void } = {}) {
  const { calls, fetchImpl } = mockFetch(opts.responder ?? echoModel);
  const quota = new QuotaTracker(new MemoryQuotaStore(), opts.limits ?? KEV_DAILY_LIMITS, opts.now);
  const kev = new KevClient({ apiKey: "test-key", fetchImpl, quota, timeoutMs: opts.timeoutMs, onCall: opts.onCall });
  return { kev, calls, quota };
}

const Q = { q: { type: "noul" as const } };

describe("construction", () => {
  test("requires an API key", () => {
    expect(() => new KevClient({ apiKey: "" })).toThrow(KevRoutingError);
    expect(() => KevClient.fromEnv({})).toThrow("KEV_API_KEY");
  });

  test("fromEnv reads KEV_API_KEY and KEV_BASE_URL and trims trailing slashes", async () => {
    const { calls, fetchImpl } = mockFetch(echoModel);
    const kev = KevClient.fromEnv(
      { KEV_API_KEY: "env-key", KEV_BASE_URL: "https://example.test///" },
      { fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore()) },
    );
    await kev.ask("prune", {}, Q);
    expect(calls[0]!.url).toBe("https://example.test/v1/systemone");
    expect((calls[0]!.init.headers as Record<string, string>).Authorization).toBe("Bearer env-key");
  });

  test("defaults to the PRD endpoint", async () => {
    const { kev, calls } = client();
    await kev.ask("prune", {}, Q);
    expect(calls[0]!.url).toBe(`${KEV_DEFAULT_BASE_URL}/v1/systemone`);
  });
});

describe("routing", () => {
  test.each<[KevPurpose, KevModel]>([
    ["prune", "kev-9b"],
    ["score_trajectory", "kev-9b"],
    ["ast_filter", "kev-9b"],
    ["paradigm_shift", "kev-27b"],
  ])("%s routes to %s", async (purpose, model) => {
    const { kev, calls, quota } = client();
    const result = await kev.ask(purpose, { diff: "+x" }, Q);
    expect(calls[0]!.body).toEqual({ model, state: { diff: "+x" }, questions: Q });
    expect(calls[0]!.init.method).toBe("POST");
    expect((calls[0]!.init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    expect(result.response.model).toBe(model);
    expect(result.roundTripMs).toBeGreaterThanOrEqual(0);
    expect(quota.snapshot().used[model]).toBe(1);
    expect(result.quota.used[model]).toBe(1);
  });

  test("onCall reports telemetry for success and failure", async () => {
    const events: KevCallEvent[] = [];
    let fail = false;
    const { kev } = client({ onCall: (e) => events.push(e), responder: (c) => (fail ? json(500, { detail: "boom" }) : json(200, okBody(c.body.model))) });
    await kev.ask("paradigm_shift", {}, Q);
    fail = true;
    await expect(kev.ask("prune", {}, Q)).rejects.toThrow(KevApiError);
    expect(events).toMatchObject([
      { purpose: "paradigm_shift", model: "kev-27b", ok: true, serverLatencyMs: 12 },
      { purpose: "prune", model: "kev-9b", ok: false, serverLatencyMs: 0 },
    ]);
  });
});

describe("question validation", () => {
  test.each([
    ["empty questions", {}],
    ["choice without criteria", { c: { type: "choice", criteria: {} } }],
    ["score without criteria", { s: { type: "score", criteria: [] } }],
  ])("%s is rejected before any quota or network use", async (_label, questions) => {
    const { kev, calls, quota } = client();
    await expect(kev.ask("prune", {}, questions as never)).rejects.toThrow(KevRoutingError);
    expect(calls).toHaveLength(0);
    expect(quota.snapshot().used["kev-9b"]).toBe(0);
  });
});

describe("quota", () => {
  test("exhaustion throws without calling the API", async () => {
    const { kev, calls } = client({ limits: { "kev-9b": 2, "kev-27b": 1 } });
    await kev.ask("prune", {}, Q);
    await kev.ask("ast_filter", {}, Q);
    await expect(kev.ask("score_trajectory", {}, Q)).rejects.toThrow(KevQuotaExceededError);
    await kev.ask("paradigm_shift", {}, Q);
    const err = await kev.ask("paradigm_shift", {}, Q).catch((e) => e);
    expect(err).toBeInstanceOf(KevQuotaExceededError);
    expect(err).toMatchObject({ model: "kev-27b", limit: 1 });
    expect(calls).toHaveLength(3);
  });

  test("422 releases the reservation", async () => {
    const { kev, quota } = client({ responder: () => json(422, { detail: [{ msg: "bad question" }] }) });
    const err = await kev.ask("prune", {}, Q).catch((e) => e);
    expect(err).toBeInstanceOf(KevApiError);
    expect(err.status).toBe(422);
    expect(err.detail).toEqual([{ msg: "bad question" }]);
    expect(quota.snapshot().used["kev-9b"]).toBe(0);
  });

  test.each([408, 429, 500, 503])("%i keeps the reservation", async (status) => {
    const { kev, quota } = client({ responder: () => json(status, { detail: "nope" }) });
    await expect(kev.ask("prune", {}, Q)).rejects.toMatchObject({ status });
    expect(quota.snapshot().used["kev-9b"]).toBe(1);
  });

  test("client-side timeout surfaces as 408 and keeps the reservation", async () => {
    const { kev, quota } = client({
      timeoutMs: 20,
      responder: (c) =>
        new Promise<Response>((_, reject) => {
          c.init.signal!.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
        }),
    });
    const err = await kev.ask("prune", {}, Q).catch((e) => e);
    expect(err).toBeInstanceOf(KevApiError);
    expect(err.status).toBe(408);
    expect(quota.snapshot().used["kev-9b"]).toBe(1);
  });

  test("UTC day rollover resets counters", () => {
    let now = new Date("2026-10-17T23:59:59Z");
    const store = new MemoryQuotaStore();
    const quota = new QuotaTracker(store, { "kev-9b": 1, "kev-27b": 1 }, () => now);
    quota.reserve("kev-9b");
    expect(() => quota.reserve("kev-9b")).toThrow(KevQuotaExceededError);
    now = new Date("2026-10-18T00:00:01Z");
    expect(quota.remaining("kev-9b")).toBe(1);
    expect(quota.snapshot()).toEqual({ date: "2026-10-18", used: { "kev-9b": 0, "kev-27b": 0 }, remaining: { "kev-9b": 1, "kev-27b": 1 } });
    expect(store.load()?.date).toBe("2026-10-18");
    quota.reserve("kev-9b");
  });

  test("tracker resumes today's usage from its store and ignores stale days", () => {
    const now = () => new Date("2026-10-17T12:00:00Z");
    const store = new MemoryQuotaStore();
    store.save({ date: "2026-10-17", used: { "kev-9b": 7, "kev-27b": 3 } });
    expect(new QuotaTracker(store, KEV_DAILY_LIMITS, now).snapshot().used).toEqual({ "kev-9b": 7, "kev-27b": 3 });
    store.save({ date: "2026-10-16", used: { "kev-9b": 999, "kev-27b": 199 } });
    expect(new QuotaTracker(store, KEV_DAILY_LIMITS, now).snapshot().used).toEqual({ "kev-9b": 0, "kev-27b": 0 });
  });

  test("release never goes below zero", () => {
    const quota = new QuotaTracker(new MemoryQuotaStore());
    quota.release("kev-27b");
    expect(quota.snapshot().used["kev-27b"]).toBe(0);
  });

  test("reconcile adopts higher server counts only for the same day", () => {
    const now = () => new Date("2026-10-17T08:00:00Z");
    const quota = new QuotaTracker(new MemoryQuotaStore(), KEV_DAILY_LIMITS, now);
    quota.reserve("kev-9b");
    quota.reserve("kev-9b");
    quota.reserve("kev-27b");
    quota.reconcile({
      date: "2026-10-17",
      usage: { "kev-9b": { used: 50, limit: 1000, remaining: 950 }, "kev-27b": { used: 0, limit: 200, remaining: 200 } },
    });
    expect(quota.snapshot().used).toEqual({ "kev-9b": 50, "kev-27b": 1 });
    quota.reconcile({
      date: "2026-10-16",
      usage: { "kev-9b": { used: 900, limit: 1000, remaining: 100 }, "kev-27b": { used: 150, limit: 200, remaining: 50 } },
    });
    expect(quota.snapshot().used).toEqual({ "kev-9b": 50, "kev-27b": 1 });
  });

  test("syncUsage GETs /v1/me and reconciles", async () => {
    const today = new Date().toISOString().slice(0, 10);
    const { kev, calls, quota } = client({
      responder: () => json(200, { date: today, usage: { "kev-9b": { used: 12, limit: 1000, remaining: 988 }, "kev-27b": { used: 2, limit: 200, remaining: 198 } } }),
    });
    const usage = await kev.syncUsage();
    expect(calls[0]!.url).toBe(`${KEV_DEFAULT_BASE_URL}/v1/me`);
    expect(calls[0]!.init.method).toBe("GET");
    expect(calls[0]!.init.body).toBeUndefined();
    expect(usage.usage["kev-9b"].used).toBe(12);
    expect(quota.snapshot().used).toEqual({ "kev-9b": 12, "kev-27b": 2 });
  });
});

describe("FileQuotaStore", () => {
  const dir = mkdtempSync(join(tmpdir(), "kev-quota-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test("round trips through disk and creates parent directories", () => {
    const path = join(dir, "nested", "usage.json");
    const store = new FileQuotaStore(path);
    expect(store.load()).toBeUndefined();
    store.save({ date: "2026-10-17", used: { "kev-9b": 4, "kev-27b": 1 } });
    expect(new FileQuotaStore(path).load()).toEqual({ date: "2026-10-17", used: { "kev-9b": 4, "kev-27b": 1 } });
  });

  test("persists across tracker instances", () => {
    const path = join(dir, "tracker.json");
    const now = () => new Date("2026-10-17T10:00:00Z");
    new QuotaTracker(new FileQuotaStore(path), KEV_DAILY_LIMITS, now).reserve("kev-27b");
    expect(new QuotaTracker(new FileQuotaStore(path), KEV_DAILY_LIMITS, now).snapshot().used["kev-27b"]).toBe(1);
  });

  test("corrupt or incomplete files load as undefined, bad counts as zero", () => {
    const bad = join(dir, "bad.json");
    writeFileSync(bad, "{not json");
    expect(new FileQuotaStore(bad).load()).toBeUndefined();
    writeFileSync(bad, JSON.stringify({ used: {} }));
    expect(new FileQuotaStore(bad).load()).toBeUndefined();
    writeFileSync(bad, JSON.stringify({ date: "2026-10-17", used: { "kev-9b": "x" } }));
    expect(new FileQuotaStore(bad).load()).toEqual({ date: "2026-10-17", used: { "kev-9b": 0, "kev-27b": 0 } });
  });
});

describe("response handling", () => {
  test("scoreCandidates batches candidates into one kev-9b call", async () => {
    const { kev, calls } = client({
      responder: (c) => json(200, okBody(c.body.model, { a: { type: "noul", noul: 0.9 }, b: { type: "noul", noul: 0.1 } })),
    });
    const { scores, result } = await kev.scoreCandidates({ parent: "n0" }, ["a", "b"]);
    expect(scores).toEqual({ a: 0.9, b: 0.1 });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.body.model).toBe("kev-9b");
    expect(calls[0]!.body.questions).toEqual({ a: { type: "noul" }, b: { type: "noul" } });
    expect(result.quota.used["kev-9b"]).toBe(1);
  });

  test("scoreCandidates rejects empty input and missing answers", async () => {
    const { kev } = client({ responder: (c) => json(200, okBody(c.body.model, { a: { type: "noul", noul: 0.9 } })) });
    await expect(kev.scoreCandidates({}, [])).rejects.toThrow(KevRoutingError);
    await expect(kev.scoreCandidates({}, ["a", "b"])).rejects.toThrow("missing noul answer for b");
  });

  test("malformed SystemOne responses raise KevApiError", async () => {
    const { kev } = client({ responder: () => json(200, { model: "kev-9b" }) });
    await expect(kev.ask("prune", {}, Q)).rejects.toThrow("missing answers");
    expect(() => parseSystemOneResponse({ answers: { q: { noul: 1 } } })).toThrow("malformed answer for q");
    expect(() => parseSystemOneResponse(null)).toThrow(KevApiError);
    expect(() => parseSystemOneResponse({ answers: [] })).toThrow(KevApiError);
  });

  test("missing usage and latency default to zero", () => {
    expect(parseSystemOneResponse({ model: "kev-9b", answers: {} })).toEqual({ model: "kev-9b", answers: {}, usage: { input_tokens: 0, output_tokens: 0 }, latency_ms: 0 });
  });

  test("non-JSON error bodies are kept as text", async () => {
    const { kev } = client({ responder: () => new Response("upstream exploded", { status: 502 }) });
    const err = await kev.ask("prune", {}, Q).catch((e) => e);
    expect(err).toMatchObject({ status: 502, detail: "upstream exploded" });
    expect(err.message).toContain("upstream exploded");
  });

  test("parseServerUsage validates shape", () => {
    expect(() => parseServerUsage({ usage: {} })).toThrow("malformed /v1/me");
    expect(() => parseServerUsage({ date: "2026-10-17" })).toThrow(KevApiError);
  });
});
