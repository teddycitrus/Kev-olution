import { describe, expect, test } from "bun:test";
import { KEV_DEFAULT_BASE_URL, KevApiError, KevClient, MemoryQuotaStore, QuotaTracker, type KevModel } from "../src/harness/kevClient";

interface Call {
  url: string;
  method: string;
  auth: string;
  body: any;
  signal?: AbortSignal;
}

const GW = "https://gateway.test";
const URL_9B = "https://me--kev-9b-api.modal.run";
const URL_27B = "https://me--kev-27b-api.modal.run";

/** Responds per route; records every call. Never touches the network. */
function mockFetch(handler?: (call: Call) => Response | Promise<Response>) {
  const calls: Call[] = [];
  const fetchImpl = (async (input: string | URL | Request, init: RequestInit = {}) => {
    const call: Call = {
      url: String(input),
      method: init.method ?? "GET",
      auth: (init.headers as Record<string, string>).Authorization,
      body: init.body ? JSON.parse(String(init.body)) : undefined,
      signal: init.signal ?? undefined,
    };
    calls.push(call);
    if (handler) return handler(call);
    if (call.url.endsWith("/v1/models")) return new Response(JSON.stringify({ models: [{ name: "kev-latest" }] }));
    const answers: Record<string, unknown> = {};
    for (const [id, q] of Object.entries<any>(call.body.questions)) {
      answers[id] = q.type === "choice" ? { type: "choice", choice: Object.keys(q.criteria)[0], probabilities: {} } : { type: "noul", noul: 0.5 };
    }
    return new Response(JSON.stringify({ model: call.body.model, answers, usage: {}, latency_ms: 9 }));
  }) as typeof fetch;
  return { calls, fetchImpl };
}

function fromEnv(env: Record<string, string | undefined>, handler?: (call: Call) => Response | Promise<Response>, extra: Record<string, unknown> = {}) {
  const { calls, fetchImpl } = mockFetch(handler);
  const quota = new QuotaTracker(new MemoryQuotaStore());
  const kev = KevClient.fromEnv(env, { fetchImpl, quota, ...extra });
  return { kev, calls, quota };
}

const Q = { q: { type: "noul" as const, instructions: "ok?" } };
const FULL_ENV = { KEV_API_KEY: "gw-key", KEV_BASE_URL: GW, KEV_9B_BASE_URL: URL_9B, KEV_9B_API_KEY: "key-9b", KEV_27B_BASE_URL: `${URL_27B}/`, KEV_27B_API_KEY: "key-27b" };

describe("per-model endpoint routing from an explicit env", () => {
  test("kev-9b and kev-27b go to their own URLs with their own keys", async () => {
    const { kev, calls } = fromEnv(FULL_ENV);
    await kev.ask("prune", "s", Q);
    await kev.ask("paradigm_shift", "s", Q);
    expect(calls.map((c) => [c.method, c.url, c.auth, c.body.model])).toEqual([
      ["POST", `${URL_9B}/v1/systemone`, "Bearer key-9b", "kev-9b"],
      ["POST", `${URL_27B}/v1/systemone`, "Bearer key-27b", "kev-27b"],
    ]);
  });

  test("a per-model key without a URL is ignored: that model uses the gateway and KEV_API_KEY", async () => {
    const { kev, calls } = fromEnv({ KEV_API_KEY: "gw-key", KEV_BASE_URL: GW, KEV_9B_API_KEY: "key-9b", KEV_27B_BASE_URL: URL_27B, KEV_27B_API_KEY: "key-27b" });
    await kev.ask("ast_filter", "s", Q);
    await kev.ask("paradigm_shift", "s", Q);
    expect(calls[0]).toMatchObject({ url: `${GW}/v1/systemone`, auth: "Bearer gw-key" });
    expect(calls[1]).toMatchObject({ url: `${URL_27B}/v1/systemone`, auth: "Bearer key-27b" });
    expect(kev.isSelfHosted("kev-9b")).toBe(false);
    expect(kev.isSelfHosted("kev-27b")).toBe(true);
  });

  test("a per-model URL without a key is ignored too", async () => {
    const { kev, calls } = fromEnv({ KEV_API_KEY: "gw-key", KEV_9B_BASE_URL: URL_9B });
    await kev.ask("prune", "s", Q);
    expect(calls[0]).toMatchObject({ url: `${KEV_DEFAULT_BASE_URL}/v1/systemone`, auth: "Bearer gw-key" });
  });

  test("kev-27b is routed independently of kev-9b", async () => {
    const { kev, calls } = fromEnv({ KEV_API_KEY: "gw-key", KEV_27B_BASE_URL: URL_27B, KEV_27B_API_KEY: "key-27b" });
    await kev.ask("score_trajectory", "s", Q);
    await kev.ask("paradigm_shift", "s", Q);
    expect(calls[0]!.url).toBe(`${KEV_DEFAULT_BASE_URL}/v1/systemone`);
    expect(calls[1]!.url).toBe(`${URL_27B}/v1/systemone`);
  });

  test("self-hosted keys alone do not satisfy the client's gateway key requirement", () => {
    expect(() => KevClient.fromEnv({ KEV_9B_BASE_URL: URL_9B, KEV_9B_API_KEY: "key-9b" }, { quota: new QuotaTracker(new MemoryQuotaStore()) })).toThrow("KEV_API_KEY is not set");
  });

  test("quota is enforced locally for self-hosted models too", async () => {
    const { calls, fetchImpl } = mockFetch();
    const kev = KevClient.fromEnv(FULL_ENV, { fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore(), { "kev-9b": 1, "kev-27b": 1 }) });
    await kev.ask("prune", "s", Q);
    await expect(kev.ask("prune", "s2", Q)).rejects.toThrow("quota exhausted for kev-9b");
    expect(calls).toHaveLength(1);
  });
});

describe("per-model timeouts", () => {
  const hang = (call: Call) =>
    new Promise<Response>((_, reject) => call.signal!.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" }))));

  test("kev-27b defaults to 30 s while kev-9b uses timeoutMs", async () => {
    const release: Array<() => void> = [];
    const ok27 = () => new Response(JSON.stringify({ model: "kev-27b", answers: { q: { type: "noul", noul: 1 } }, latency_ms: 1 }));
    const held = () => new Promise<Response>((resolve) => release.push(() => resolve(ok27())));
    const { kev } = fromEnv(FULL_ENV, (call) => (call.url.startsWith(URL_27B) ? held() : hang(call)), { timeoutMs: 25 });
    const started = performance.now();
    await expect(kev.ask("prune", "s", Q)).rejects.toMatchObject({ status: 408, detail: "timeout after 25 ms" });
    expect(performance.now() - started).toBeLessThan(2000);
    const slow = kev.ask("paradigm_shift", "s", Q);
    const winner = await Promise.race([slow.then(() => "settled"), new Promise((r) => setTimeout(() => r("still waiting"), 150))]);
    expect(winner).toBe("still waiting");
    release.forEach((r) => r());
    expect((await slow).source).toBe("network");
  });

  test("modelTimeoutMs overrides per model", async () => {
    const { kev } = fromEnv(FULL_ENV, hang, { timeoutMs: 5000, modelTimeoutMs: { "kev-27b": 30, "kev-9b": 40 } });
    await expect(kev.ask("paradigm_shift", "s", Q)).rejects.toMatchObject({ status: 408, detail: "timeout after 30 ms" });
    await expect(kev.ask("prune", "s", Q)).rejects.toMatchObject({ status: 408, detail: "timeout after 40 ms" });
  });
});

describe("warmup()", () => {
  test("self-hosted endpoints get a free GET /v1/models with their own key", async () => {
    const { kev, calls, quota } = fromEnv(FULL_ENV);
    const out = await kev.warmup();
    expect(typeof out["kev-9b"]).toBe("number");
    expect(typeof out["kev-27b"]).toBe("number");
    expect(calls.map((c) => [c.method, c.url, c.auth, c.body]).sort()).toEqual(
      [
        ["GET", `${URL_27B}/v1/models`, "Bearer key-27b", undefined],
        ["GET", `${URL_9B}/v1/models`, "Bearer key-9b", undefined],
      ].sort(),
    );
    expect(quota.snapshot().used).toEqual({ "kev-9b": 0, "kev-27b": 0 });
  });

  test("the shared gateway gets one minimal SystemOne POST and one quota call per model", async () => {
    const { kev, calls, quota } = fromEnv({ KEV_API_KEY: "gw-key", KEV_BASE_URL: GW });
    const out = await kev.warmup(["kev-9b"]);
    expect(Object.keys(out)).toEqual(["kev-9b"]);
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ method: "POST", url: `${GW}/v1/systemone`, auth: "Bearer gw-key" });
    expect(calls[0]!.body).toEqual({ model: "kev-9b", state: "warmup", questions: { ready: { type: "noul", instructions: "Is this a warmup request?" } } });
    expect(quota.snapshot().used).toEqual({ "kev-9b": 1, "kev-27b": 0 });
  });

  test("mixed: self-hosted 9b is free, gateway 27b spends one call", async () => {
    const { kev, quota } = fromEnv({ KEV_API_KEY: "gw-key", KEV_9B_BASE_URL: URL_9B, KEV_9B_API_KEY: "key-9b" });
    await kev.warmup();
    expect(quota.snapshot().used).toEqual({ "kev-9b": 0, "kev-27b": 1 });
  });

  test("errors are returned per model, not thrown", async () => {
    const { kev } = fromEnv(FULL_ENV, (call) =>
      call.url.startsWith(URL_27B) ? new Response(JSON.stringify({ detail: "cold start failed" }), { status: 503 }) : new Response(JSON.stringify({ models: [] })),
    );
    const out = await kev.warmup();
    expect(typeof out["kev-9b"]).toBe("number");
    expect(out["kev-27b"]).toBe("Kev API error 503: cold start failed");
  });

  test("a thrown fetch and an exhausted gateway quota are reported as strings", async () => {
    const { calls, fetchImpl } = mockFetch(() => {
      throw new TypeError("network down");
    });
    const kev = KevClient.fromEnv({ KEV_API_KEY: "gw-key", KEV_27B_BASE_URL: URL_27B, KEV_27B_API_KEY: "k" }, { fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore(), { "kev-9b": 0, "kev-27b": 5 }) });
    const out = await kev.warmup();
    expect(out["kev-9b"]).toContain("quota exhausted for kev-9b");
    expect(out["kev-27b"]).toBe("network down");
    expect(calls).toHaveLength(1);
  });

  test("uses the warmup timeout for the long cold-start wait", async () => {
    const { kev } = fromEnv(FULL_ENV, (call) =>
      new Promise<Response>((_, reject) => call.signal!.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })))),
    );
    const out = await kev.warmup(["kev-27b"], 20);
    expect(out["kev-27b"]).toBe("Kev API error 408: timeout after 20 ms");
  });
});

describe("isSelfHosted", () => {
  test("reflects which models have their own endpoint", () => {
    const { kev } = fromEnv(FULL_ENV);
    expect(kev.isSelfHosted("kev-9b")).toBe(true);
    expect(kev.isSelfHosted("kev-27b")).toBe(true);
    const { kev: gw } = fromEnv({ KEV_API_KEY: "gw-key" });
    for (const m of ["kev-9b", "kev-27b"] as KevModel[]) expect(gw.isSelfHosted(m)).toBe(false);
  });
});

describe("choose()", () => {
  const question = { type: "choice" as const, instructions: "Which rewrite?", criteria: { tma: "TMA", ws: "warp specialization", persistent: "persistent" } };
  const reply = (decision: unknown) => () => new Response(JSON.stringify({ model: "kev-27b", answers: { decision }, usage: {}, latency_ms: 17 }));

  test("sends one kev-27b choice question named decision and returns the choice", async () => {
    const { kev, calls } = fromEnv(FULL_ENV, reply({ type: "choice", choice: "ws", confidence: 0.7, probabilities: { tma: 0.2, ws: 0.7, persistent: 0.1 } }));
    const r = await kev.choose("paradigm_shift", "state", question);
    expect(r.choice).toBe("ws");
    expect(r.probabilities).toEqual({ tma: 0.2, ws: 0.7, persistent: 0.1 });
    expect(r.result.response.latency_ms).toBe(17);
    expect(calls[0]!.body).toEqual({ model: "kev-27b", state: "state", questions: { decision: question } });
    expect(calls[0]!.url).toBe(`${URL_27B}/v1/systemone`);
  });

  test("falls back to the argmax of probabilities when choice is missing", async () => {
    const { kev } = fromEnv(FULL_ENV, reply({ type: "choice", probabilities: { tma: 0.1, ws: 0.3, persistent: 0.6, junk: "x" } }));
    const r = await kev.choose("paradigm_shift", "s", question);
    expect(r.choice).toBe("persistent");
    expect(r.probabilities).toEqual({ tma: 0.1, ws: 0.3, persistent: 0.6 });
  });

  test("throws when the choice is outside the offered options", async () => {
    const { kev } = fromEnv(FULL_ENV, reply({ type: "choice", choice: "rewrite_in_rust" }));
    const err = await kev.choose("paradigm_shift", "s", question).catch((e) => e);
    expect(err).toBeInstanceOf(KevApiError);
    expect(err.message).toContain("is not one of tma, ws, persistent");
  });

  test("throws when there is neither a choice nor probabilities", async () => {
    const { kev } = fromEnv(FULL_ENV, reply({ type: "choice" }));
    await expect(kev.choose("paradigm_shift", "s", question)).rejects.toThrow("is not one of");
  });

  test("an argmax outside the options is rejected", async () => {
    const { kev } = fromEnv(FULL_ENV, reply({ type: "choice", probabilities: { other: 0.9, tma: 0.1 } }));
    await expect(kev.choose("paradigm_shift", "s", question)).rejects.toThrow("is not one of");
  });
});

describe("scoreCandidates with instructions", () => {
  test("sends one noul question per candidate, with instructions when given", async () => {
    const { kev, calls } = fromEnv(FULL_ENV);
    const { scores } = await kev.scoreCandidates("state", [{ id: "n1", instructions: "Will tile 128 help?" }, "n2", { id: "n3", instructions: "Will exp2 help?" }]);
    expect(scores).toEqual({ n1: 0.5, n2: 0.5, n3: 0.5 });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.url).toBe(`${URL_9B}/v1/systemone`);
    expect(calls[0]!.body.questions).toEqual({
      n1: { type: "noul", instructions: "Will tile 128 help?" },
      n2: { type: "noul" },
      n3: { type: "noul", instructions: "Will exp2 help?" },
    });
  });
});
