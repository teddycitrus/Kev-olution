/**
 * Kev SystemOne routing client.
 *
 * Wire format (confirmed against the gateway on 2026-10-17):
 *   POST /v1/systemone
 *   { "model": "kev-9b", "state": {...}, "questions": { "<id>": Question } }
 *   -> { "model", "answers": { "<id>": Answer }, "usage": {...}, "latency_ms" }
 *   GET  /v1/me -> server-side daily usage per model.
 *
 * Question variants accepted by the gateway: "noul" (criteria optional, object),
 * "choice" (criteria: object), "score" (criteria: list). Only the "noul" answer
 * shape ({ type, noul: number }) has been observed; the others are parsed
 * loosely until confirmed.
 *
 * Routing rules (PRD section 3):
 *   kev-9b  -> high-frequency pruning, trajectory scoring, AST filtering.
 *   kev-27b -> only on a Paradigm Shift event from the Plateau Detector.
 *
 * Quota: 1000 calls/day for kev-9b, 200 for kev-27b. A call is reserved
 * locally before it is sent and released if the gateway rejects the request
 * with 4xx validation (those are not billed). Usage persists across runs via a
 * QuotaStore so restarting the harness cannot overspend (see quota.ts).
 *
 * Demo mode (`demo` option, the --demo flag): exact-match response cache with
 * speculative prefetch and rehearsal snapshots (see kevCache.ts). Cache hits
 * spend no quota and every result reports its `source`.
 */

import { FileQuotaStore, QuotaTracker, type QuotaSnapshot } from "./quota";
import { cacheKey, KevResponseCache, type DemoCacheOptions, type KevResultSource } from "./kevCache";

export { FileQuotaStore, KEV_DAILY_LIMITS, KevQuotaExceededError, MemoryQuotaStore, QuotaTracker, type QuotaSnapshot, type QuotaState, type QuotaStore } from "./quota";
export { KevResponseCache, type CacheStats, type DemoCacheOptions, type KevResultSource } from "./kevCache";

export type KevModel = "kev-9b" | "kev-27b";

export const KEV_DEFAULT_BASE_URL = "https://yohann-60613--kev-demo.modal.run";

// ---------------------------------------------------------------------------
// API schemas
// ---------------------------------------------------------------------------

// Kev is a classifier: it answers typed questions about a state and never
// generates text. `instructions` is the question itself (per the TypeSafe
// protocol documented in .devin/skills/kev-deploy/SKILL.md).

export interface NoulQuestion {
  type: "noul";
  instructions?: string;
  criteria?: Record<string, string>;
}

export interface ChoiceQuestion {
  type: "choice";
  instructions?: string;
  /** Option label -> description. */
  criteria: Record<string, string>;
}

export interface ScoreQuestion {
  type: "score";
  instructions?: string;
  /** Ordered levels, lowest first. */
  criteria: string[];
}

export type KevQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion;

/** Context the model reasons over. Plain text is the documented form. */
export type KevState = string | Record<string, unknown>;

export interface SystemOneRequest {
  model: KevModel;
  state: KevState;
  questions: Record<string, KevQuestion>;
}

export interface NoulAnswer {
  type: "noul";
  /** Probability of "yes" in [0, 1] (observed live). */
  noul: number;
  probabilities?: Record<string, number>;
}

/** Documented but not yet observed live; parse defensively. */
export interface ChoiceAnswer {
  type: "choice";
  choice: string;
  probabilities?: Record<string, number>;
}

/** Answer shapes not yet observed are kept as raw objects. */
export interface UnknownAnswer {
  type: string;
  [field: string]: unknown;
}

export type KevAnswer = NoulAnswer | ChoiceAnswer | UnknownAnswer;

/** One candidate to score: an id, or an id plus the question to ask about it. */
export type ScoreCandidate = string | { id: string; instructions: string };

export interface SystemOneResponse {
  model: KevModel;
  answers: Record<string, KevAnswer>;
  usage: { input_tokens: number; output_tokens: number };
  /** Server-reported inference latency. */
  latency_ms: number;
}

export interface ServerUsage {
  date: string;
  usage: Record<KevModel, { used: number; limit: number; remaining: number }>;
}

/** What callers get back: the response plus client-side telemetry. */
export interface KevResult {
  response: SystemOneResponse;
  /** Client-observed time to answer, including network when source is "network". */
  roundTripMs: number;
  /** Where the answer came from. Only "network" spends quota. */
  source: KevResultSource;
  quota: QuotaSnapshot;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

export class KevApiError extends Error {
  constructor(readonly status: number, readonly detail: unknown) {
    super(`Kev API error ${status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
    this.name = "KevApiError";
  }
}

export class KevRoutingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "KevRoutingError";
  }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/** Why a call is being made. Drives model selection and enforces PRD routing. */
export type KevPurpose = "prune" | "score_trajectory" | "ast_filter" | "paradigm_shift";

const PURPOSE_MODEL: Readonly<Record<KevPurpose, KevModel>> = {
  prune: "kev-9b",
  score_trajectory: "kev-9b",
  ast_filter: "kev-9b",
  paradigm_shift: "kev-27b",
};

export interface KevClientOptions {
  apiKey: string;
  /**
   * Optional per-model endpoints (e.g. a self-hosted kev-deploy URL). Models
   * without an entry use baseUrl + apiKey.
   */
  endpoints?: Partial<Record<KevModel, { baseUrl: string; apiKey: string }>>;
  baseUrl?: string;
  /** Per-request timeout. Default 5000 ms. */
  timeoutMs?: number;
  /** Per-model overrides. Default kev-27b: 30000 ms (larger model, slower cold path). */
  modelTimeoutMs?: Partial<Record<KevModel, number>>;
  quota?: QuotaTracker;
  fetchImpl?: typeof fetch;
  /** Called after every completed request, for dashboard telemetry. */
  onCall?: (event: KevCallEvent) => void;
  /**
   * Enables the speculative response cache (the --demo flag). `true` uses
   * defaults; pass options to load a rehearsal snapshot.
   */
  demo?: boolean | DemoCacheOptions;
}

export interface KevCallEvent {
  purpose: KevPurpose;
  model: KevModel;
  roundTripMs: number;
  /** Server-reported inference time; 0 for cache hits. */
  serverLatencyMs: number;
  source: KevResultSource;
  speculative: boolean;
  ok: boolean;
}

export class KevClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly modelTimeoutMs: Record<KevModel, number>;
  private readonly fetchImpl: typeof fetch;
  readonly quota: QuotaTracker;
  /** Present only in demo mode. */
  readonly cache?: KevResponseCache;

  constructor(private readonly options: KevClientOptions) {
    if (!options.apiKey) throw new KevRoutingError("KEV_API_KEY is not set");
    this.baseUrl = (options.baseUrl ?? KEV_DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.timeoutMs = options.timeoutMs ?? 5000;
    this.modelTimeoutMs = { "kev-9b": this.timeoutMs, "kev-27b": 30_000, ...options.modelTimeoutMs };
    this.fetchImpl = options.fetchImpl ?? fetch;
    this.quota = options.quota ?? new QuotaTracker(new FileQuotaStore());
    if (options.demo) this.cache = new KevResponseCache(options.demo === true ? {} : options.demo);
  }

  /** Builds a client from KEV_API_KEY and optional KEV_BASE_URL. */
  static fromEnv(env: Record<string, string | undefined> = process.env, overrides: Partial<KevClientOptions> = {}): KevClient {
    return new KevClient({
      apiKey: env.KEV_API_KEY ?? "",
      endpoints: endpointsFromEnv(env),
      baseUrl: env.KEV_BASE_URL,
      ...overrides,
    });
  }

  /**
   * Sends one SystemOne request. The model is chosen from `purpose`, so
   * kev-27b cannot be spent on routine pruning by accident.
   */
  async ask(purpose: KevPurpose, state: KevState, questions: Record<string, KevQuestion>): Promise<KevResult> {
    validateQuestions(questions);
    const model = PURPOSE_MODEL[purpose];
    const started = performance.now();
    if (!this.cache) return this.finish(purpose, model, await this.send(purpose, model, state, questions, false), started, "network");

    const key = cacheKey(model, state, questions);
    const hit = this.cache.get(key);
    if (hit) return this.finish(purpose, model, hit, started, "cache");
    const pending = this.cache.getInflight(key);
    if (pending) return this.finish(purpose, model, await pending, started, "inflight");
    this.cache.recordMiss();
    const request = this.send(purpose, model, state, questions, false);
    this.cache.track(key, request);
    return this.finish(purpose, model, await request, started, "network");
  }

  /**
   * Demo mode only: starts a request in the background so a later identical
   * ask() is served from cache. Returns false when skipped (no cache, already
   * cached or in flight, speculation limit reached, or quota exhausted).
   * kev-27b is never speculated.
   */
  speculate(purpose: Exclude<KevPurpose, "paradigm_shift">, state: KevState, questions: Record<string, KevQuestion>): boolean {
    if (!this.cache) return false;
    validateQuestions(questions);
    const model = PURPOSE_MODEL[purpose];
    const key = cacheKey(model, state, questions);
    if (this.cache.has(key) || !this.cache.canSpeculate() || this.quota.remaining(model) === 0) return false;
    this.cache.track(key, this.send(purpose, model, state, questions, true), true);
    return true;
  }

  /** Network path: reserves quota, sends, validates, and reports failures. */
  private async send(purpose: KevPurpose, model: KevModel, state: KevState, questions: Record<string, KevQuestion>, speculative: boolean): Promise<SystemOneResponse> {
    this.quota.reserve(model);
    const body: SystemOneRequest = { model, state, questions };
    const started = performance.now();
    try {
      const response = parseSystemOneResponse(await this.request("POST", "/v1/systemone", body, this.options.endpoints?.[model], this.modelTimeoutMs[model]));
      if (speculative) {
        this.emit({ purpose, model, roundTripMs: performance.now() - started, serverLatencyMs: response.latency_ms, source: "network", speculative, ok: true });
      }
      return response;
    } catch (err) {
      // 4xx validation errors are not billed by the gateway.
      // Timeouts (408) and rate limits (429) may still have been billed.
      if (err instanceof KevApiError && err.status >= 400 && err.status < 500 && err.status !== 408 && err.status !== 429) {
        this.quota.release(model);
      }
      this.emit({ purpose, model, roundTripMs: performance.now() - started, serverLatencyMs: 0, source: "network", speculative, ok: false });
      throw err;
    }
  }

  private finish(purpose: KevPurpose, model: KevModel, response: SystemOneResponse, started: number, source: KevResultSource): KevResult {
    const roundTripMs = performance.now() - started;
    this.emit({ purpose, model, roundTripMs, serverLatencyMs: source === "cache" ? 0 : response.latency_ms, source, speculative: false, ok: true });
    return { response, roundTripMs, source, quota: this.quota.snapshot() };
  }

  private emit(event: KevCallEvent): void {
    try {
      this.options.onCall?.(event);
    } catch {
      // Telemetry must never break routing.
    }
  }

  /**
   * Convenience wrapper: one "noul" score per candidate in a single kev-9b
   * call. Returns scores in [0, 1] keyed by candidate id.
   */
  async scoreCandidates(state: KevState, candidates: readonly ScoreCandidate[], purpose: Exclude<KevPurpose, "paradigm_shift"> = "prune"): Promise<{ scores: Record<string, number>; result: KevResult }> {
    if (candidates.length === 0) throw new KevRoutingError("scoreCandidates requires at least one candidate");
    const questions: Record<string, KevQuestion> = {};
    const candidateIds: string[] = [];
    for (const c of candidates) {
      const id = typeof c === "string" ? c : c.id;
      candidateIds.push(id);
      questions[id] = typeof c === "string" ? { type: "noul" } : { type: "noul", instructions: c.instructions };
    }
    const result = await this.ask(purpose, state, questions);
    const scores: Record<string, number> = {};
    for (const id of candidateIds) {
      const answer = result.response.answers[id];
      if (!answer || answer.type !== "noul" || typeof answer.noul !== "number") {
        throw new KevApiError(200, `missing noul answer for ${id}`);
      }
      scores[id] = answer.noul;
    }
    return { scores, result };
  }

  /**
   * Asks one choice question. Returns the chosen option and, when the server
   * provides them, per-option probabilities. Throws if the choice is not one
   * of the offered options.
   */
  async choose(purpose: KevPurpose, state: KevState, question: ChoiceQuestion): Promise<{ choice: string; probabilities: Record<string, number>; result: KevResult }> {
    const result = await this.ask(purpose, state, { decision: question });
    const answer = result.response.answers.decision;
    const options = Object.keys(question.criteria);
    const raw = answer && typeof answer === "object" ? (answer as Record<string, unknown>) : {};
    const probabilities: Record<string, number> = {};
    if (raw.probabilities && typeof raw.probabilities === "object") {
      for (const [k, v] of Object.entries(raw.probabilities as Record<string, unknown>)) if (typeof v === "number") probabilities[k] = v;
    }
    let choice = typeof raw.choice === "string" ? raw.choice : undefined;
    if (choice === undefined && Object.keys(probabilities).length) {
      choice = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0]![0];
    }
    if (choice === undefined || !options.includes(choice)) {
      throw new KevApiError(200, `choice answer ${JSON.stringify(answer)} is not one of ${options.join(", ")}`);
    }
    return { choice, probabilities, result };
  }

  /**
   * Wakes cold Modal containers before a live run. Self-hosted endpoints get
   * `GET /v1/models` (free; returns once the model is loaded, cold starts take
   * minutes). The shared gateway has no free warm path, so it gets one minimal
   * SystemOne call (one call of quota). Never throws; returns round trip ms or
   * the error message per model.
   */
  async warmup(models: readonly KevModel[] = ["kev-9b", "kev-27b"], timeoutMs = 900_000): Promise<Record<string, number | string>> {
    const out: Record<string, number | string> = {};
    await Promise.all(
      models.map(async (model) => {
        const endpoint = this.options.endpoints?.[model];
        const started = performance.now();
        try {
          if (endpoint) {
            await this.request("GET", "/v1/models", undefined, endpoint, timeoutMs);
          } else {
            this.quota.reserve(model);
            const body: SystemOneRequest = { model, state: "warmup", questions: { ready: { type: "noul", instructions: "Is this a warmup request?" } } };
            parseSystemOneResponse(await this.request("POST", "/v1/systemone", body, undefined, timeoutMs));
          }
          out[model] = performance.now() - started;
        } catch (err) {
          out[model] = err instanceof Error ? err.message : String(err);
        }
      }),
    );
    return out;
  }

  /** True when `model` is routed to its own (self-hosted) endpoint. */
  isSelfHosted(model: KevModel): boolean {
    return Boolean(this.options.endpoints?.[model]);
  }

  /** Fetches shared-gateway usage (GET /v1/me) and reconciles the local counter. Self-hosted servers have no /v1/me. */
  async syncUsage(): Promise<ServerUsage> {
    const res = await this.request("GET", "/v1/me");
    const usage = parseServerUsage(res);
    this.quota.reconcile(usage);
    return usage;
  }

  private async request(method: "GET" | "POST", path: string, body?: unknown, endpoint?: { baseUrl: string; apiKey: string }, timeoutMs = this.timeoutMs): Promise<unknown> {
    const baseUrl = endpoint ? endpoint.baseUrl.replace(/\/+$/, "") : this.baseUrl;
    const apiKey = endpoint?.apiKey ?? this.options.apiKey;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await this.fetchImpl(`${baseUrl}${path}`, {
        method,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      const text = await res.text();
      let json: unknown;
      try {
        json = text ? JSON.parse(text) : undefined;
      } catch {
        json = text;
      }
      if (!res.ok) {
        const detail = json && typeof json === "object" && "detail" in json ? (json as { detail: unknown }).detail : json;
        throw new KevApiError(res.status, detail);
      }
      return json;
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        throw new KevApiError(408, `timeout after ${timeoutMs} ms`);
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * KEV_9B_BASE_URL + KEV_9B_API_KEY (and the 27B pair) route that model to its
 * own endpoint. A per-model key without a per-model URL is ignored: those keys
 * are not valid on the shared gateway (verified 2026-10-17, HTTP 401).
 */
function endpointsFromEnv(env: Record<string, string | undefined>): KevClientOptions["endpoints"] {
  const out: NonNullable<KevClientOptions["endpoints"]> = {};
  const pairs: Array<[KevModel, string | undefined, string | undefined]> = [
    ["kev-9b", env.KEV_9B_BASE_URL, env.KEV_9B_API_KEY],
    ["kev-27b", env.KEV_27B_BASE_URL, env.KEV_27B_API_KEY],
  ];
  for (const [model, baseUrl, apiKey] of pairs) if (baseUrl && apiKey) out[model] = { baseUrl, apiKey };
  return Object.keys(out).length ? out : undefined;
}

// ---------------------------------------------------------------------------
// Boundary validation
// ---------------------------------------------------------------------------

function validateQuestions(questions: Record<string, KevQuestion>): void {
  const ids = Object.keys(questions);
  if (ids.length === 0) throw new KevRoutingError("questions must be a non-empty object");
  for (const id of ids) {
    const q = questions[id];
    if (q.type === "choice" && Object.keys(q.criteria).length === 0) {
      throw new KevRoutingError(`choice question ${id} needs at least one criterion`);
    }
    if (q.type === "score" && q.criteria.length === 0) {
      throw new KevRoutingError(`score question ${id} needs at least one criterion`);
    }
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

export function parseSystemOneResponse(raw: unknown): SystemOneResponse {
  if (!isRecord(raw) || !isRecord(raw.answers)) {
    throw new KevApiError(200, "malformed SystemOne response: missing answers");
  }
  const answers: Record<string, KevAnswer> = {};
  for (const [id, a] of Object.entries(raw.answers)) {
    if (!isRecord(a) || typeof a.type !== "string") {
      throw new KevApiError(200, `malformed answer for ${id}`);
    }
    answers[id] = a as KevAnswer;
  }
  const usage = isRecord(raw.usage) ? raw.usage : {};
  return {
    model: raw.model as KevModel,
    answers,
    usage: { input_tokens: Number(usage.input_tokens) || 0, output_tokens: Number(usage.output_tokens) || 0 },
    latency_ms: Number(raw.latency_ms) || 0,
  };
}

export function parseServerUsage(raw: unknown): ServerUsage {
  if (!isRecord(raw) || typeof raw.date !== "string" || !isRecord(raw.usage)) {
    throw new KevApiError(200, "malformed /v1/me response");
  }
  return { date: raw.date, usage: raw.usage as ServerUsage["usage"] };
}
