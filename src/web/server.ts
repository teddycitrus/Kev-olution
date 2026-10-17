/**
 * Localhost web visualizer: bun run web [--port 4173]
 *
 * Serves src/web/index.html and streams one evolution run to the browser over
 * Server-Sent Events. Separate from the terminal dashboard and CLI (src/index.ts
 * is not used or changed); the loop wiring mirrors it.
 *
 *   GET  /            the page
 *   GET  /events      SSE: live events (history replayed only while a run is in progress)
 *   POST /run?mode=demo|live|offline&pace=MS   start a run (one at a time)
 *
 * Modes: demo = Kev answers from .kev/demo-cache.json (deterministic, no network
 * on a full cache); live = Kev over the network (saved to .kev/web-live-cache.json,
 * never to the demo cache); offline = no Kev, heuristic routing and fallback order.
 */

import { join } from "node:path";
import { configFromSource, describeConfig } from "../agents/kernelTemplate";
import { Mutator } from "../agents/mutator";
import { Reflector } from "../agents/reflector";
import { loadBenchmark } from "../harness/benchmark";
import { Evaluator } from "../harness/evaluator";
import type { GraphEvent } from "../harness/graphStore";
import { InMemoryGraphStore } from "../harness/inMemoryGraphStore";
import { KevClient } from "../harness/kevClient";
import { ParadigmShiftEngine, PlateauDetector } from "../harness/plateauDetector";
import { evolve, type EvolveEvent } from "../harness/runner";

const DEMO_SNAPSHOT = ".kev/demo-cache.json";
const LIVE_SNAPSHOT = ".kev/web-live-cache.json";
const PAGE = join(import.meta.dir, "index.html");
type Mode = "demo" | "live" | "offline";

/** What the browser receives. Graph nodes are trimmed: no kernel source over the wire. */
/**
 * Work done by one run, counted from its events. Used to compare a Kev-routed
 * run against the same search without Kev (the counterfactual baseline).
 */
export interface RunWork {
  cycles: number;
  candidates: number;
  /** Sandbox evaluations that got past Tier 1 (each needs a compile + correctness run). */
  compiles: number;
  /** Evaluations that reached the full Tier 3 benchmark. */
  benchmarks: number;
  gpuSeconds: number;
  kevCalls: number;
  bestScore: number;
}

export function emptyWork(): RunWork {
  return { cycles: 0, candidates: 0, compiles: 0, benchmarks: 0, gpuSeconds: 0, kevCalls: 0, bestScore: 0 };
}

/** Folds one EvolveEvent into a RunWork tally. */
export function tallyWork(w: RunWork, e: EvolveEvent): void {
  if (e.type === "evaluated") {
    if (e.tier >= 2) w.compiles += 1;
    if (e.tier >= 3) w.benchmarks += 1;
  } else if (e.type === "finished") {
    const st = e.summary.stats;
    w.cycles = e.summary.cycles;
    w.candidates = st.candidates;
    w.gpuSeconds = st.gpuSecondsSpent;
    w.kevCalls = st.kevCalls["kev-9b"] + st.kevCalls["kev-27b"];
    w.bestScore = e.summary.bestScore;
  }
}

export type WebEvent =
  | { kind: "run"; mode: Mode; pace: number; benchmark: string; problem: unknown; hardware: string; t: number }
  | { kind: "baseline"; work: RunWork; note: string; t: number }
  | { kind: "evolve"; e: EvolveEvent; t: number }
  | { kind: "graph"; e: TrimmedGraphEvent; t: number }
  | { kind: "error"; message: string; t: number };

type TrimmedGraphEvent =
  | { type: "node_added"; id: string; parentIds: readonly string[]; generation: number; kind: string; strategy: string; config: string }
  | Exclude<GraphEvent, { type: "node_added" }>;

export function trimGraphEvent(e: GraphEvent): TrimmedGraphEvent {
  if (e.type !== "node_added") return e;
  let config = "?";
  try {
    config = describeConfig(configFromSource(e.node.source));
  } catch {
    // Unparseable candidate; keep the placeholder.
  }
  const n = e.node;
  return { type: "node_added", id: n.id, parentIds: n.parentIds, generation: n.generation, kind: n.mutation.kind, strategy: n.mutation.strategy, config };
}

class RunHub {
  private history: string[] = [];
  private readonly clients = new Set<ReadableStreamDefaultController<string>>();
  running = false;

  publish(event: WebEvent): void {
    const frame = `data: ${JSON.stringify(event)}\n\n`;
    this.history.push(frame);
    for (const c of this.clients) {
      try {
        c.enqueue(frame);
      } catch {
        this.clients.delete(c);
      }
    }
  }

  reset(): void {
    this.history = [];
  }

  subscribe(): Response {
    let ctrl: ReadableStreamDefaultController<string>;
    const stream = new ReadableStream<string>({
      start: (c) => {
        ctrl = c;
        // Catch up only on a run in progress; a finished run is not replayed, so a fresh page starts empty.
        if (this.running) for (const frame of this.history) c.enqueue(frame);
        this.clients.add(c);
      },
      cancel: () => {
        this.clients.delete(ctrl);
      },
    });
    return new Response(stream, { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" } });
  }
}

let baselineCache: Promise<RunWork> | undefined;

/** The identical search without Kev, unpaced (about 1 s). Computed once and cached. */
export function noKevBaseline(): Promise<RunWork> {
  baselineCache ??= (async () => {
    const benchmark = await loadBenchmark("triton_flash_attention");
    const store = new InMemoryGraphStore();
    const sources = new Set<string>();
    store.subscribe((e) => {
      if (e.type === "node_added") sources.add(e.node.source);
    });
    const work = emptyWork();
    await evolve(
      {
        benchmark,
        store,
        evaluator: new Evaluator(benchmark, { seed: 0 }),
        mutator: new Mutator({ benchmarkDescription: benchmark.description, isKnown: (s) => sources.has(s) }),
        reflector: new Reflector(),
        plateau: new PlateauDetector({ window: 3 }),
        shiftEngine: new ParadigmShiftEngine({ store, benchmark, kev: null }),
        kev: null,
      },
      { seed: 0, onEvent: (e) => tallyWork(work, e) },
    );
    return work;
  })();
  return baselineCache;
}

async function startRun(hub: RunHub, mode: Mode, pace: number): Promise<void> {
  hub.running = true;
  hub.reset();
  const started = performance.now();
  const t = () => performance.now() - started;
  try {
    const benchmark = await loadBenchmark("triton_flash_attention");
    hub.publish({ kind: "run", mode, pace, benchmark: benchmark.name, problem: benchmark.problem, hardware: benchmark.hardware.name, t: t() });
    const baseline = await noKevBaseline();
    hub.publish({ kind: "baseline", work: baseline, note: "Same search (seed 0, same task and evaluator) with no Kev: heuristic routing and fixed shift order.", t: t() });
    const hasKey = Boolean(process.env.KEV_API_KEY || process.env.KEV_9B_API_KEY);
    // demo: load the recorded snapshot. live: empty in-memory cache, so every decision goes to Kev.
    const kev =
      mode === "offline" || !hasKey ? null : KevClient.fromEnv(process.env, { demo: mode === "demo" ? { snapshotPath: DEMO_SNAPSHOT } : true });
    const store = new InMemoryGraphStore();
    const sources = new Set<string>();
    store.subscribe((e) => {
      if (e.type === "node_added") sources.add(e.node.source);
      hub.publish({ kind: "graph", e: trimGraphEvent(e), t: t() });
    });
    await evolve(
      {
        benchmark,
        store,
        evaluator: new Evaluator(benchmark, { seed: 0, wallMsPerGpuSecond: pace }),
        mutator: new Mutator({ benchmarkDescription: benchmark.description, isKnown: (s) => sources.has(s) }),
        reflector: new Reflector(),
        plateau: new PlateauDetector({ window: 3 }),
        shiftEngine: new ParadigmShiftEngine({ store, benchmark, kev }),
        kev,
      },
      { seed: 0, onEvent: (e) => hub.publish({ kind: "evolve", e, t: t() }) },
    );
    // Live answers go to their own file so they never change what the terminal demos replay.
    if (kev?.cache) kev.cache.saveSnapshot(mode === "live" ? LIVE_SNAPSHOT : DEMO_SNAPSHOT);
  } catch (err) {
    hub.publish({ kind: "error", message: err instanceof Error ? err.message : String(err), t: t() });
  } finally {
    hub.running = false;
  }
}

export function createServer(port: number) {
  const hub = new RunHub();
  return Bun.serve({
    port,
    hostname: "127.0.0.1",
    idleTimeout: 0,
    async fetch(req) {
      const url = new URL(req.url);
      if (req.method === "GET" && url.pathname === "/") return new Response(Bun.file(PAGE), { headers: { "Content-Type": "text/html; charset=utf-8" } });
      if (req.method === "GET" && url.pathname === "/events") return hub.subscribe();
      if (req.method === "POST" && url.pathname === "/run") {
        if (hub.running) return Response.json({ ok: false, error: "a run is already in progress" }, { status: 409 });
        const mode = (url.searchParams.get("mode") ?? "demo") as Mode;
        if (!["demo", "live", "offline"].includes(mode)) return Response.json({ ok: false, error: "mode must be demo, live, or offline" }, { status: 400 });
        const pace = Math.min(1000, Math.max(0, Number(url.searchParams.get("pace") ?? 100) || 0));
        void startRun(hub, mode, pace);
        return Response.json({ ok: true, mode, pace });
      }
      return new Response("not found", { status: 404 });
    },
  });
}

if (import.meta.main) {
  const i = process.argv.indexOf("--port");
  const port = i === -1 ? 4173 : Number(process.argv[i + 1]);
  const server = createServer(port);
  void noKevBaseline();
  console.log(`kev-evolve web visualizer: http://localhost:${server.port}`);
}
