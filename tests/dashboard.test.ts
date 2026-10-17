import { describe, expect, test } from "bun:test";
import { Mutator } from "../src/agents/mutator";
import { Reflector } from "../src/agents/reflector";
import { loadBenchmark } from "../src/harness/benchmark";
import { Evaluator } from "../src/harness/evaluator";
import { InMemoryGraphStore } from "../src/harness/inMemoryGraphStore";
import { ParadigmShiftEngine, PlateauDetector } from "../src/harness/plateauDetector";
import { evolve, type EvolveEvent } from "../src/harness/runner";
import { Dashboard } from "../src/ui/dashboard";
import { visibleLength } from "../src/ui/render";

const spec = await loadBenchmark("triton_flash_attention");

const ENTER_ALT = "\x1b[?1049h";
const LEAVE_ALT = "\x1b[?1049l";
const SYNC_END = "[?2026l";
const HOME = "\x1b[?2026h\x1b[H"; // synchronized-output begin + cursor home

function fakeStream(columns: number | undefined = 100, rows: number | undefined = 30) {
  const chunks: string[] = [];
  return { chunks, columns, rows, write: (c: string) => chunks.push(c), frames: () => chunks.filter((c) => c.startsWith(HOME)) };
}

function setup(opts: { fps?: number; columns?: number; rows?: number } = {}) {
  const stream = fakeStream(opts.columns, opts.rows);
  let clock = 1_000;
  const dash = new Dashboard({ stream, fps: opts.fps ?? 50, color: false, now: () => clock });
  return { stream, dash, tick: (ms: number) => (clock += ms) };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const cycleStarted = (cycle: number): EvolveEvent => ({ type: "cycle_started", cycle, parentId: "n0", parentSummary: "64x32 w4 s1" });
const champion = (cycle: number): EvolveEvent => ({ type: "champion", cycle, id: `n${cycle}`, score: 40 + cycle, summary: "x" });
const shift: EvolveEvent = {
  type: "paradigm_shift",
  cycle: 3,
  directive: { strategyId: "tma_descriptors", strategyLabel: "TMA tensor descriptors", decidedBy: "fallback", probabilities: {}, rootId: "n1", purgedIds: [], rationale: "r" },
};

describe("alternate screen lifecycle", () => {
  test("start() enters the alternate screen, hides the cursor, and draws one frame", () => {
    const { stream, dash } = setup();
    dash.start();
    expect(stream.chunks[0]).toBe(`${ENTER_ALT}\x1b[?25l`);
    expect(stream.frames()).toHaveLength(1);
    dash.stop();
  });

  test("stop() draws a final frame, leaves the alternate screen, then prints the summary", () => {
    const { stream, dash } = setup();
    dash.start();
    dash.onEvent(champion(0));
    const before = stream.chunks.length;
    dash.stop();
    const tail = stream.chunks.slice(before);
    expect(tail).toHaveLength(3);
    expect(tail[0]!.startsWith(HOME)).toBe(true);
    expect(tail[1]).toBe(`\x1b[?25h${LEAVE_ALT}`);
    expect(tail[2]).toBe("kev-evolve stopped before finishing\n");
  });

  test("start() is idempotent", () => {
    const { stream, dash } = setup();
    dash.start();
    dash.start();
    expect(stream.chunks.filter((c) => c.includes(ENTER_ALT))).toHaveLength(1);
    dash.stop();
  });

  test("stop() is idempotent and onEvent after stop() never writes", async () => {
    const { stream, dash } = setup();
    dash.start();
    dash.stop();
    const after = stream.chunks.length;
    dash.stop();
    dash.onEvent(champion(1));
    dash.onEvent(shift);
    dash.onEvent(cycleStarted(2));
    dash.onEvent({ type: "finished", summary: { cycles: 1, stopReason: "aborted", bestScore: 41, baselineScore: 41, speedup: 1, targetReached: false, backend: "simulated", stats: { candidates: 0, astRejected: 0, kevPruned: 0, evaluated: 0, gpuSecondsSpent: 0, gpuSecondsSavedEarlyExit: 0, gpuSecondsSavedByPruningEstimate: 0, kevCalls: { "kev-9b": 0, "kev-27b": 0 }, kevLatencyMs: [], paradigmShifts: 0 } } });
    await sleep(60);
    expect(stream.chunks.length).toBe(after);
    expect(dash.state.summary?.stopReason).toBe("aborted");
  });

  test("a redraw pending at stop() is cancelled", async () => {
    const { stream, dash } = setup({ fps: 20 });
    dash.start();
    dash.onEvent(cycleStarted(1));
    dash.stop();
    const after = stream.chunks.length;
    await sleep(120);
    expect(stream.chunks.length).toBe(after);
  });

  test("the SIGINT listener is added by start() and removed by stop()", () => {
    const base = process.listenerCount("SIGINT");
    const { dash } = setup();
    dash.start();
    expect(process.listenerCount("SIGINT")).toBe(base + 1);
    dash.stop();
    expect(process.listenerCount("SIGINT")).toBe(base);
  });
});

describe("redraw policy", () => {
  test("bursts of ordinary events are throttled to far fewer frames", async () => {
    const { stream, dash, tick } = setup({ fps: 50 });
    dash.start();
    const start = stream.frames().length;
    for (let i = 1; i <= 100; i++) {
      tick(1);
      dash.onEvent(cycleStarted(i));
    }
    expect(stream.frames().length).toBe(start);
    await sleep(80);
    const drawn = stream.frames().length - start;
    expect(drawn).toBeGreaterThanOrEqual(1);
    expect(drawn).toBeLessThanOrEqual(2);
    expect(dash.state.cycle).toBe(100);
    expect(stream.frames().at(-1)).toContain("cycle 100");
    dash.stop();
  });

  test("champion, paradigm_shift, and finished redraw immediately", () => {
    const { stream, dash } = setup({ fps: 1 });
    dash.start();
    const start = stream.frames().length;
    dash.onEvent(champion(0));
    expect(stream.frames().length).toBe(start + 1);
    dash.onEvent(shift);
    expect(stream.frames().length).toBe(start + 2);
    expect(stream.frames().at(-1)).toContain("paradigm shift (1)");
    dash.stop();
  });

  test("graph store events update the DAG and schedule a redraw; stop() unsubscribes", async () => {
    const store = new InMemoryGraphStore({ now: () => 0 });
    const { stream, dash } = setup({ fps: 50 });
    dash.start(store);
    const start = stream.frames().length;
    const root = store.addNode({ parentIds: [], source: spec.baselineSource, mutation: { kind: "seed", strategy: "baseline", rationale: "" } });
    expect(dash.state.nodes.get(root.id)?.config).toBe("64x32 w4 s1");
    await sleep(60);
    expect(stream.frames().length).toBeGreaterThan(start);
    dash.stop();
    store.addNode({ parentIds: [root.id], source: "x", mutation: { kind: "local", strategy: "s", rationale: "" } });
    expect(dash.state.nodes.size).toBe(1);
  });
});

describe("frames", () => {
  test("frame() uses the stream size and every written frame line is exactly that wide", () => {
    const { stream, dash } = setup({ columns: 100, rows: 30 });
    dash.start();
    dash.onEvent(champion(0));
    expect(dash.frame()).toHaveLength(30);
    for (const f of stream.frames()) {
      const lines = f.slice(HOME.length).replace(SYNC_END, "").split("\n");
      expect(lines).toHaveLength(30);
      for (const l of lines) expect(visibleLength(l)).toBe(100);
    }
    dash.stop();
  });

  test("frame() defaults to 120x40 when the stream has no size", () => {
    const stream = { chunks: [] as string[], write(c: string) { this.chunks.push(c); } };
    const dash = new Dashboard({ stream, color: false, now: () => 0 });
    const frame = dash.frame();
    expect(frame).toHaveLength(40);
    expect(visibleLength(frame[0]!)).toBe(120);
  });

  test("color defaults on and can be disabled", () => {
    const colored = new Dashboard({ stream: fakeStream(), now: () => 0 }).frame().join("");
    const plain = new Dashboard({ stream: fakeStream(), color: false, now: () => 0 }).frame().join("");
    expect(colored).toContain("\x1b[");
    expect(plain).not.toContain("\x1b[");
  });
});

describe("live offline run through the dashboard", () => {
  test("drives a full evolve() and prints the final summary after leaving the alternate screen", async () => {
    const store = new InMemoryGraphStore({ now: () => 0 });
    const known = new Set<string>();
    store.subscribe((e) => {
      if (e.type === "node_added") known.add(e.node.source);
    });
    const stream = fakeStream(140, 44);
    const dash = new Dashboard({ stream, fps: 1000, color: false });
    dash.start(store);
    let summary;
    try {
      summary = await evolve(
        {
          benchmark: spec, store, evaluator: new Evaluator(spec), reflector: new Reflector(),
          mutator: new Mutator({ isKnown: (s) => known.has(s) }), plateau: new PlateauDetector(),
          shiftEngine: new ParadigmShiftEngine({ store, benchmark: spec, kev: null }), kev: null,
        },
        { seed: 0, maxCycles: 40, onEvent: dash.onEvent },
      );
    } finally {
      dash.stop();
    }
    const leave = stream.chunks.findIndex((c) => c.includes(LEAVE_ALT));
    expect(leave).toBeGreaterThan(0);
    const lastFrame = stream.chunks.slice(0, leave).filter((c) => c.startsWith(HOME)).at(-1)!;
    expect(lastFrame).toContain(`done: ${summary.stopReason}`);
    expect(lastFrame).toContain("<- champion");
    const final = stream.chunks.slice(leave + 1).join("");
    expect(final).toContain(`${summary.bestScore.toFixed(1)} TFLOPS`);
    expect(final).toContain("target reached [simulated]");
    expect(final).toContain(`paradigm shifts: c`);
  });
});
