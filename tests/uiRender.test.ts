import { describe, expect, test } from "bun:test";
import type { EvolveStats } from "../src/harness/runner";
import {
  GPU_DOLLARS_PER_HOUR, fitWidth, paint, renderChart, renderCost, renderDag, renderDashboard, renderFinalSummary, renderHeader, renderKev, renderTelemetry, sparkline, visibleLength,
} from "../src/ui/render";
import { initialState, reduceEvolve, reduceGraph, type DagNode, type DashboardState } from "../src/ui/state";
import { recordRun } from "./fixtures/recordedRun";

const run = await recordRun();

function populated(): DashboardState {
  const s = initialState(0);
  let t = 0;
  for (const item of run.timeline) {
    if (item.kind === "graph") reduceGraph(s, item.event);
    else reduceEvolve(s, item.event, (t += 50));
  }
  return s;
}

const STATS: EvolveStats = {
  candidates: 12, astRejected: 1, kevPruned: 5, evaluated: 6, gpuSecondsSpent: 100, gpuSecondsSavedEarlyExit: 40,
  gpuSecondsSavedByPruningEstimate: 107.5, kevCalls: { "kev-9b": 2, "kev-27b": 1 }, kevLatencyMs: [], paradigmShifts: 1,
};

const ESC = "\x1b";

describe("fitWidth and visibleLength", () => {
  test("pads short text to the exact width", () => {
    expect(fitWidth("abc", 6)).toBe("abc   ");
    expect(fitWidth("abcdef", 6)).toBe("abcdef");
  });

  test("truncates long text and marks it with ~", () => {
    expect(fitWidth("abcdefgh", 5)).toBe("abcd~");
    expect(fitWidth("ab", 1)).toBe("~");
  });

  test("measures visible width with ANSI codes present", () => {
    const red = paint("hello", "red", true);
    expect(red).toBe(`${ESC}[31mhello${ESC}[0m`);
    expect(visibleLength(red)).toBe(5);
    const padded = fitWidth(red, 8);
    expect(visibleLength(padded)).toBe(8);
    expect(padded.startsWith(`${ESC}[31m`)).toBe(true);
  });

  test("closes an open style before the truncation marker", () => {
    const out = fitWidth(`${ESC}[31mhello world${ESC}[0m`, 6);
    expect(out).toBe(`${ESC}[31mhello${ESC}[0m~`);
    expect(visibleLength(out)).toBe(6);
  });

  test("returns an empty string for width 0 or less", () => {
    expect(fitWidth("abc", 0)).toBe("");
    expect(fitWidth("abc", -3)).toBe("");
  });

  test("paint is a no-op when color is off or the text is empty", () => {
    expect(paint("x", "green", false)).toBe("x");
    expect(paint("", "green", true)).toBe("");
  });
});

describe("renderDashboard sizing", () => {
  const sizes: Array<[number, number]> = [[140, 44], [120, 30], [100, 40], [60, 24], [40, 10], [59, 23]];
  for (const [label, state] of [["empty", () => initialState(0)], ["populated", populated]] as const) {
    for (const color of [false, true]) {
      test.each(sizes)(`${label} state, color ${color ? "on" : "off"}: %ix%i gives an exact frame`, (width, height) => {
        const frame = renderDashboard(state(), { width, height, color });
        const w = Math.max(60, width);
        const h = Math.max(24, height);
        expect(frame).toHaveLength(h);
        for (const line of frame) {
          expect(visibleLength(line)).toBe(w);
          expect(line.includes("\n")).toBe(false);
        }
        if (!color) expect(frame.join("").includes(ESC)).toBe(false);
      });
    }
  }

  test("wide frames use two columns; narrow frames stack panels", () => {
    const wide = renderDashboard(populated(), { width: 140, height: 44, color: false }).join("\n");
    const narrow = renderDashboard(populated(), { width: 100, height: 60, color: false });
    const sideBySide = wide.split("\n").find((l) => l.includes("Search DAG") && l.includes("Throughput"));
    expect(sideBySide).toBeDefined();
    const titleRow = (t: string) => narrow.findIndex((l) => l.includes(t));
    expect(titleRow("Throughput")).toBeLessThan(titleRow("Kev decision latency"));
    expect(titleRow("Kev decision latency")).toBeLessThan(titleRow("Compute cost"));
    expect(titleRow("Compute cost")).toBeLessThan(titleRow("Search DAG"));
  });

  test("before any event, panels show their waiting or empty text", () => {
    const frame = renderDashboard(initialState(0), { width: 140, height: 44, color: false }).join("\n");
    expect(frame).toContain("waiting for the baseline...");
    expect(frame).toContain("no routing decisions yet");
    expect(frame).toContain("no evaluations yet");
    expect(frame).toContain("(no nodes yet)");
    expect(frame).toContain("cycle 0");
  });

  test("a finished run shows the done state, champion, and target reached", () => {
    const frame = renderDashboard(populated(), { width: 140, height: 44, color: false }).join("\n");
    expect(frame).toContain(`done: ${run.summary.stopReason}`);
    expect(frame).toContain("TARGET REACHED");
    expect(frame).toContain("SIMULATED H100");
    expect(frame).toContain(`${run.summary.bestScore.toFixed(1)} TFLOPS`);
  });

  test.each([[140, 44], [120, 40], [100, 50]])("the full champion marker stays visible at %ix%i (demo-sized terminals)", (width, height) => {
    const frame = renderDashboard(populated(), { width, height, color: false });
    const line = frame.find((l) => l.includes(`${run.summary.championId!} `) && l.includes("<- champion"));
    expect(line).toBeDefined();
    expect(line).toContain(`${run.summary.bestScore.toFixed(1)} <- champion`);
  });
});

describe("renderHeader", () => {
  test("shows elapsed time from the injected clock, speedup, and labels", () => {
    const s = initialState(0);
    reduceEvolve(s, { type: "started", benchmark: "b", backend: "gpu", routing: "kev", baselineScore: 40, targetScore: 140 }, 1000);
    reduceEvolve(s, { type: "champion", cycle: 0, id: "n0", score: 80, summary: "x" }, 3500);
    const [top, best] = renderHeader(s, { width: 120, height: 40, color: false });
    expect(top).toBe("KEV-EVOLVE: Frontier Research & Optimization Harness | GPU | Kev routing");
    expect(best).toBe("b | cycle 0 | 2.5s | best 80.0 TFLOPS (2.00x baseline 40.0, target 140.0)");
  });
});

describe("renderChart", () => {
  function chartState(cycles: number, shiftCycles: number[]): DashboardState {
    const s = initialState(0);
    s.targetScore = 140;
    s.series = Array.from({ length: cycles + 1 }, (_, c) => ({ cycle: c, score: 41 + c * (130 / cycles) }));
    s.shifts = shiftCycles.map((cycle) => ({ cycle, label: `shift@${cycle}`, decidedBy: "kev-27b" as const }));
    return s;
  }
  const markerRow = (lines: string[], rows: number) => lines[rows + 1]!;
  const carets = (line: string) => [...line].filter((ch) => ch === "^").length;

  test("a ^ appears in the marker row for each shift cycle", () => {
    const lines = renderChart(chartState(10, [3, 7]), 80, 10, false);
    expect(lines).toHaveLength(10 + 3);
    expect(carets(markerRow(lines, 10))).toBe(2);
    expect(lines.at(-1)).toContain("paradigm shift (2)  last: shift@7 [kev-27b] @ cycle 7");
  });

  test("bucketing when cycles exceed columns still marks every shift", () => {
    const lines = renderChart(chartState(300, [40, 150, 290]), 40, 8, false);
    expect(carets(markerRow(lines, 8))).toBe(3);
    for (const l of lines.slice(0, 8)) expect(visibleLength(l)).toBeLessThanOrEqual(40);
  });

  test("labels the y max, target, and zero rows; columns reaching the target are drawn", () => {
    const lines = renderChart(chartState(10, []), 80, 10, false);
    expect(lines.some((l) => l.trimStart().startsWith("140 |"))).toBe(true);
    expect(lines[9]!.trimStart().startsWith("0 |")).toBe(true);
    expect(lines[0]!).toMatch(/^\s*\d+ \|/);
    expect(lines.slice(0, 10).join("")).toContain("#");
    expect(lines.at(-1)).toContain("paradigm shift (0)");
  });

  test("waits for the baseline when there is no data", () => {
    expect(renderChart(initialState(0), 80, 10, false)).toEqual(["       | waiting for the baseline..."]);
  });
});

describe("renderDag", () => {
  function deepLineage(depth: number, candidates: number): DashboardState {
    const s = initialState(0);
    const add = (n: DagNode) => {
      s.nodes.set(n.id, n);
      s.order.push(n.id);
    };
    for (let i = 0; i < depth; i++) add({ id: `n${i}`, parentId: i ? `n${i - 1}` : undefined, generation: i, kind: i ? "local" : "seed", strategy: "s", config: `cfg${i}`, status: "evaluated", score: 40 + i });
    add({ id: "dead", parentId: "n0", generation: 1, kind: "local", strategy: "s", config: "x", status: "failed" });
    for (let c = 0; c < candidates; c++) {
      add({ id: `c${c}`, parentId: `n${depth - 1}`, generation: depth, kind: "local", strategy: `cand${c}`, config: "x", status: "pruned", kevScore: 0.1 });
      s.lastCandidates.push(`c${c}`);
    }
    s.champion = { id: `n${depth - 1}`, score: 40 + depth - 1, summary: "x" };
    s.cycle = 9;
    return s;
  }

  test("the champion line is marked and ancestors list their other children", () => {
    const lines = renderDag(deepLineage(4, 2), 30, false);
    expect(lines[0]).toStartWith("n0 40.0 cfg0");
    expect(lines[0]).toContain("[1 failed]");
    expect(lines.find((l) => l.includes("<- champion"))).toContain("n3 43.0 <- champion cfg3");
    expect(lines).toContain("cycle 9 candidates");
    expect(lines.some((l) => l.includes("c1 cand1 kev 0.10"))).toBe(true);
  });

  test("the lineage is kept over candidates when maxLines is small", () => {
    const lines = renderDag(deepLineage(10, 5), 5, false);
    expect(lines.length).toBeLessThanOrEqual(5);
    expect(lines[0]).toStartWith("n0 ");
    expect(lines.some((l) => l.includes("more ancestors"))).toBe(true);
    expect(lines.at(-1)).toContain("<- champion");
    expect(lines.some((l) => l.includes("candidates"))).toBe(false);
  });

  test("shift nodes are tagged on the populated run", () => {
    const lines = renderDag(populated(), 40, false);
    expect(lines.some((l) => l.includes(" SHIFT"))).toBe(true);
    expect(lines.filter((l) => l.includes("<- champion"))).toHaveLength(1);
  });

  test("no champion yet", () => {
    expect(renderDag(initialState(0), 10, false)).toEqual(["(no nodes yet)"]);
  });
});

describe("renderCost", () => {
  test("always labels pruning savings as an estimate; simulated label only on the simulated backend", () => {
    const s = initialState(0);
    s.stats = STATS;
    const sim = renderCost(s, false).join("\n");
    expect(sim).toContain("pruning 107.5 (est.)");
    expect(sim).toContain("(simulated tier costs)");
    expect(sim).toContain("share avoided 60%");
    expect(sim).toContain("candidates 12  AST-rejected 1  Kev-pruned 5  evaluated 6");
    s.backend = "gpu";
    const gpu = renderCost(s, false).join("\n");
    expect(gpu).toContain("(est.)");
    expect(gpu).not.toContain("simulated");
  });

  test("before any evaluation", () => {
    expect(renderCost(initialState(0), false)).toEqual(["no evaluations yet"]);
  });
});

describe("renderKev", () => {
  test("lists only sources with samples, with p50 and p95", () => {
    const s = initialState(0);
    s.kevLatency.network = [10, 20, 30, 40];
    s.kevLatency.cache = [0.1];
    s.stats = STATS;
    const lines = renderKev(s, false);
    expect(lines[0]).toBe("source     calls   p50 ms   p95 ms");
    expect(lines.some((l) => l.startsWith("network        4     20.0     40.0"))).toBe(true);
    expect(lines.some((l) => l.startsWith("cache          1      0.1      0.1"))).toBe(true);
    expect(lines.some((l) => l.startsWith("inflight"))).toBe(false);
    expect(lines.some((l) => l.startsWith("heuristic"))).toBe(false);
    expect(lines.at(-1)).toBe("billed calls: kev-9b 2, kev-27b 1");
  });

  test("shows errors and the empty state", () => {
    const s = initialState(0);
    s.kevErrors = 3;
    const lines = renderKev(s, false);
    expect(lines).toContain("no routing decisions yet");
    expect(lines.at(-1)).toBe("billed calls: kev-9b 0, kev-27b 0  errors 3");
  });
});

describe("renderFinalSummary", () => {
  test("before finished", () => {
    expect(renderFinalSummary(initialState(0))).toEqual(["kev-evolve stopped before finishing"]);
  });

  test("after finished", () => {
    const lines = renderFinalSummary(populated());
    expect(lines).toHaveLength(5);
    expect(lines[0]).toBe(`kev-evolve triton_flash_attention: ${run.summary.bestScore.toFixed(1)} TFLOPS, ${run.summary.speedup.toFixed(2)}x baseline, target reached [simulated]`);
    expect(lines[1]).toStartWith(`stop: ${run.summary.stopReason} after ${run.summary.cycles} cycles; champion ${run.summary.championId}`);
    expect(lines[2]).toStartWith("paradigm shifts: c");
    expect(lines[2]!.split("; ")).toHaveLength(run.summary.stats.paradigmShifts);
    expect(lines[4]).toContain("saved by pruning (est.)");
  });
});

describe("renderTelemetry", () => {
  const bar = (s: DashboardState) => renderTelemetry(s, false)[0]!;
  const status = (s: DashboardState) => renderTelemetry(s, false)[1]!;

  test("shows - for every metric before any decision", () => {
    const s = initialState(0);
    expect(renderTelemetry(s, false)).toEqual([
      "KEV decisions/s - | routing latency - (model -) | cost savings - ($0.00 est.) | active model heuristic (offline)",
      "status: improving (local search)",
    ]);
  });

  test.each<[string, Partial<Record<"network" | "cache" | "inflight" | "heuristic", number[]>>, string]>([
    ["network only", { network: [100] }, " (network)"],
    ["cache only", { cache: [1] }, " (cache)"],
    ["inflight counts as cached", { inflight: [1] }, " (cache)"],
    ["network + cache", { network: [100], cache: [1] }, " (mixed)"],
    ["heuristic only", { heuristic: [0] }, " (heuristic)"],
  ])("decisions/s label: %s", (_label, latency, suffix) => {
    const s = initialState(0);
    Object.assign(s.kevLatency, latency);
    s.decisions = 6;
    s.decisionMs = 200;
    expect(bar(s)).toStartWith(`KEV decisions/s 30.0${suffix} | `);
  });

  test("decisions/s of 10000 or more renders as Nk; zero routing time shows -", () => {
    const s = initialState(0);
    s.kevLatency.cache = [0.1];
    s.decisions = 120;
    s.decisionMs = 2;
    expect(bar(s)).toStartWith("KEV decisions/s 60k (cache)");
    s.decisionMs = 0;
    expect(bar(s)).toStartWith("KEV decisions/s - (cache)");
  });

  test("model and round-trip p50 come from modelLatency and non-heuristic latencies", () => {
    const s = initialState(0);
    s.modelLatency = [100, 120, 140];
    s.kevLatency = { network: [200, 300], cache: [1], inflight: [], heuristic: [9999] };
    expect(bar(s)).toContain("| routing latency 200.0 ms (model 120.0 ms) |");
  });

  test("dollar estimate is (earlyExit + pruning) / 3600 * GPU_DOLLARS_PER_HOUR, always labeled est.", () => {
    const s = initialState(0);
    s.stats = STATS;
    const expected = ((STATS.gpuSecondsSavedEarlyExit + STATS.gpuSecondsSavedByPruningEstimate) / 3600) * GPU_DOLLARS_PER_HOUR;
    expect(GPU_DOLLARS_PER_HOUR).toBe(3.95);
    expect(bar(s)).toContain(`cost savings 60% ($${expected.toFixed(2)} est.)`);
    expect(bar(s)).toContain("$0.16");
  });

  test("status: PARADIGM SHIFT banner during the shift cycle and the next one", () => {
    const s = initialState(0);
    s.shifts = [{ cycle: 5, label: "TMA tensor descriptors", decidedBy: "kev-27b" }];
    s.lastShiftCycle = 5;
    s.stalledCycles = 2;
    for (const cycle of [5, 6]) {
      s.cycle = cycle;
      expect(status(s)).toBe(" !! PARADIGM SHIFT TRIGGERED (PLATEAU DETECTED AT CYCLE 5 - ROUTED TO KEV-27B): TMA TENSOR DESCRIPTORS !! ");
    }
    s.cycle = 7;
    expect(status(s)).toBe("status: stagnating 2/3 cycles without >1% gain");
    s.stalledCycles = 0;
    expect(status(s)).toBe("status: improving (local search)");
  });

  test("status: the stop reason once summary is set, even right after a shift", () => {
    const s = populated();
    expect(s.summary).toBeDefined();
    s.lastShiftCycle = s.cycle;
    expect(status(s)).toBe(`status: ${run.summary.stopReason}`);
  });

  test("the dashboard header is now 4 lines (header + telemetry) and frames stay exact", () => {
    const plain = renderDashboard(populated(), { width: 140, height: 44, color: false });
    expect(plain[2]).toContain("KEV decisions/s - (heuristic)");
    expect(plain[3]).toContain("status: converged");
    const colored = renderDashboard(populated(), { width: 140, height: 44, color: true });
    for (const l of colored) expect(visibleLength(l)).toBe(140);
  });
});

describe("sparkline", () => {
  test("empty input gives an empty string", () => {
    expect(sparkline([], 10)).toBe("");
  });

  test("constant input renders at full height", () => {
    expect(sparkline([5, 5, 5], 10)).toBe("###");
  });

  test("scales to the max and keeps only the last `width` samples", () => {
    expect(sparkline([0, 1, 2, 3, 4, 5, 6, 7], 8)).toBe(" .:-=+*#");
    const out = sparkline(Array.from({ length: 50 }, (_, i) => i), 20);
    expect(out).toHaveLength(20);
    expect(out.at(-1)).toBe("#");
  });

  test("all zeros do not divide by zero", () => {
    expect(sparkline([0, 0], 5)).toBe("  ");
  });
});
