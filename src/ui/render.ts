/**
 * Pure dashboard rendering: DashboardState -> lines of text. No I/O, no
 * terminal state, so every panel is testable as plain strings. Color is ANSI
 * SGR only and can be disabled; all layout math uses visibleLength().
 */

import type { RoutingSource } from "../harness/runner";
import { percentile, type DagNode, type DashboardState, type LogLine } from "./state";

export interface RenderOptions {
  width: number;
  height: number;
  color: boolean;
}

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

const SGR = {
  reset: "0", bold: "1", dim: "2", red: "31", green: "32", yellow: "33", blue: "34", magenta: "35", cyan: "36", gray: "90",
  gold: "1;33", alertRed: "1;97;41", alertYellow: "1;30;43", badge: "1;30;42",
} as const;
type Style = keyof typeof SGR;

export const paint = (text: string, style: Style, on: boolean) => (on && text ? `\x1b[${SGR[style]}m${text}\x1b[0m` : text);

// eslint-disable-next-line no-control-regex
const ANSI = /\x1b\[[0-9;]*m/g;
export const visibleLength = (s: string) => s.replace(ANSI, "").length;

/** Pads or truncates to exactly `width` visible columns, preserving ANSI codes. */
export function fitWidth(s: string, width: number): string {
  if (width <= 0) return "";
  const len = visibleLength(s);
  if (len <= width) return s + " ".repeat(width - len);
  let out = "";
  let seen = 0;
  let styled = false;
  for (let i = 0; i < s.length && seen < width - 1; ) {
    if (s[i] === "\x1b") {
      const end = s.indexOf("m", i);
      out += s.slice(i, end + 1);
      styled = true;
      i = end + 1;
    } else {
      out += s[i++];
      seen += 1;
    }
  }
  // "~" marks truncation; close any open style first.
  return `${out}${styled ? "\x1b[0m" : ""}~`;
}

function box(title: string, body: string[], width: number, color: boolean): string[] {
  const inner = width - 4;
  const head = `+- ${paint(title, "bold", color)} ${"-".repeat(Math.max(0, width - visibleLength(title) - 5))}+`;
  return [head, ...body.map((l) => `| ${fitWidth(l, inner)} |`), `+${"-".repeat(width - 2)}+`];
}

/** Modal H100 rate used for the dollar estimate (Devin's deployment, about $3.95/h). */
export const GPU_DOLLARS_PER_HOUR = 3.95;

const fmt = (x: number | undefined, digits = 1) => (x === undefined || !Number.isFinite(x) ? "-" : x.toFixed(digits));

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

export const TITLE = "KEV-EVOLVE: Frontier Research & Optimization Harness";

export function renderHeader(s: DashboardState, o: RenderOptions): string[] {
  const elapsed = s.startedAt === undefined ? 0 : (s.now - s.startedAt) / 1000;
  const best = s.champion?.score;
  const speedup = best && s.baselineScore ? best / s.baselineScore : undefined;
  const backend = s.backend === "simulated" ? paint("SIMULATED H100", "yellow", o.color) : paint("GPU", "green", o.color);
  const routing = s.routing === "kev" ? paint("Kev routing", "cyan", o.color) : paint("offline routing", "yellow", o.color);
  const state = s.summary ? paint(`done: ${s.summary.stopReason}`, "green", o.color) : `cycle ${s.cycle}`;
  const hit = best !== undefined && best >= s.targetScore ? paint(" TARGET REACHED", "green", o.color) : "";
  return [
    `${paint(TITLE, "bold", o.color)} | ${backend} | ${routing}`,
    `${s.benchmark} | ${state} | ${elapsed.toFixed(1)}s | best ${paint(`${fmt(best)} TFLOPS`, "bold", o.color)} (${speedup ? `${speedup.toFixed(2)}x` : "-"} baseline ${fmt(s.baselineScore)}, target ${fmt(s.targetScore)})${hit}`,
  ];
}

/** Which Kev model is driving decisions right now. */
export function activeModel(s: DashboardState): string {
  const last = s.shifts[s.shifts.length - 1];
  if (last && last.cycle === s.cycle && !s.summary) return last.decidedBy === "kev-27b" ? "Kev-27B" : "fallback order";
  return s.routing === "kev" ? "Kev-9B" : "heuristic (offline)";
}

/**
 * Kev telemetry bar plus stagnation status. Every number is measured from the
 * run; dollar savings are an estimate from GPU-seconds avoided at
 * GPU_DOLLARS_PER_HOUR and are labeled as such.
 */
export function renderTelemetry(s: DashboardState, color: boolean): string[] {
  const perSec = s.decisionMs > 0 ? (s.decisions * 1000) / s.decisionMs : undefined;
  const model = percentile(s.modelLatency, 50);
  const rtt = percentile([...s.kevLatency.network, ...s.kevLatency.cache, ...s.kevLatency.inflight], 50);
  const st = s.stats;
  const savedS = st ? st.gpuSecondsSavedEarlyExit + st.gpuSecondsSavedByPruningEstimate : 0;
  const share = st && savedS + st.gpuSecondsSpent > 0 ? (100 * savedS) / (savedS + st.gpuSecondsSpent) : undefined;
  const dollars = (savedS / 3600) * GPU_DOLLARS_PER_HOUR;
  const bar = [
    `${paint("KEV", "bold", color)} decisions/s ${paint(perSec === undefined ? "-" : perSec >= 10_000 ? `${(perSec / 1000).toFixed(0)}k` : perSec.toFixed(1), "cyan", color)}${decisionSource(s)}`,
    `routing latency ${paint(rtt === undefined ? "-" : `${rtt.toFixed(1)} ms`, "cyan", color)} (model ${model === undefined ? "-" : `${model.toFixed(1)} ms`})`,
    `cost savings ${paint(share === undefined ? "-" : `${share.toFixed(0)}%`, "green", color)} ($${dollars.toFixed(2)} est.)`,
    `active model ${paint(activeModel(s), "bold", color)}`,
  ].join(" | ");
  const shiftNow = s.lastShiftCycle !== undefined && s.cycle - s.lastShiftCycle <= 1 && !s.summary;
  const last = s.shifts[s.shifts.length - 1];
  // Alternates red/yellow every 400 ms while the shift is active.
  const flash: Style = Math.floor(s.now / 400) % 2 === 0 ? "alertRed" : "alertYellow";
  const route = last?.decidedBy === "kev-27b" ? "ROUTED TO KEV-27B" : "KEV-27B UNAVAILABLE, FALLBACK ORDER";
  const status = shiftNow && last
    ? paint(` !! PARADIGM SHIFT TRIGGERED (PLATEAU DETECTED AT CYCLE ${last.cycle} - ${route}): ${last.label.toUpperCase()} !! `, flash, color)
    : s.summary
      ? paint(`status: ${s.summary.stopReason}`, "green", color)
      : s.stalledCycles > 0
        ? paint(`status: stagnating ${s.stalledCycles}/3 cycles without >1% gain`, "yellow", color)
        : paint("status: improving (local search)", "green", color);
  return [bar, status];
}

/** " (cache)" / " (network)" / " (mixed)" so a cache-replay rate is never mistaken for live Kev throughput. */
function decisionSource(s: DashboardState): string {
  const net = s.kevLatency.network.length;
  const cached = s.kevLatency.cache.length + s.kevLatency.inflight.length;
  if (net + cached === 0) return s.kevLatency.heuristic.length ? " (heuristic)" : "";
  return net === 0 ? " (cache)" : cached === 0 ? " (network)" : " (mixed)";
}

const SPARK = " .:-=+*#";

/** One-line latency history (last `width` samples), scaled to the max. */
export function sparkline(values: readonly number[], width: number): string {
  const v = values.slice(-width);
  const max = Math.max(...v, 1e-9);
  return v.map((x) => SPARK[Math.min(SPARK.length - 1, Math.round((x / max) * (SPARK.length - 1)))]).join("");
}

/**
 * Column chart of best TFLOPS per cycle, target line, and "^" markers on the
 * cycles where a Paradigm Shift fired. Cycles are bucketed (max) when there
 * are more cycles than columns.
 */
export function renderChart(s: DashboardState, width: number, height: number, color: boolean): string[] {
  const labelW = 7;
  const plotW = Math.max(8, width - labelW - 1);
  const rows = Math.max(3, height);
  const values = s.series.map((p) => p.score);
  if (values.length === 0) return [`${" ".repeat(labelW)}| waiting for the baseline...`];

  // Up to 4 columns per cycle so short runs fill the plot; bucket (max) when cycles exceed columns.
  const colW = Math.max(1, Math.min(4, Math.floor(plotW / Math.max(values.length, 30))));
  const buckets = Math.min(Math.floor(plotW / colW), values.length);
  const per = values.length / buckets;
  const cols: Array<{ v: number; shift: boolean }> = [];
  for (let b = 0; b < buckets; b++) {
    const from = Math.floor(b * per);
    const to = Math.max(from + 1, Math.floor((b + 1) * per));
    const slice = s.series.slice(from, to);
    const col = { v: Math.max(...slice.map((p) => p.score)), shift: slice.some((p) => s.shifts.some((m) => m.cycle === p.cycle)) };
    for (let k = 0; k < colW; k++) cols.push({ ...col, shift: col.shift && k === 0 });
  }

  const yMax = Math.max(s.targetScore * 1.1, ...values) * 1.02;
  const rowValue = (r: number) => (yMax * (rows - r)) / rows;
  const targetRow = Math.min(rows - 1, Math.max(0, Math.round(rows - (s.targetScore / yMax) * rows)));
  const out: string[] = [];
  for (let r = 0; r < rows; r++) {
    const top = rowValue(r);
    const half = top - yMax / rows / 2;
    let line = "";
    for (const c of cols) {
      if (c.v >= top) line += paint("#", c.v >= s.targetScore ? "green" : "cyan", color);
      else if (c.v >= half) line += paint(".", c.v >= s.targetScore ? "green" : "cyan", color);
      else line += r === targetRow ? paint("-", "yellow", color) : " ";
    }
    if (r === targetRow) line += paint("-".repeat(Math.max(0, plotW - cols.length)), "yellow", color);
    const label = r === 0 ? fmt(yMax, 0) : r === targetRow ? fmt(s.targetScore, 0) : r === rows - 1 ? "0" : "";
    out.push(`${label.padStart(labelW - 1)} |${line}`);
  }
  out.push(`${" ".repeat(labelW - 1)} +${"-".repeat(plotW)}`);
  out.push(`${" ".repeat(labelW + 1)}${cols.map((c) => (c.shift ? paint("^", "magenta", color) : " ")).join("")}`);
  const lastShift = s.shifts[s.shifts.length - 1];
  out.push(
    `${" ".repeat(labelW + 1)}${paint("^", "magenta", color)} paradigm shift (${s.shifts.length})` +
      (lastShift ? `  last: ${lastShift.label} [${lastShift.decidedBy}] @ cycle ${lastShift.cycle}` : "") +
      `  ${paint("-", "yellow", color)} target`,
  );
  return out;
}

/** Green = viable, red = pruned or failed, gold = Paradigm Shift, gray = purged/pending. */
const STATUS_STYLE: Record<DagNode["status"], Style> = { pending: "gray", pruned: "red", evaluated: "green", failed: "red", purged: "gray" };

/**
 * Search DAG summary: the champion's lineage as a tree (root first), each
 * ancestor annotated with how its other children ended, then the latest
 * generation of candidates.
 */
export function renderDag(s: DashboardState, maxLines: number, color: boolean): string[] {
  if (!s.champion) return ["(no nodes yet)"];
  const children = new Map<string, DagNode[]>();
  for (const id of s.order) {
    const n = s.nodes.get(id)!;
    if (n.parentId) children.set(n.parentId, [...(children.get(n.parentId) ?? []), n]);
  }
  const path: DagNode[] = [];
  for (let n = s.nodes.get(s.champion.id); n; n = n.parentId ? s.nodes.get(n.parentId) : undefined) path.unshift(n);

  const tally = (id: string, skip?: string) => {
    const counts: Record<string, number> = {};
    for (const c of children.get(id) ?? []) if (c.id !== skip) counts[c.status] = (counts[c.status] ?? 0) + 1;
    const parts = (["evaluated", "failed", "pruned", "purged", "pending"] as const).filter((k) => counts[k]).map((k) => paint(`${counts[k]} ${k}`, STATUS_STYLE[k], color));
    return parts.length ? `  [${parts.join(", ")}]` : "";
  };

  const lineage: string[] = [];
  path.forEach((n, i) => {
    // The lineage is a single chain, so one column per level (capped) keeps the champion marker on screen.
    const prefix = i === 0 ? "" : `${" ".repeat(Math.min(i - 1, 3))}\`-`;
    const isChamp = n.id === s.champion!.id;
    const shift = n.kind === "paradigm_shift" ? paint(" SHIFT", "gold", color) : "";
    const name = paint(n.id, n.kind === "paradigm_shift" ? "gold" : n.status === "evaluated" || isChamp ? "green" : "red", color);
    // TFLOPS and the champion marker come before the config so truncation only ever cuts the config.
    lineage.push(`${prefix}${name} ${fmt(n.score)}${isChamp ? paint(" <- champion", "green", color) : ""}${shift} ${n.config}${tally(n.id, path[i + 1]?.id)}`);
  });

  const candidates: string[] = [];
  const latest = s.lastCandidates.map((id) => s.nodes.get(id)).filter((n): n is DagNode => Boolean(n));
  if (latest.length) {
    candidates.push("", paint(`cycle ${s.cycle} candidates`, "bold", color));
    for (const n of latest) {
      const score = n.kevScore === undefined ? "" : ` kev ${n.kevScore.toFixed(2)}`;
      candidates.push(`  ${paint(n.status.padEnd(9), n.kind === "paradigm_shift" && n.status === "evaluated" ? "gold" : STATUS_STYLE[n.status], color)} ${n.id} ${n.strategy}${score}${n.score !== undefined ? ` ${fmt(n.score)}` : ""}`);
    }
  }
  // The lineage has priority; candidates get what is left (or nothing if under 3 lines).
  const lineageRoom = Math.min(lineage.length, maxLines);
  const trimmed =
    lineage.length <= lineageRoom ? lineage
    : [lineage[0]!, paint(`  ... ${lineage.length - lineageRoom + 1} more ancestors`, "gray", color), ...lineage.slice(lineage.length - lineageRoom + 2)];
  const candRoom = maxLines - trimmed.length;
  return candRoom >= 3 ? [...trimmed, ...candidates.slice(0, candRoom)] : trimmed;
}

const SOURCES: readonly RoutingSource[] = ["network", "cache", "inflight", "heuristic"];

export function renderKev(s: DashboardState, color: boolean): string[] {
  const lines = ["source     calls   p50 ms   p95 ms"];
  for (const src of SOURCES) {
    const v = s.kevLatency[src];
    if (v.length === 0) continue;
    const row = `${src.padEnd(9)} ${String(v.length).padStart(6)} ${fmt(percentile(v, 50)).padStart(8)} ${fmt(percentile(v, 95)).padStart(8)}`;
    lines.push(src === "heuristic" ? paint(row, "yellow", color) : row);
  }
  if (lines.length === 1) lines.push(paint("no routing decisions yet", "gray", color));
  const rtts = [...s.kevLatency.network, ...s.kevLatency.cache, ...s.kevLatency.inflight];
  if (rtts.length > 1) lines.push(`latency ${paint(sparkline(rtts, 40), "cyan", color)} max ${fmt(Math.max(...rtts))} ms`);
  const calls = s.stats?.kevCalls;
  lines.push(`billed calls: kev-9b ${calls?.["kev-9b"] ?? 0}, kev-27b ${calls?.["kev-27b"] ?? 0}${s.kevErrors ? paint(`  errors ${s.kevErrors}`, "red", color) : ""}`);
  return lines;
}

export function renderCost(s: DashboardState, color: boolean): string[] {
  const st = s.stats;
  if (!st) return [paint("no evaluations yet", "gray", color)];
  const saved = st.gpuSecondsSavedEarlyExit + st.gpuSecondsSavedByPruningEstimate;
  const share = saved + st.gpuSecondsSpent > 0 ? (100 * saved) / (saved + st.gpuSecondsSpent) : 0;
  return [
    `candidates ${st.candidates}  AST-rejected ${st.astRejected}  Kev-pruned ${st.kevPruned}  evaluated ${st.evaluated}`,
    `GPU-s spent ${fmt(st.gpuSecondsSpent)}`,
    `GPU-s saved: early exit ${fmt(st.gpuSecondsSavedEarlyExit)} + pruning ${fmt(st.gpuSecondsSavedByPruningEstimate)} (est.)`,
    `share avoided ${paint(`${share.toFixed(0)}%`, "green", color)}${s.backend === "simulated" ? paint("  (simulated tier costs)", "yellow", color) : ""}`,
  ];
}

const LOG_STYLE: Record<LogLine["kind"], Style> = { champion: "green", plateau: "yellow", shift: "magenta", fail: "red", kev: "yellow", info: "cyan" };

export function renderLog(s: DashboardState, maxLines: number, color: boolean): string[] {
  return s.log.slice(-maxLines).map((l) => `${paint(`c${String(l.cycle).padStart(2)}`, "gray", color)} ${paint(l.text, LOG_STYLE[l.kind], color)}`);
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

/**
 * Verification badge for the frozen final frame. Worded by backend: the
 * simulated model's error is a model output, not a PyTorch run on hardware.
 */
export function renderBadge(s: DashboardState, color: boolean): string | undefined {
  if (!s.summary || !s.champion) return undefined;
  const err = s.nodes.get(s.champion.id)?.maxAbsError;
  if (err === undefined) return undefined;
  const result = `${s.champion.score.toFixed(1)} TFLOPS (${(s.champion.score / s.baselineScore).toFixed(2)}x)`;
  const text =
    s.backend === "gpu"
      ? `[VERIFIED ON GPU: max abs error ${err.toExponential(2)} vs PYTORCH REFERENCE | ${result}]`
      : err === 0
        ? `[0.000% NUMERICAL ERROR vs REFERENCE (SIMULATED H100 MODEL) | ${result}]`
        : `[max abs error ${err.toExponential(2)} (SIMULATED H100 MODEL) | ${result}]`;
  return paint(` ${text} `, err === 0 || s.backend === "gpu" ? "badge" : "alertYellow", color);
}

/**
 * Full frame: exactly `height` lines, each exactly `width` columns.
 * Wide (>= 120): top = title + telemetry + status, left = DAG, right = chart
 * with Kev and cost below it, bottom = event log. Narrow: stacked.
 */
export function renderDashboard(s: DashboardState, o: RenderOptions): string[] {
  const width = Math.max(60, o.width);
  const height = Math.max(24, o.height);
  const badge = renderBadge(s, o.color);
  const top = [...renderHeader(s, o), ...renderTelemetry(s, o.color), ...(badge ? [badge] : [])].map((l) => fitWidth(l, width));
  const logH = Math.max(4, Math.floor(height * 0.2));
  const midH = height - top.length - logH;
  const log = box("Event log", renderLog(s, logH - 2, o.color), width, o.color);

  let middle: string[];
  if (width >= 120) {
    const leftW = Math.floor(width * 0.5);
    const rightW = width - leftW - 1;
    const kev = box("Kev decision latency", renderKev(s, o.color), rightW, o.color);
    const cost = box("Compute cost", renderCost(s, o.color), rightW, o.color);
    const chartRows = Math.max(5, midH - kev.length - cost.length - 6);
    const right = [...box("Throughput: best TFLOPS per cycle", renderChart(s, rightW - 4, chartRows, o.color), rightW, o.color), ...kev, ...cost];
    const left = box("Search DAG trajectory (green viable, red pruned, gold shift)", renderDag(s, Math.max(3, midH - 2), o.color), leftW, o.color);
    middle = Array.from({ length: midH }, (_, i) => `${fitWidth(left[i] ?? "", leftW)} ${fitWidth(right[i] ?? "", rightW)}`);
  } else {
    const chart = box("Throughput: best TFLOPS per cycle", renderChart(s, width - 4, Math.max(4, Math.floor(midH * 0.4)), o.color), width, o.color);
    const kev = box("Kev decision latency", renderKev(s, o.color), width, o.color);
    const cost = box("Compute cost", renderCost(s, o.color), width, o.color);
    const dag = box("Search DAG trajectory", renderDag(s, Math.max(3, midH - chart.length - kev.length - cost.length - 2), o.color), width, o.color);
    middle = [...chart, ...kev, ...cost, ...dag].slice(0, midH);
  }
  const frame = [...top, ...middle, ...log].slice(0, height).map((l) => fitWidth(l, width));
  while (frame.length < height) frame.push(" ".repeat(width));
  return frame;
}

/** Final plain summary printed to normal scrollback after the live view closes. */
export function renderFinalSummary(s: DashboardState): string[] {
  const sum = s.summary;
  if (!sum) return ["kev-evolve stopped before finishing"];
  const st = sum.stats;
  return [
    `kev-evolve ${s.benchmark}: ${sum.bestScore.toFixed(1)} TFLOPS, ${sum.speedup.toFixed(2)}x baseline, target ${sum.targetReached ? "reached" : "missed"} [${sum.backend}]`,
    `stop: ${sum.stopReason} after ${sum.cycles} cycles; champion ${sum.championId} ${sum.championSummary ?? ""}`,
    `paradigm shifts: ${s.shifts.map((m) => `c${m.cycle} ${m.label} [${m.decidedBy}]`).join("; ") || "none"}`,
    `candidates ${st.candidates}, AST-rejected ${st.astRejected}, Kev-pruned ${st.kevPruned}, evaluated ${st.evaluated}`,
    `kev calls 9b=${st.kevCalls["kev-9b"]} 27b=${st.kevCalls["kev-27b"]}; GPU-s spent ${st.gpuSecondsSpent.toFixed(1)}, saved early-exit ${st.gpuSecondsSavedEarlyExit.toFixed(1)}, saved by pruning (est.) ${st.gpuSecondsSavedByPruningEstimate.toFixed(1)}`,
  ];
}
