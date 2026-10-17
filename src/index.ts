/**
 * CLI: bun run evolve --benchmark triton_flash_attention [options]
 *
 *   --cycles N        max cycles (default 40)
 *   --duration S      wall-clock budget in seconds
 *   --seed N          RNG + evaluator noise seed (default 0)
 *   --demo            Kev response cache (.kev/demo-cache.json), 45 s pitch pacing
 *                     (pace 100 unless --pace is given), final frame held 7 s
 *   --offline         no Kev calls; heuristic routing and fallback shifts
 *   --gpu             use the real Triton runner when torch + triton + CUDA exist
 *   --pace MS         ms slept per simulated GPU-second (demo pacing)
 *   --warmup          wake Kev containers first (self-hosted cold starts take minutes)
 *   --plain           plain-text event log instead of the live dashboard
 *
 * The live dashboard (src/ui/dashboard.ts) runs when stdout is a TTY; otherwise,
 * or with --plain, events are logged as plain text.
 */

import { join } from "node:path";
import { Mutator } from "./agents/mutator";
import { Reflector } from "./agents/reflector";
import { BENCHMARKS_DIR, loadBenchmark } from "./harness/benchmark";
import { Evaluator } from "./harness/evaluator";
import { InMemoryGraphStore } from "./harness/inMemoryGraphStore";
import { KevClient } from "./harness/kevClient";
import { ParadigmShiftEngine, PlateauDetector } from "./harness/plateauDetector";
import { evolve, type EvolveEvent } from "./harness/runner";
import { detectGpuRunner } from "./harness/runners";
import { Dashboard } from "./ui/dashboard";

export const DEMO_SNAPSHOT = ".kev/demo-cache.json";
export const DEMO_PACE_MS = 100;
export const DEMO_HOLD_MS = 7000;

export interface CliArgs {
  benchmark: string;
  cycles?: number;
  durationMs?: number;
  seed: number;
  demo: boolean;
  offline: boolean;
  gpu: boolean;
  paceMs: number;
  warmup: boolean;
  plain: boolean;
}

export function parseArgs(argv: readonly string[]): CliArgs {
  const args: CliArgs = { benchmark: "triton_flash_attention", seed: 0, demo: false, offline: false, gpu: false, paceMs: 0, warmup: false, plain: false };
  const num = (flag: string, v: string | undefined) => {
    const n = Number(v);
    if (v === undefined || !Number.isFinite(n) || n < 0) throw new Error(`${flag} expects a non-negative number`);
    return n;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    switch (a) {
      case "--benchmark": {
        const v = argv[++i];
        if (!v || v.startsWith("--")) throw new Error("--benchmark expects a name");
        args.benchmark = v;
        break;
      }
      case "--cycles": args.cycles = num(a, argv[++i]); break;
      case "--duration": args.durationMs = num(a, argv[++i]) * 1000; break;
      case "--seed": args.seed = num(a, argv[++i]); break;
      case "--pace": args.paceMs = num(a, argv[++i]); break;
      case "--demo": args.demo = true; break;
      case "--offline": args.offline = true; break;
      case "--gpu": args.gpu = true; break;
      case "--warmup": args.warmup = true; break;
      case "--plain": args.plain = true; break;
      default: throw new Error(`unknown argument ${a}`);
    }
  }
  // --demo is the 45 s pitch: ~38 s paced run plus a 7 s freeze on the final frame.
  if (args.demo && !argv.includes("--pace")) args.paceMs = DEMO_PACE_MS;
  return args;
}

function logEvent(e: EvolveEvent): void {
  const t = (x: number) => x.toFixed(1);
  switch (e.type) {
    case "started": return console.log(`kev-evolve ${e.benchmark} | backend=${e.backend} routing=${e.routing} | baseline ${t(e.baselineScore)} -> target ${t(e.targetScore)} TFLOPS`);
    case "cycle_started": return console.log(`\n[cycle ${e.cycle}] parent ${e.parentId} (${e.parentSummary})`);
    case "kev_decision": {
      const scores = Object.entries(e.scores).map(([id, s]) => `${id}:${s.toFixed(2)}`).join(" ");
      return console.log(`  route ${e.model ?? "offline"} via ${e.source} ${t(e.latencyMs)}ms kept=${e.kept.join(",") || "-"} pruned=${e.pruned.join(",") || "-"}${scores ? ` [${scores}]` : ""}${e.error ? ` error=${e.error}` : ""}`);
    }
    case "evaluated": return console.log(`  ${e.ok ? "ok  " : "FAIL"} ${e.id} T${e.tier} ${e.summary}`);
    case "champion": return console.log(`  ** champion ${e.id} ${t(e.score)} TFLOPS (${e.summary})`);
    case "plateau": return console.log(`  !! plateau: ${e.signal.stalledCycles} stalled cycles at ${t(e.signal.bestScore)} TFLOPS`);
    case "paradigm_shift": return console.log(`  >> PARADIGM SHIFT: ${e.directive.strategyLabel} [${e.directive.decidedBy}${e.directive.fallbackReason ? `: ${e.directive.fallbackReason}` : ""}] purged ${e.directive.purgedIds.length}`);
    case "converged": return console.log(`  == ${e.reason}`);
    case "finished": {
      const s = e.summary;
      console.log(`\ndone (${s.stopReason}) after ${s.cycles} cycles: ${t(s.bestScore)} TFLOPS, ${s.speedup.toFixed(2)}x baseline, target ${s.targetReached ? "reached" : "missed"} [${s.backend}]`);
      console.log(`champion ${s.championId} ${s.championSummary}`);
      console.log(`candidates ${s.stats.candidates}, AST-rejected ${s.stats.astRejected}, Kev-pruned ${s.stats.kevPruned}, evaluated ${s.stats.evaluated}, shifts ${s.stats.paradigmShifts}`);
      console.log(`kev calls 9b=${s.stats.kevCalls["kev-9b"]} 27b=${s.stats.kevCalls["kev-27b"]}; GPU-s spent ${t(s.stats.gpuSecondsSpent)}, saved early-exit ${t(s.stats.gpuSecondsSavedEarlyExit)}, saved by pruning (est.) ${t(s.stats.gpuSecondsSavedByPruningEstimate)}`);
      return;
    }
  }
}

export async function main(argv: readonly string[]): Promise<void> {
  const args = parseArgs(argv);
  const benchmark = await loadBenchmark(args.benchmark);
  const runner = args.gpu ? await detectGpuRunner({ benchmarkDir: join(BENCHMARKS_DIR, args.benchmark) }) : undefined;
  if (args.gpu && !runner) console.log("--gpu: torch/triton/CUDA not available, using the simulated backend");
  const evaluator = new Evaluator(benchmark, { seed: args.seed, wallMsPerGpuSecond: args.paceMs, runner });

  const hasKey = Boolean(process.env.KEV_API_KEY || process.env.KEV_9B_API_KEY);
  if (!args.offline && !hasKey) console.log("KEV_API_KEY not set, running offline");
  const kev = args.offline || !hasKey ? null : KevClient.fromEnv(process.env, args.demo ? { demo: { snapshotPath: DEMO_SNAPSHOT } } : {});

  if (kev && args.warmup) {
    console.log("kev warmup: waking containers (cold starts can take several minutes)...");
    const warm = await kev.warmup();
    console.log(`kev warmup: ${Object.entries(warm).map(([m, v]) => `${m} ${typeof v === "number" ? `${v.toFixed(0)}ms` : v}`).join(", ")}`);
  }

  const store = new InMemoryGraphStore();
  const sources = new Set<string>();
  store.subscribe((e) => {
    if (e.type === "node_added") sources.add(e.node.source);
  });

  const dashboard = !args.plain && process.stdout.isTTY ? new Dashboard({ color: !process.env.NO_COLOR }) : undefined;
  dashboard?.start(store);
  try {
    await evolve(
    {
      benchmark,
      store,
      evaluator,
      mutator: new Mutator({ benchmarkDescription: benchmark.description, isKnown: (s) => sources.has(s) }),
      reflector: new Reflector({ logDir: ".kev/logs" }),
      plateau: new PlateauDetector({ window: 3 }),
      shiftEngine: new ParadigmShiftEngine({ store, benchmark, kev }),
      kev,
    },
    { maxCycles: args.cycles, durationMs: args.durationMs, seed: args.seed, onEvent: dashboard?.onEvent ?? logEvent },
    );
    if (dashboard && args.demo) await dashboard.hold(DEMO_HOLD_MS);
  } finally {
    dashboard?.stop();
  }
  if (kev?.cache) kev.cache.saveSnapshot();
}

if (import.meta.main) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
