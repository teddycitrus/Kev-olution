/**
 * Live demo runner: bun run demo
 *
 * 1. Wakes the Kev containers with KevClient.warmup() and prints per-model
 *    readiness. Self-hosted cold starts take up to about 6.5 minutes.
 * 2. Runs the paced evolution loop with the demo cache:
 *    main(["--demo", "--pace", "120", "--seed", "0"]) takes about 45 s on cache replay.
 * 3. Prints where the demo cache snapshot was saved; a second run with the
 *    same seed replays Kev decisions from that cache.
 *
 * Extra CLI arguments are appended to the main() call, and later flags win:
 *   bun run demo --plain        plain-text log instead of the dashboard
 *   bun run demo --pace 140     slower pacing (about 60 s on cache replay)
 *   bun run demo:offline        no Kev calls and no warmup (fallback if endpoints are cold), ~50 s
 */

import { resolve } from "node:path";
import { KevClient, type KevModel } from "../src/harness/kevClient";
import { DEMO_SNAPSHOT, main } from "../src/index";

export const DEMO_ARGS = ["--demo", "--pace", "100", "--seed", "0"] as const;

export interface DemoDeps {
  env: Record<string, string | undefined>;
  /** Builds the client used for warmup. Default: KevClient.fromEnv(env). */
  makeKev?: (env: Record<string, string | undefined>) => Pick<KevClient, "warmup" | "isSelfHosted">;
  run?: (argv: readonly string[]) => Promise<void>;
  log?: (line: string) => void;
  extraArgs?: readonly string[];
}

export interface DemoResult {
  warmup: Record<string, number | string> | null;
  ready: boolean;
  argv: string[];
  snapshotPath: string;
}

const MODELS: readonly KevModel[] = ["kev-9b", "kev-27b"];

export async function runDemo(deps: DemoDeps): Promise<DemoResult> {
  const log = deps.log ?? ((line: string) => console.log(line));
  const run = deps.run ?? main;
  const argv = [...DEMO_ARGS, ...(deps.extraArgs ?? [])];
  const snapshotPath = resolve(DEMO_SNAPSHOT);
  const offline = argv.includes("--offline");
  const hasKey = Boolean(deps.env.KEV_API_KEY || deps.env.KEV_9B_API_KEY) && !offline;

  let warmup: Record<string, number | string> | null = null;
  let ready = false;
  if (offline) {
    log("demo: --offline, skipping Kev warmup; routing is heuristic and shifts use the fallback order (26 cycles, ~50 s at pace 120)");
  } else if (hasKey) {
    const kev = (deps.makeKev ?? ((env) => KevClient.fromEnv(env)))(deps.env);
    log("demo: waking Kev containers (self-hosted cold starts can take up to ~6.5 min)...");
    warmup = await kev.warmup(MODELS);
    for (const model of MODELS) {
      const v = warmup[model];
      const where = kev.isSelfHosted(model) ? "self-hosted" : "gateway";
      log(`demo: ${model} (${where}) ${typeof v === "number" ? `ready in ${(v / 1000).toFixed(1)}s` : `NOT READY: ${v}`}`);
    }
    ready = MODELS.every((m) => typeof warmup![m] === "number");
    if (!ready) log("demo: continuing; routing falls back to heuristics or the fallback order for models that are not ready");
  } else {
    log("demo: no KEV_API_KEY or KEV_9B_API_KEY set; the run will be offline");
  }

  log(`demo: running ${["bun run evolve", ...argv].join(" ")}`);
  await run(argv);
  log(
    hasKey
      ? `demo: Kev demo cache saved to ${snapshotPath}; run again with the same seed to replay decisions from cache`
      : "demo: offline run, no Kev decisions were made, so no demo cache was written",
  );
  return { warmup, ready, argv, snapshotPath };
}

if (import.meta.main) {
  // Default KevClient.fromEnv uses the persistent quota store, so warmup calls count toward the daily limits.
  runDemo({ env: process.env, extraArgs: process.argv.slice(2) }).catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
