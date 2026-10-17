/**
 * Offline demo rehearsal: bun run rehearse
 *
 * Drives the real evolution loop and live dashboard with no Kev calls (no
 * network, no Modal cost), seed 0, paced so the full run takes about 45 s:
 * baseline 41.2 TFLOPS, four Paradigm Shift alerts, 176.1 TFLOPS at the end.
 * The seed-0 run spends 418.7 simulated GPU-seconds, so --pace 107 gives ~45 s.
 *
 * Extra CLI arguments are appended (e.g. --plain, or --pace 60 for a faster pass).
 */

import { main } from "../src/index";

export const REHEARSAL_ARGS = ["--offline", "--seed", "0", "--pace", "107"] as const;

export function rehearsalArgs(extra: readonly string[] = []): string[] {
  return [...REHEARSAL_ARGS, ...extra];
}

if (import.meta.main) {
  main(rehearsalArgs(process.argv.slice(2))).catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
