/**
 * Keeps the self-hosted Kev containers warm: bun run keepalive [--every 180] [--minutes 60]
 *
 * Pings each model with KevClient.warmup() (GET /v1/models on self-hosted
 * endpoints: free, no Kev quota). Modal scales idle containers to zero after
 * 5 minutes, so the default 3-minute interval keeps them up. Containers bill
 * while up (about $3.95/h kev-9b + $6.25/h kev-27b), so this stops on its own
 * after --minutes. Gateway-only models are skipped because warming them costs quota.
 */

import { KevClient, type KevModel } from "../src/harness/kevClient";

function flag(argv: readonly string[], name: string, fallback: number): number {
  const i = argv.indexOf(name);
  const n = i === -1 ? fallback : Number(argv[i + 1]);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${name} expects a positive number`);
  return n;
}

export async function keepalive(argv: readonly string[], env: Record<string, string | undefined> = process.env): Promise<void> {
  const everyMs = flag(argv, "--every", 180) * 1000;
  const stopAt = Date.now() + flag(argv, "--minutes", 60) * 60_000;
  const kev = KevClient.fromEnv(env);
  const models = (["kev-9b", "kev-27b"] as KevModel[]).filter((m) => kev.isSelfHosted(m));
  if (models.length === 0) throw new Error("no self-hosted Kev endpoints configured (KEV_9B_BASE_URL / KEV_27B_BASE_URL)");

  while (Date.now() < stopAt) {
    const result = await kev.warmup(models);
    const stamp = new Date().toLocaleTimeString();
    console.log(`${stamp} ${models.map((m) => `${m} ${typeof result[m] === "number" ? `ok ${(result[m] as number).toFixed(0)}ms` : `FAIL ${result[m]}`}`).join(", ")}`);
    await Bun.sleep(Math.max(0, Math.min(everyMs, stopAt - Date.now())));
  }
  console.log("keepalive: time limit reached, stopping (containers will scale to zero after 5 idle minutes)");
}

if (import.meta.main) {
  keepalive(process.argv.slice(2)).catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
