import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { DEMO_ARGS, runDemo } from "../scripts/demo";
import { KevClient, MemoryQuotaStore, QuotaTracker } from "../src/harness/kevClient";
import { DEMO_SNAPSHOT, parseArgs } from "../src/index";

function fakeKev(results: Record<string, number | string>, selfHosted: Record<string, boolean> = {}) {
  const calls: unknown[] = [];
  return {
    calls,
    makeKev: () => ({
      warmup: async (models?: readonly string[]) => {
        calls.push(models);
        return results;
      },
      isSelfHosted: (m: string) => Boolean(selfHosted[m]),
    }),
  };
}

function capture() {
  const lines: string[] = [];
  const runs: Array<readonly string[]> = [];
  return { lines, runs, log: (l: string) => lines.push(l), run: async (argv: readonly string[]) => void runs.push(argv) };
}

describe("scripts/demo.ts", () => {
  test("the demo arguments parse and match the paced 60 s run", () => {
    expect([...DEMO_ARGS]).toEqual(["--demo", "--pace", "100", "--seed", "0"]);
    expect(parseArgs(DEMO_ARGS)).toMatchObject({ demo: true, paceMs: 100, seed: 0, offline: false, warmup: false });
  });

  test("warms both models, prints readiness, runs main with the demo args, and prints the snapshot path", async () => {
    const kev = fakeKev({ "kev-9b": 1234, "kev-27b": 45678 }, { "kev-9b": true, "kev-27b": true });
    const io = capture();
    const r = await runDemo({ env: { KEV_API_KEY: "k" }, makeKev: kev.makeKev, run: io.run, log: io.log });
    expect(kev.calls).toEqual([["kev-9b", "kev-27b"]]);
    expect(r).toMatchObject({ ready: true, argv: [...DEMO_ARGS], snapshotPath: resolve(DEMO_SNAPSHOT) });
    expect(io.runs).toEqual([[...DEMO_ARGS]]);
    expect(io.lines).toContain("demo: kev-9b (self-hosted) ready in 1.2s");
    expect(io.lines).toContain("demo: kev-27b (self-hosted) ready in 45.7s");
    expect(io.lines).toContain("demo: running bun run evolve --demo --pace 100 --seed 0");
    expect(io.lines.at(-1)).toBe(`demo: Kev demo cache saved to ${resolve(DEMO_SNAPSHOT)}; run again with the same seed to replay decisions from cache`);
  });

  test("a model that fails warmup is reported and the run still proceeds", async () => {
    const kev = fakeKev({ "kev-9b": 900, "kev-27b": "Kev API error 408: timeout after 900000 ms" }, { "kev-9b": true });
    const io = capture();
    const r = await runDemo({ env: { KEV_API_KEY: "k" }, makeKev: kev.makeKev, run: io.run, log: io.log });
    expect(r.ready).toBe(false);
    expect(io.lines).toContain("demo: kev-27b (gateway) NOT READY: Kev API error 408: timeout after 900000 ms");
    expect(io.lines.some((l) => l.includes("falls back"))).toBe(true);
    expect(io.runs).toHaveLength(1);
  });

  test("without any Kev key it skips warmup and runs offline", async () => {
    const kev = fakeKev({});
    const io = capture();
    const r = await runDemo({ env: {}, makeKev: kev.makeKev, run: io.run, log: io.log });
    expect(kev.calls).toHaveLength(0);
    expect(r.warmup).toBeNull();
    expect(io.lines[0]).toContain("the run will be offline");
    expect(io.runs).toHaveLength(1);
    expect(io.lines.at(-1)).toBe("demo: offline run, no Kev decisions were made, so no demo cache was written");
    expect(io.lines.some((l) => l.includes("cache saved"))).toBe(false);
  });

  test("extra CLI arguments are appended", async () => {
    const io = capture();
    const r = await runDemo({ env: {}, run: io.run, log: io.log, extraArgs: ["--plain"] });
    expect(r.argv).toEqual([...DEMO_ARGS, "--plain"]);
    expect(parseArgs(r.argv).plain).toBe(true);
  });

  test("with a real KevClient and mocked fetch, warmup hits only self-hosted /v1/models", async () => {
    const urls: string[] = [];
    const fetchImpl = (async (u: string | URL | Request) => {
      urls.push(String(u));
      return new Response("{}");
    }) as typeof fetch;
    const env = { KEV_API_KEY: "gw", KEV_9B_BASE_URL: "https://a.test", KEV_9B_API_KEY: "a", KEV_27B_BASE_URL: "https://b.test", KEV_27B_API_KEY: "b" };
    const io = capture();
    const r = await runDemo({ env, makeKev: (e) => KevClient.fromEnv(e, { fetchImpl, quota: new QuotaTracker(new MemoryQuotaStore()) }), run: io.run, log: io.log });
    expect(urls.sort()).toEqual(["https://a.test/v1/models", "https://b.test/v1/models"]);
    expect(r.ready).toBe(true);
  });

  test("a failing run propagates its error", async () => {
    await expect(runDemo({ env: {}, log: () => {}, run: async () => { throw new Error("boom"); } })).rejects.toThrow("boom");
  });
});

describe("demo:offline", () => {
  test("package.json demo:offline passes --offline through to the demo script", async () => {
    const pkg = await Bun.file("package.json").json();
    expect(pkg.scripts["demo:offline"]).toBe("bun run scripts/demo.ts --offline");
  });

  test("--offline skips warmup even when keys are set, runs offline, and writes no cache", async () => {
    const kev = fakeKev({ "kev-9b": 1, "kev-27b": 1 });
    const io = capture();
    const r = await runDemo({ env: { KEV_API_KEY: "k", KEV_9B_API_KEY: "k9" }, makeKev: kev.makeKev, run: io.run, log: io.log, extraArgs: ["--offline"] });
    expect(kev.calls).toHaveLength(0);
    expect(r.warmup).toBeNull();
    expect(io.lines[0]).toContain("--offline, skipping Kev warmup");
    expect(parseArgs(io.runs[0]!)).toMatchObject({ offline: true, demo: true });
    expect(io.lines.at(-1)).toContain("no demo cache was written");
  });

  test("a later --pace overrides the demo default", () => {
    expect(parseArgs([...DEMO_ARGS, "--pace", "140"]).paceMs).toBe(140);
  });
});
