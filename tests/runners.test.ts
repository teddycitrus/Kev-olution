import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadBenchmark } from "../src/harness/benchmark";
import { Evaluator, extractFeatures } from "../src/harness/evaluator";
import { detectGpuRunner, parseHarnessOutput, PythonGpuRunner, type RunRequest } from "../src/harness/runners";
import type { BenchmarkSpec } from "../src/types";

const STUB_HARNESS = `
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--candidate")
ap.add_argument("--reference")
ap.add_argument("--problem")
ap.add_argument("--tolerance", type=float)
ap.add_argument("--tier", type=int)
a = ap.parse_args()
src = open(a.candidate).read()
mode = src.split("# MODE:", 1)[1].split()[0] if "# MODE:" in src else "success"

if mode == "success":
    print("compiling kernel...")
    print(json.dumps({"issues": ["stale"]}))
    print(json.dumps({"issues": [], "score": 123.5, "latency_ms": 4.25, "max_abs_error": 0.001}))
elif mode == "numerical":
    print(json.dumps({"issues": ["numerical: max abs error 0.5 exceeds tolerance 0.02"], "max_abs_error": 0.5}))
elif mode == "crash":
    print("Traceback (most recent call last):", file=sys.stderr)
    print("RuntimeError: CUDA error: illegal memory access", file=sys.stderr)
    sys.exit(3)
elif mode == "sleep":
    time.sleep(30)
elif mode == "env":
    leaked = sorted(k for k in os.environ if "KEV" in k.upper() or "API_KEY" in k.upper() or "SECRET" in k.upper())
    print(json.dumps({"issues": ["leaked: " + ",".join(leaked)] if leaked else []}))
elif mode == "echo":
    info = {
        "problem": json.loads(a.problem),
        "tolerance": a.tolerance,
        "tier": a.tier,
        "cwd": os.getcwd(),
        "candidate_in_cwd": os.path.dirname(os.path.abspath(a.candidate)) == os.getcwd(),
        "reference": os.path.basename(a.reference),
        "flags": {"ignore_environment": sys.flags.ignore_environment, "safe_path": getattr(sys.flags, "safe_path", None)},
    }
    print(json.dumps({"issues": ["echo: " + json.dumps(info)]}))
`;

const spec: BenchmarkSpec = await loadBenchmark("triton_flash_attention");
const dir = mkdtempSync(join(tmpdir(), "kev-runner-"));
writeFileSync(join(dir, "harness.py"), STUB_HARNESS);
writeFileSync(join(dir, "reference.py"), "");
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const request = (mode: string, tier: 2 | 3 = 2, overrides: Partial<BenchmarkSpec> = {}): RunRequest => {
  const source = `${spec.baselineSource}\n# MODE: ${mode}\n`;
  return { tier, source, features: extractFeatures(source), spec: { ...spec, ...overrides } };
};

const echo = async (runner: PythonGpuRunner, req: RunRequest) => {
  const out = await runner.run(req);
  const line = out.issues[0]!;
  expect(line.startsWith("echo: ")).toBe(true);
  return JSON.parse(line.slice("echo: ".length));
};

describe("parseHarnessOutput", () => {
  test("last JSON line wins over earlier JSON and log lines", () => {
    expect(parseHarnessOutput('{"issues": ["first"]}\nlog line\n{"issues": [], "score": 9.5}\ntrailing log\n')).toEqual({
      issues: [],
      score: 9.5,
      latencyMs: undefined,
      maxAbsError: undefined,
    });
  });

  test("maps snake_case fields and stringifies issues", () => {
    expect(parseHarnessOutput('{"issues": ["compile: x", 7], "score": 1, "latency_ms": 2, "max_abs_error": 0}')).toEqual({
      issues: ["compile: x", "7"],
      score: 1,
      latencyMs: 2,
      maxAbsError: 0,
    });
  });

  test("drops non-finite and non-numeric values", () => {
    expect(parseHarnessOutput('{"issues": [], "score": "fast", "latency_ms": null, "max_abs_error": 1e999}')).toEqual({
      issues: [],
      score: undefined,
      latencyMs: undefined,
      maxAbsError: undefined,
    });
  });

  test.each([
    ["empty output", ""],
    ["no JSON", "Traceback (most recent call last):\n  boom\n"],
    ["malformed JSON", "{oops\n"],
    ["issues missing", '{"score": 1}\n'],
    ["issues not an array", '{"issues": "none"}\n'],
  ])("returns undefined for %s", (_label, stdout) => {
    expect(parseHarnessOutput(stdout)).toBeUndefined();
  });
});

describe("PythonGpuRunner with a stub harness", () => {
  const runner = new PythonGpuRunner({ benchmarkDir: dir, timeoutMs: { 2: 20_000, 3: 20_000 } });

  test("success: parses the final JSON line and measures cost", async () => {
    const out = await runner.run(request("success"));
    expect(out).toMatchObject({ issues: [], score: 123.5, latencyMs: 4.25, maxAbsError: 0.001 });
    expect(out.costGpuSeconds).toBeGreaterThan(0);
  });

  test("numerical failure is passed through with its prefix", async () => {
    const out = await runner.run(request("numerical"));
    expect(out.issues).toEqual(["numerical: max abs error 0.5 exceeds tolerance 0.02"]);
    expect(out.maxAbsError).toBe(0.5);
  });

  test("a crash with no JSON output becomes a runtime issue with the stderr tail", async () => {
    const out = await runner.run(request("crash"));
    expect(out.issues).toHaveLength(1);
    expect(out.issues[0]).toStartWith("runtime: harness exited 3 without a result.");
    expect(out.issues[0]).toContain("illegal memory access");
  });

  test(
    "a hung harness is killed at the tier timeout",
    async () => {
      const fast = new PythonGpuRunner({ benchmarkDir: dir, timeoutMs: { 2: 750 } });
      const started = performance.now();
      const out = await fast.run(request("sleep"));
      expect(out.issues).toEqual(["timeout: tier 2 exceeded 750 ms"]);
      expect(performance.now() - started).toBeLessThan(10_000);
    },
    20_000,
  );

  describe("secrets are not forwarded to the child", () => {
    const saved: Record<string, string | undefined> = {};
    const planted = ["KEV_API_KEY", "KEV_9B_API_KEY", "KEV_27B_API_KEY", "OPENAI_API_KEY", "MY_SECRET"];
    beforeAll(() => {
      for (const k of planted) {
        saved[k] = process.env[k];
        process.env[k] = `test-only-${k.toLowerCase()}`;
      }
    });
    afterAll(() => {
      for (const k of planted) {
        if (saved[k] === undefined) delete process.env[k];
        else process.env[k] = saved[k];
      }
    });

    test("KEV_API_KEY and other keys are absent from the harness environment", async () => {
      expect(process.env.KEV_API_KEY).toBe("test-only-kev_api_key");
      const out = await runner.run(request("env"));
      expect(out.issues).toEqual([]);
    });
  });

  test("runs in a fresh temp cwd with the candidate inside it, and removes it afterwards", async () => {
    const info = await echo(runner, request("echo"));
    expect(info.candidate_in_cwd).toBe(true);
    expect(info.cwd).not.toBe(process.cwd());
    expect(existsSync(info.cwd)).toBe(false);
    expect(info.reference).toBe("reference.py");
    expect(info.flags.ignore_environment).toBe(1);
  });

  test("Tier 2 runs the mini problem, Tier 3 the full problem", async () => {
    const t2 = await echo(runner, request("echo", 2));
    const t3 = await echo(runner, request("echo", 3));
    expect(t2.tier).toBe(2);
    expect(t2.problem).toMatchObject({ batch: 1, heads: 4, seqLen: 1024, headDim: 128 });
    expect(t3.tier).toBe(3);
    expect(t3.problem).toEqual(spec.problem);
  });

  test("tolerance defaults to 2e-2 when the spec demands bit-exactness; options override", async () => {
    expect((await echo(runner, request("echo"))).tolerance).toBe(0.02);
    expect((await echo(runner, request("echo", 2, { errorTolerance: 1e-3 }))).tolerance).toBe(0.001);
    const custom = new PythonGpuRunner({ benchmarkDir: dir, errorTolerance: 5e-4, miniProblem: { seqLen: 256 } });
    const info = await echo(custom, request("echo"));
    expect(info.tolerance).toBe(0.0005);
    expect(info.problem.seqLen).toBe(256);
  });

  test("plugs into the Evaluator: runner results drive Tiers 2 and 3", async () => {
    const evaluator = new Evaluator(spec, { runner });
    expect(evaluator.backend).toBe("gpu");
    const ok = await evaluator.evaluate(`${spec.baselineSource}\n# MODE: success\n`);
    expect(ok.ok).toBe(true);
    expect(ok.metrics).toMatchObject({ tier: 3, score: 123.5 });
    const crash = await evaluator.evaluate(`${spec.baselineSource}\n# MODE: crash\n`);
    expect(crash.failure).toMatchObject({ kind: "runtime", tier: 2 });
  });
});

describe("detectGpuRunner", () => {
  test(
    "returns undefined on a machine without torch+triton+CUDA",
    async () => {
      expect(await detectGpuRunner({ benchmarkDir: dir })).toBeUndefined();
    },
    60_000,
  );

  test("returns undefined when the python binary does not exist", async () => {
    expect(await detectGpuRunner({ benchmarkDir: dir, python: "definitely-not-a-python-binary" })).toBeUndefined();
  });

  test("returns a runner when the probe prints CUDA_OK", async () => {
    const fake = join(dir, process.platform === "win32" ? "fakepy.cmd" : "fakepy.sh");
    writeFileSync(fake, process.platform === "win32" ? "@echo CUDA_OK\r\n" : "#!/bin/sh\necho CUDA_OK\n", { mode: 0o755 });
    const runner = await detectGpuRunner({ benchmarkDir: dir, python: fake });
    expect(runner).toBeInstanceOf(PythonGpuRunner);
    expect(runner?.kind).toBe("gpu");
  });
});
