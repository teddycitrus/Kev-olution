import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { renderKernel } from "../src/agents/kernelTemplate";
import { REFLECTOR_MAX_CHARS, Reflector, compressIssue } from "../src/agents/reflector";
import { loadBenchmark } from "../src/harness/benchmark";
import { Evaluator, TIER_COST_GPU_SECONDS, type EvalResult } from "../src/harness/evaluator";
import type { FailureKind, FidelityTier } from "../src/harness/graphStore";

const spec = await loadBenchmark("triton_flash_attention");
const evaluator = new Evaluator(spec);
const reflector = new Reflector();
const setConst = (src: string, name: string, v: number) => src.replace(new RegExp(`^${name} = \\d+`, "m"), `${name} = ${v}`);

function failed(kind: FailureKind, tier: FidelityTier, issues: string[]): EvalResult {
  return {
    ok: false,
    failure: { kind, tier, summary: issues.join("; ").slice(0, 280) },
    features: null,
    tiers: [
      ...(tier > 1 ? [{ tier: 1 as const, passed: true, issues: [], costGpuSeconds: TIER_COST_GPU_SECONDS[1], wallMs: 0 }] : []),
      { tier, passed: false, issues, costGpuSeconds: TIER_COST_GPU_SECONDS[tier], wallMs: 0 },
    ],
    costGpuSeconds: 0,
    savedGpuSeconds: 0,
  };
}

const KINDS: FailureKind[] = ["syntax", "compile", "runtime", "numerical", "timeout", "regression"];

describe("summary length", () => {
  test.each(KINDS)("%s stays at most 279 chars even with 5000+ char issues", (kind) => {
    const tier: FidelityTier = kind === "syntax" ? 1 : kind === "regression" ? 2 : 3;
    const huge = `${kind}: ${"x".repeat(5200)}`;
    const s = reflector.reflect({ nodeId: "n1", source: spec.baselineSource, result: failed(kind, tier, [huge, huge]) });
    expect(REFLECTOR_MAX_CHARS).toBe(279);
    expect(s.summary.length).toBeLessThanOrEqual(279);
    expect(s.summary.startsWith(`T${tier} ${kind}: `)).toBe(true);
    expect(s.summary.endsWith("...")).toBe(true);
    expect(s).toMatchObject({ kind, tier });
  });

  test("drops the hint before truncating the core", () => {
    const issue = `runtime: ${"y".repeat(200)}`;
    const s = reflector.reflect({ nodeId: "n", source: spec.baselineSource, result: failed("runtime", 2, [issue]) });
    expect(s.summary).not.toContain("fix:");
    expect(s.summary).toContain("| cfg 64x32 w4 s1");
    expect(s.summary.endsWith("...")).toBe(false);
  });
});

describe("compressions on real evaluator failures", () => {
  test("shared memory overflow", async () => {
    let src = spec.baselineSource;
    for (const [k, v] of [["BLOCK_M", 128], ["BLOCK_N", 128], ["NUM_WARPS", 8], ["NUM_STAGES", 3]] as const) src = setConst(src, k, v);
    const result = await evaluator.evaluate(src);
    const s = reflector.reflect({ nodeId: "n", source: src, result });
    expect(s.summary).toBe("T2 compile: smem 294912B > 232448B | cfg 128x128 w8 s3 | fix: shrink BLOCK_N or NUM_STAGES");
  });

  test("numerical error", async () => {
    const src = spec.baselineSource.replace("dtype=tl.float32)\n\n    for", "dtype=tl.float16)\n\n    for");
    const result = await evaluator.evaluate(src);
    expect(result.failure?.kind).toBe("numerical");
    const s = reflector.reflect({ nodeId: "n", source: src, result });
    expect(s.summary).toBe("T2 numerical: max|err| 0.00310 > tol 0 | cfg 64x32 w4 s1 | fix: keep fp32 acc and running max; fold log2e into exp2");
  });

  test("regression below the promotion bar", async () => {
    const result = await evaluator.evaluate(spec.baselineSource, { promotionScore: 80 });
    const s = reflector.reflect({ nodeId: "n", source: spec.baselineSource, result });
    expect(s.summary).toMatch(/^T2 regression: est \d+\.\d < bar 80\.0 TFLOPS \| cfg 64x32 w4 s1 \| fix: local basin exhausted; try a structural change$/);
  });

  test("compressIssue strips kind prefixes and collapses whitespace for other issues", () => {
    expect(compressIssue("runtime:   CUDA   error:\n illegal  access")).toBe("CUDA error: illegal access");
    expect(compressIssue("timeout: tier 3 exceeded 300000 ms")).toBe("tier 3 exceeded 300000 ms");
  });

  test("only the first two issues are included", () => {
    const s = reflector.reflect({ nodeId: "n", source: spec.baselineSource, result: failed("compile", 2, ["compile: a", "compile: b", "compile: c"]) });
    expect(s.summary).toContain("a; b |");
    expect(s.summary).not.toContain("c |");
  });
});

describe("tier 1", () => {
  test("summaries have no cfg part", async () => {
    const src = `import subprocess\n${spec.baselineSource}`;
    const result = await evaluator.evaluate(src);
    const s = reflector.reflect({ nodeId: "n", source: src, result });
    expect(s.summary).toBe("T1 syntax: forbidden import 'subprocess' | fix: keep Tier 1 contract (imports, constants, @triton.jit)");
    expect(s.summary).not.toContain("cfg");
  });

  test("an unparseable source does not throw", () => {
    const s = reflector.reflect({ nodeId: "n", source: "garbage", result: failed("syntax", 1, ["empty source"]) });
    expect(s.summary).toStartWith("T1 syntax: empty source");
  });
});

describe("logDir", () => {
  const dir = mkdtempSync(join(tmpdir(), "kev-reflector-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  test("writes the full trace and sets logPath", () => {
    const logDir = join(dir, "logs");
    const result = failed("runtime", 3, [`runtime: ${"z".repeat(5000)}`]);
    const s = new Reflector({ logDir }).reflect({ nodeId: "n42", source: renderKernel({ blockM: 64, blockN: 32, numWarps: 4, numStages: 1, exp2: false, memory: "pointer", warpSpecialize: false, persistent: false }), result });
    expect(s.logPath).toBe(join(logDir, "n42.log"));
    const log = JSON.parse(readFileSync(s.logPath!, "utf8"));
    expect(log.tiers.at(-1).issues[0]).toHaveLength(5000 + "runtime: ".length);
    expect(log.failure.kind).toBe("runtime");
  });

  test("sanitizes ids so logs never escape logDir", () => {
    const logDir = join(dir, "safe");
    const s = new Reflector({ logDir }).reflect({ nodeId: "../x", source: spec.baselineSource, result: failed("compile", 2, ["compile: x"]) });
    expect(dirname(s.logPath!)).toBe(logDir);
    expect(readdirSync(logDir)).toEqual(["___x.log"]);
    expect(existsSync(join(dir, "x.log"))).toBe(false);
    const evil = new Reflector({ logDir }).reflect({ nodeId: "a/b\\c:d*e", source: spec.baselineSource, result: failed("compile", 2, ["compile: x"]) });
    expect(dirname(evil.logPath!)).toBe(logDir);
  });

  test("no logDir means no logPath", () => {
    const s = reflector.reflect({ nodeId: "n", source: spec.baselineSource, result: failed("timeout", 3, ["timeout: x"]) });
    expect(s.logPath).toBeUndefined();
    expect("logPath" in s).toBe(false);
  });
});
