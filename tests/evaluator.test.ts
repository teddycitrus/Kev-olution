import { describe, expect, test } from "bun:test";
import { loadBenchmark } from "../src/harness/benchmark";
import {
  Evaluator,
  TIER_COST_GPU_SECONDS,
  compileIssues,
  extractFeatures,
  sharedMemBytes,
  staticCheck,
} from "../src/harness/evaluator";
import { MAX_FAILURE_SUMMARY_CHARS } from "../src/harness/graphStore";
import { parseHarnessOutput, type KernelRunner, type RunOutcome, type RunRequest } from "../src/harness/runners";
import { configFromSource, renderKernel, type KernelConfig } from "../src/agents/kernelTemplate";
import type { BenchmarkSpec } from "../src/types";

const spec: BenchmarkSpec = await loadBenchmark("triton_flash_attention");
const baseline = spec.baselineSource;
const evaluator = new Evaluator(spec);

const setConst = (src: string, name: string, value: number) => src.replace(new RegExp(`^${name}\\s*=\\s*\\d+`, "m"), `${name} = ${value}`);

const tune = (src: string, c: { m?: number; n?: number; warps?: number; stages?: number }) => {
  let out = src;
  if (c.m !== undefined) out = setConst(out, "BLOCK_M", c.m);
  if (c.n !== undefined) out = setConst(out, "BLOCK_N", c.n);
  if (c.warps !== undefined) out = setConst(out, "NUM_WARPS", c.warps);
  if (c.stages !== undefined) out = setConst(out, "NUM_STAGES", c.stages);
  return out;
};

const withExp2 = (src: string, rescaled = true) => {
  const out = src.replaceAll("tl.exp(", "tl.exp2(");
  return rescaled ? out.replace("qk = tl.dot(q, k) * sm_scale", "qk = tl.dot(q, k) * (sm_scale * 1.44269504)") : out;
};

const PARADIGM: KernelConfig = { blockM: 128, blockN: 64, numWarps: 8, numStages: 3, exp2: true, memory: "tma", warpSpecialize: true, persistent: true };

/** Full structural rewrite rendered by the template: every feature lives in the launched kernel. */
const paradigmShift = (overrides: Partial<KernelConfig> = {}) => renderKernel({ ...PARADIGM, ...overrides });

const DEAD_CODE_MARKERS = [
  "q_ptr = tl.make_block_ptr(Q, (N, D), (D, 1), (0, 0), (128, 128), (1, 0))",
  "desc = tl.make_tensor_descriptor(Q, [N, D], [D, 1], [128, 128])",
  "tiles = tl.num_programs(0)",
  "for _ in tl.range(0, tiles, warp_specialize=True):",
  "    p = tl.exp2(qk * 1.44269504)",
];

describe("loadBenchmark", () => {
  test("loads manifest and sources", () => {
    expect(spec.name).toBe("triton_flash_attention");
    expect(spec.baselineTflops).toBeGreaterThan(40);
    expect(spec.referenceSource).toContain("torch.softmax");
    expect(spec.baselineSource).not.toContain("\r");
  });

  test("rejects unknown and unsafe names", async () => {
    await expect(loadBenchmark("does_not_exist")).rejects.toThrow("Unknown benchmark");
    await expect(loadBenchmark("../src")).rejects.toThrow("Invalid benchmark name");
  });
});

describe("Tier 1 static checks", () => {
  test("baseline passes", () => {
    expect(staticCheck(baseline)).toEqual([]);
  });

  test.each([
    ["empty source", "   ", "empty source"],
    ["missing jit", baseline.replace("@triton.jit", ""), "no @triton.jit kernel"],
    ["forbidden import", `import os\n${baseline}`, "forbidden import 'os'"],
    ["unbalanced bracket", baseline.replace("tl.program_id(0)", "tl.program_id(0"), "unclosed '('"],
    ["stray closer", `${baseline}\n)\n`, "unbalanced ')'"],
    ["unterminated string", `${baseline}\nx = "oops\n`, "unterminated string"],
    ["tab indentation", baseline.replace("    start_m =", "\tstart_m ="), "tab characters"],
    ["non power of two block", setConst(baseline, "BLOCK_M", 96), "BLOCK_M=96"],
    ["invalid warps", setConst(baseline, "NUM_WARPS", 3), "NUM_WARPS=3"],
    ["invalid stages", setConst(baseline, "NUM_STAGES", 9), "NUM_STAGES=9"],
    ["missing constant", baseline.replace(/^BLOCK_N = \d+$/m, ""), "missing module constant BLOCK_N"],
    ["no tl.dot", baseline.replaceAll("tl.dot(", "tl.mul("), "never calls tl.dot"],
    ["kernel never launched", baseline.replace("_attn_fwd[grid](", "_unused = (").replace("num_warps=NUM_WARPS, num_stages=NUM_STAGES,", ""), "no @triton.jit kernel is launched"],
    ["launch only from dead host code", baseline.replace("def attention(", "def _never_called("), "no @triton.jit kernel is launched"],
  ])("flags %s", (_label, source, expected) => {
    expect(staticCheck(source).join("; ")).toContain(expected);
  });

  test.each([
    ["__import__", 'm = __import__("os")'],
    ["importlib", "import importlib"],
    ["importlib", 'm = importlib.import_module("os")'],
    ["eval(", 'eval("1 + 1")'],
    ["exec(", 'exec("x = 1")'],
    ["compile(", 'c = compile("1", "f", "eval")'],
    ["open(", 'f = open("/etc/passwd")'],
    ["builtins", "import builtins"],
    ["builtins", "b = __builtins__"],
    ["globals(", 'g = globals()["__builtins__"]'],
  ])("rejects dynamic code: %s (%s)", (label, line) => {
    const issues = staticCheck(`${baseline}\n${line}\n`).join("; ");
    expect(issues).toContain(`forbidden dynamic code: ${label}`);
  });

  test("dynamic-code patterns ignore comments, strings, and attribute calls", () => {
    const benign = `${baseline}\n# eval( exec( open( __import__\ns = "compile( globals( builtins"\nmodel = None\n# model.eval()\nx = torch.compile\n`;
    expect(staticCheck(benign)).toEqual([]);
    expect(staticCheck(`${baseline}\ny = q.open(1) if False else 0\n`)).toEqual([]);
  });

  test("template-rendered kernels pass Tier 1 for every memory layout", () => {
    for (const memory of ["pointer", "block_ptr", "tma"] as const) {
      for (const persistent of [false, true]) {
        expect(staticCheck(paradigmShift({ memory, persistent }))).toEqual([]);
      }
    }
  });

  test("ignores brackets inside strings and comments", () => {
    expect(staticCheck(`${baseline}\n# ) ] }\ns = "(["\nt = '''\n)]\n'''\n`)).toEqual([]);
  });

  test("rejects oversized sources", () => {
    expect(staticCheck(baseline + "#".repeat(70 * 1024))[0]).toContain("exceeds");
  });
});

describe("feature extraction and hardware model", () => {
  test("baseline features", () => {
    expect(extractFeatures(baseline)).toEqual({
      blockM: 64,
      blockN: 32,
      numWarps: 4,
      numStages: 1,
      onlineSoftmax: true,
      fp32Accumulator: true,
      fp8: false,
      exp2: false,
      exp2Rescaled: false,
      blockPtr: false,
      tensorDescriptor: false,
      warpSpecialize: false,
      persistent: false,
    });
  });

  test("structural features are detected inside the launched kernel", () => {
    expect(extractFeatures(paradigmShift())).toMatchObject({ exp2: true, exp2Rescaled: true, tensorDescriptor: true, warpSpecialize: true, persistent: true });
    expect(extractFeatures(paradigmShift({ memory: "block_ptr" }))).toMatchObject({ blockPtr: true, tensorDescriptor: false });
  });

  test("template config round-trips through feature detection", () => {
    for (const memory of ["pointer", "block_ptr", "tma"] as const) {
      const config: KernelConfig = { ...PARADIGM, memory };
      expect(configFromSource(renderKernel(config))).toEqual(config);
    }
  });

  describe("dead code earns nothing", () => {
    const evaluate = (src: string) => evaluator.evaluate(src);
    const indent = (lines: string[]) => lines.map((l) => `    ${l}`).join("\n");

    test.each([
      ["an uncalled plain function", `\ndef _markers(Q, N, D, qk):\n${indent(DEAD_CODE_MARKERS)}\n        pass\n`],
      ["an uncalled @triton.jit function", `\n@triton.jit\ndef _ghost_kernel(Q, N, D, qk):\n${indent(DEAD_CODE_MARKERS)}\n        pass\n`],
      ["a jit kernel launched only from an uncalled host function", `\n@triton.jit\ndef _ghost_kernel(Q, N, D, qk):\n${indent(DEAD_CODE_MARKERS)}\n        pass\n\n\ndef _never_called(q):\n    _ghost_kernel[(1,)](q, 1, 1, 1)\n`],
      ["comments", `\n${DEAD_CODE_MARKERS.map((l) => `# ${l.trim()}`).join("\n")}\n`],
      ["a docstring", `\n_doc = """\n${DEAD_CODE_MARKERS.join("\n")}\n"""\n`],
    ])("markers in %s give no speedup", async (_label, extra) => {
      const gamed = tune(baseline, { m: 128, n: 64, warps: 8, stages: 3 }) + extra;
      const honest = tune(baseline, { m: 128, n: 64, warps: 8, stages: 3 });
      expect(staticCheck(gamed)).toEqual([]);
      expect(extractFeatures(gamed)).toEqual(extractFeatures(honest));
      const [g, h] = [await evaluate(gamed), await evaluate(honest)];
      expect(g.ok).toBe(true);
      expect(g.metrics!.score!).toBeLessThan(100);
      expect(Math.abs(g.metrics!.score! - h.metrics!.score!) / h.metrics!.score!).toBeLessThan(0.02);
    });

    test("a jit helper called from the launched kernel does count", () => {
      const helper = `\n\n@triton.jit\ndef _scale(x):\n    return tl.exp2(x * 1.44269504)\n`;
      const src = baseline.replace("p = tl.exp(qk - m_ij[:, None])", "p = _scale(qk - m_ij[:, None])") + helper;
      expect(staticCheck(src)).toEqual([]);
      expect(extractFeatures(src)).toMatchObject({ exp2: true, exp2Rescaled: true });
    });

    test("a module-level LOG2E that the kernel never uses does not count as rescaled", () => {
      const src = `LOG2E = 1.44269504\n${baseline.replaceAll("tl.exp(", "tl.exp2(")}`;
      expect(extractFeatures(src)).toMatchObject({ exp2: true, exp2Rescaled: false });
    });
  });

  test("shared memory budget", () => {
    const f = extractFeatures(tune(baseline, { m: 128, n: 128, stages: 3 }));
    expect(sharedMemBytes(f, 128)).toBe((128 * 128 * 2 + 2 * 128 * 128 * 2) * 3);
    expect(compileIssues(f, spec)[0]).toContain("OutOfResources");
  });

  test("baseline maps exactly to the PyTorch reference throughput", () => {
    expect(evaluator.trueTflops(extractFeatures(baseline))).toBeCloseTo(spec.baselineTflops, 6);
  });

  test("parameter tuning alone plateaus well below target (local maximum)", () => {
    let best = 0;
    for (const m of [16, 32, 64, 128, 256])
      for (const n of [16, 32, 64, 128, 256])
        for (const warps of [1, 2, 4, 8, 16])
          for (const stages of [1, 2, 3, 4, 5, 6]) {
            const f = extractFeatures(tune(baseline, { m, n, warps, stages }));
            if (compileIssues(f, spec).length === 0) best = Math.max(best, evaluator.trueTflops(f));
          }
    expect(best).toBeGreaterThan(spec.baselineTflops * 1.5);
    expect(best).toBeLessThan(100);
  });

  test("paradigm shift reaches the 140+ TFLOPS target with zero error", async () => {
    const result = await evaluator.evaluate(paradigmShift());
    expect(result.ok).toBe(true);
    expect(result.metrics?.tier).toBe(3);
    expect(result.metrics?.score).toBeGreaterThanOrEqual(spec.targetTflops);
    expect(result.metrics?.maxAbsError).toBe(0);
  });

  test("memory layouts are ordered pointer < block_ptr < tma", () => {
    const t = (memory: KernelConfig["memory"]) => evaluator.trueTflops(extractFeatures(paradigmShift({ memory })));
    expect(t("pointer")).toBeLessThan(t("block_ptr"));
    expect(t("block_ptr")).toBeLessThan(t("tma"));
  });
});

describe("Evaluator tiers", () => {
  test("baseline passes all three tiers near reference throughput", async () => {
    const r = await evaluator.evaluate(baseline);
    expect(r.ok).toBe(true);
    expect(r.failure).toBeUndefined();
    expect(r.tiers.map((t) => t.tier)).toEqual([1, 2, 3]);
    expect(r.metrics?.tier).toBe(3);
    expect(Math.abs(r.metrics!.score! - spec.baselineTflops) / spec.baselineTflops).toBeLessThanOrEqual(0.005);
    expect(r.metrics?.latencyMs).toBeGreaterThan(0);
    expect(r.costGpuSeconds).toBeCloseTo(TIER_COST_GPU_SECONDS[1] + TIER_COST_GPU_SECONDS[2] + TIER_COST_GPU_SECONDS[3]);
    expect(r.savedGpuSeconds).toBe(0);
  });

  test("is deterministic per source and seed", async () => {
    const a = await evaluator.evaluate(baseline);
    const b = await evaluator.evaluate(baseline);
    const c = await new Evaluator(spec, { seed: 42 }).evaluate(baseline);
    expect(a.metrics?.score).toBe(b.metrics?.score);
    expect(c.metrics?.score).not.toBe(a.metrics?.score);
  });

  test("Tier 1 failure prunes before any GPU time is spent", async () => {
    const r = await evaluator.evaluate(`import subprocess\n${baseline}`);
    expect(r.ok).toBe(false);
    expect(r.metrics).toBeUndefined();
    expect(r.features).toBeNull();
    expect(r.failure).toMatchObject({ kind: "syntax", tier: 1 });
    expect(r.tiers).toHaveLength(1);
    expect(r.savedGpuSeconds).toBe(TIER_COST_GPU_SECONDS[2] + TIER_COST_GPU_SECONDS[3]);
  });

  test("shared memory overflow fails Tier 2 as a compile error", async () => {
    const r = await evaluator.evaluate(tune(baseline, { m: 128, n: 128, stages: 3 }));
    expect(r.failure).toMatchObject({ kind: "compile", tier: 2 });
    expect(r.metrics?.tier).toBe(1);
    expect(r.metrics?.score).toBeUndefined();
    expect(r.savedGpuSeconds).toBe(TIER_COST_GPU_SECONDS[3]);
  });

  test("warp specialization without pipelining fails to compile", async () => {
    const r = await evaluator.evaluate(paradigmShift({ numStages: 1 }));
    expect(r.failure?.kind).toBe("compile");
    expect(r.failure?.summary).toContain("warp_specialize");
  });

  test.each([
    ["fp16 accumulator", (s: string) => s.replace("acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)", "acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float16)")],
    ["exp2 without log2(e) rescale", (s: string) => withExp2(s, false)],
    ["fp8 operands", (s: string) => s.replace("p.to(tl.float16)", "p.to(tl.float8e4nv)")],
    ["no running max", (s: string) => s.replace("m_ij = tl.maximum(m_i, tl.max(qk, 1))", "m_ij = tl.max(qk, 1)")],
  ])("%s fails the numerical check at Tier 2", async (_label, mutate) => {
    const r = await evaluator.evaluate(mutate(baseline));
    expect(r.ok).toBe(false);
    expect(r.failure).toMatchObject({ kind: "numerical", tier: 2 });
    expect(r.tiers.at(-1)?.maxAbsError).toBeGreaterThan(0);
  });

  test("correct exp2 rewrite is faster than baseline", async () => {
    const r = await evaluator.evaluate(withExp2(baseline));
    expect(r.ok).toBe(true);
    expect(r.metrics!.score!).toBeGreaterThan(spec.baselineTflops);
  });

  test("Tier 2 estimate below promotion bar is cut as a regression", async () => {
    const r = await evaluator.evaluate(baseline, { promotionScore: 1000 });
    expect(r.ok).toBe(false);
    expect(r.failure).toMatchObject({ kind: "regression", tier: 2 });
    expect(r.metrics?.tier).toBe(2);
    expect(r.metrics?.score).toBeGreaterThan(0);
    expect(r.tiers).toHaveLength(2);
    expect(r.savedGpuSeconds).toBe(TIER_COST_GPU_SECONDS[3]);
  });

  test("maxTier limits fidelity", async () => {
    const t1 = await evaluator.evaluate(baseline, { maxTier: 1 });
    expect(t1.ok).toBe(true);
    expect(t1.metrics?.tier).toBe(1);
    expect(t1.savedGpuSeconds).toBe(0);
    const t2 = await evaluator.evaluate(baseline, { maxTier: 2 });
    expect(t2.tiers.map((t) => t.tier)).toEqual([1, 2]);
    expect(t2.metrics?.tier).toBe(2);
  });

  test("failure summaries are capped", async () => {
    const noisy = Array.from({ length: 50 }, (_, i) => `import os${i}\n(`).join("") + "import os\n";
    const r = await evaluator.evaluate(noisy);
    expect(r.failure!.summary.length).toBeLessThanOrEqual(MAX_FAILURE_SUMMARY_CHARS);
  });

  test("wallMsPerGpuSecond paces the simulation", async () => {
    const paced = new Evaluator(spec, { wallMsPerGpuSecond: 10 });
    const r = await paced.evaluate(baseline, { maxTier: 2 });
    expect(r.metrics!.evalWallMs).toBeGreaterThanOrEqual(14);
  });

  test("rejects a benchmark whose baseline fails Tier 1", () => {
    expect(() => new Evaluator({ ...spec, baselineSource: "print(1)" })).toThrow("fails Tier 1");
  });
});

describe("KernelRunner backend", () => {
  const fakeRunner = (outcome: (tier: 2 | 3) => RunOutcome) => {
    const requests: RunRequest[] = [];
    const runner: KernelRunner = {
      kind: "gpu",
      run: async (req) => {
        requests.push(req);
        return outcome(req.tier);
      },
    };
    return { runner, requests };
  };

  test("defaults to the simulated backend", () => {
    expect(evaluator.backend).toBe("simulated");
  });

  test("delegates Tiers 2 and 3 to the runner and uses measured cost", async () => {
    const { runner, requests } = fakeRunner((tier) => ({ issues: [], score: tier === 2 ? 50 : 52, latencyMs: 3, maxAbsError: 1e-3, costGpuSeconds: 0.25 }));
    const gpu = new Evaluator(spec, { runner, wallMsPerGpuSecond: 1000 });
    const started = performance.now();
    const r = await gpu.evaluate(baseline);
    expect(performance.now() - started).toBeLessThan(500);
    expect(gpu.backend).toBe("gpu");
    expect(requests.map((q) => q.tier)).toEqual([2, 3]);
    expect(requests[0]!.features.blockM).toBe(64);
    expect(r.ok).toBe(true);
    expect(r.metrics).toMatchObject({ tier: 3, score: 52, maxAbsError: 1e-3 });
    expect(r.costGpuSeconds).toBeCloseTo(TIER_COST_GPU_SECONDS[1] + 0.5);
  });

  test("Tier 1 still gates the runner", async () => {
    const { runner, requests } = fakeRunner(() => ({ issues: [] }));
    const r = await new Evaluator(spec, { runner }).evaluate(`import socket\n${baseline}`);
    expect(r.failure?.kind).toBe("syntax");
    expect(requests).toHaveLength(0);
  });

  test.each([
    ["runtime: illegal memory access", "runtime"],
    ["timeout: tier 2 exceeded 60000 ms", "timeout"],
    ["numerical: max abs error 0.5", "numerical"],
    ["compile: triton.CompilationError", "compile"],
    ["unprefixed message", "compile"],
  ])("classifies runner issue %p as %s", async (issue, kind) => {
    const { runner } = fakeRunner(() => ({ issues: [issue] }));
    const r = await new Evaluator(spec, { runner }).evaluate(baseline);
    expect(r.failure).toMatchObject({ kind, tier: 2 });
  });

  test("Tier 3 runner failure is reported at Tier 3", async () => {
    const { runner } = fakeRunner((tier) => (tier === 2 ? { issues: [], score: 60 } : { issues: ["runtime: OOM"] }));
    const r = await new Evaluator(spec, { runner }).evaluate(baseline);
    expect(r.failure).toMatchObject({ kind: "runtime", tier: 3 });
    expect(r.metrics?.tier).toBe(2);
  });
});

describe("parseHarnessOutput", () => {
  test("reads the last JSON line", () => {
    const out = 'compiling...\n{"issues": ["old"]}\nwarmup\n{"issues": [], "score": 88.5, "latency_ms": 12.1, "max_abs_error": 0}\n';
    expect(parseHarnessOutput(out)).toEqual({ issues: [], score: 88.5, latencyMs: 12.1, maxAbsError: 0 });
  });

  test("handles CRLF and drops non-finite numbers", () => {
    expect(parseHarnessOutput('{"issues": ["runtime: x"], "score": "fast"}\r\n')).toEqual({
      issues: ["runtime: x"],
      score: undefined,
      latencyMs: undefined,
      maxAbsError: undefined,
    });
  });

  test.each([["no json", "Traceback ...\n"], ["bad json", "{oops\n"], ["missing issues", '{"score": 1}\n'], ["empty", ""]])("returns undefined for %s", (_l, out) => {
    expect(parseHarnessOutput(out)).toBeUndefined();
  });
});
