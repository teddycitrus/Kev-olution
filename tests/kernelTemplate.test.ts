import { describe, expect, test } from "bun:test";
import { configFromSource, describeConfig, renderKernel, sameConfig, type KernelConfig, type MemoryLayout } from "../src/agents/kernelTemplate";
import { loadBenchmark } from "../src/harness/benchmark";
import { Evaluator, extractFeatures, numericalError, staticCheck } from "../src/harness/evaluator";

const spec = await loadBenchmark("triton_flash_attention");
const evaluator = new Evaluator(spec);

const TILES = [
  { blockM: 64, blockN: 32, numWarps: 4, numStages: 1 },
  { blockM: 128, blockN: 64, numWarps: 8, numStages: 3 },
  { blockM: 128, blockN: 128, numWarps: 8, numStages: 2 },
];
const MEMORY: MemoryLayout[] = ["pointer", "block_ptr", "tma"];

const ALL: KernelConfig[] = TILES.flatMap((t) =>
  MEMORY.flatMap((memory) => [false, true].flatMap((warpSpecialize) => [false, true].flatMap((persistent) => [false, true].map((exp2) => ({ ...t, memory, warpSpecialize, persistent, exp2 }))))),
);

describe("renderKernel", () => {
  test("covers every combination (3 tiles x 3 layouts x ws x persistent x exp2)", () => {
    expect(ALL).toHaveLength(72);
  });

  test.each(ALL.map((c) => [describeConfig(c), c] as const))("%s passes Tier 1 and round-trips", (label, config) => {
    const source = renderKernel(config);
    expect(staticCheck(source)).toEqual([]);
    const parsed = configFromSource(source);
    expect(describeConfig(parsed)).toBe(label);
    expect(sameConfig(parsed, config)).toBe(true);
  });

  test("structural code lives inside the kernel, the host entry point is attention()", () => {
    const source = renderKernel({ ...TILES[1]!, memory: "tma", warpSpecialize: true, persistent: true, exp2: true });
    const kernelStart = source.indexOf("def _attn_fwd(");
    const hostStart = source.indexOf("def attention(q, k, v, sm_scale):");
    expect(kernelStart).toBeGreaterThan(0);
    expect(hostStart).toBeGreaterThan(kernelStart);
    const kernel = source.slice(kernelStart, hostStart);
    for (const marker of ["make_tensor_descriptor(", "warp_specialize=True", "tl.num_programs(0)", "tl.exp2(", "1.44269504"]) {
      expect(kernel).toContain(marker);
    }
    expect(source.slice(hostStart)).toContain("triton.set_allocator");
  });

  test("is deterministic", () => {
    const c = ALL[17]!;
    expect(renderKernel(c)).toBe(renderKernel({ ...c }));
  });
});

describe("configFromSource and describeConfig", () => {
  test("the baseline parses as 64x32 w4 s1 with no structural flags", () => {
    const c = configFromSource(spec.baselineSource);
    expect(describeConfig(c)).toBe("64x32 w4 s1");
    expect(c).toEqual({ blockM: 64, blockN: 32, numWarps: 4, numStages: 1, exp2: false, memory: "pointer", warpSpecialize: false, persistent: false });
  });

  test("labels list flags in a fixed order", () => {
    expect(describeConfig({ ...TILES[2]!, exp2: true, memory: "tma", warpSpecialize: true, persistent: true })).toBe("128x128 w8 s2 exp2 tma ws persistent");
    expect(describeConfig({ ...TILES[0]!, exp2: false, memory: "block_ptr", warpSpecialize: false, persistent: false })).toBe("64x32 w4 s1 block_ptr");
  });

  test("exp2 without the log2(e) rescale is not reported as exp2", () => {
    const unscaled = spec.baselineSource.replaceAll("tl.exp(", "tl.exp2(");
    expect(configFromSource(unscaled).exp2).toBe(false);
  });
});

describe("simulator", () => {
  test.each(MEMORY)("exp2 renders with zero numerical error (%s)", async (memory) => {
    const source = renderKernel({ ...TILES[1]!, memory, warpSpecialize: false, persistent: false, exp2: true });
    expect(numericalError(extractFeatures(source))).toBe(0);
    const r = await evaluator.evaluate(source);
    expect(r.ok).toBe(true);
    expect(r.metrics?.maxAbsError).toBe(0);
  });

  test("the fully structural kernel reaches the target", async () => {
    const r = await evaluator.evaluate(renderKernel({ ...TILES[1]!, memory: "tma", warpSpecialize: true, persistent: true, exp2: true }));
    expect(r.metrics!.score!).toBeGreaterThanOrEqual(spec.targetTflops);
  });
});
