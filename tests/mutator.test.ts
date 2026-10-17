import { describe, expect, test } from "bun:test";
import { configFromSource, renderKernel, type KernelConfig } from "../src/agents/kernelTemplate";
import { LlmGenerator, MICRO_STRATEGIES, Mutator, STRUCTURAL_STRATEGIES, TemplateGenerator, lineDiff, strategyById } from "../src/agents/mutator";
import { loadBenchmark } from "../src/harness/benchmark";
import { staticCheck } from "../src/harness/evaluator";
import { InMemoryGraphStore } from "../src/harness/inMemoryGraphStore";
import { createRng } from "../src/harness/rng";
import { KERNEL_CONTRACT, extractCodeBlock } from "../src/prompts/mutationPrompts";

const spec = await loadBenchmark("triton_flash_attention");
const BASE: KernelConfig = { blockM: 64, blockN: 32, numWarps: 4, numStages: 1, exp2: false, memory: "pointer", warpSpecialize: false, persistent: false };

function parentNode(source = spec.baselineSource) {
  const store = new InMemoryGraphStore({ now: () => 0 });
  const node = store.addNode({ parentIds: [], source, mutation: { kind: "seed", strategy: "baseline", rationale: "" } });
  return { store, node, lineage: store.lineageContext(node.id) };
}

const micro = (id: string) => MICRO_STRATEGIES.find((s) => s.id === id)!;

describe("micro strategies", () => {
  test.each<[string, Partial<KernelConfig>]>([
    ["tile_m_up", { blockM: 256 }],
    ["tile_m_down", { blockM: 16 }],
    ["tile_n_up", { blockN: 256 }],
    ["tile_n_down", { blockN: 16 }],
    ["warps_up", { numWarps: 16 }],
    ["warps_down", { numWarps: 1 }],
    ["stages_up", { numStages: 6 }],
    ["stages_down", { numStages: 1 }],
    ["stages_down", { numStages: 2, warpSpecialize: true }],
    ["exp2_softmax", { exp2: true }],
  ])("%s returns null at its bound %o", (id, bound) => {
    expect(micro(id).apply({ ...BASE, ...bound })).toBeNull();
  });

  test.each<[string, Partial<KernelConfig>]>([
    ["tile_m_up", { blockM: 128 }],
    ["tile_m_down", { blockM: 32 }],
    ["tile_n_up", { blockN: 64 }],
    ["tile_n_down", { blockN: 16 }],
    ["warps_up", { numWarps: 8 }],
    ["warps_down", { numWarps: 2 }],
    ["stages_up", { numStages: 2 }],
    ["exp2_softmax", { exp2: true }],
  ])("%s changes exactly one knob from the baseline", (id, expected) => {
    expect(micro(id).apply(BASE)).toEqual({ ...BASE, ...expected });
  });

  test("stages_down works above the warp-specialization floor", () => {
    expect(micro("stages_down").apply({ ...BASE, numStages: 3, warpSpecialize: true })).toMatchObject({ numStages: 2 });
  });

  test("off-grid values are treated as not applicable", () => {
    expect(micro("tile_m_up").apply({ ...BASE, blockM: 96 })).toBeNull();
  });

  test("strategyById finds micro and structural strategies", () => {
    expect(strategyById("warps_up")?.kind).toBe("local");
    expect(strategyById("tma_descriptors")?.kind).toBe("paradigm_shift");
    expect(strategyById("nope")).toBeUndefined();
  });
});

describe("structural strategies", () => {
  test("priority order is block_ptr, tma, warp specialization, persistent", () => {
    expect(STRUCTURAL_STRATEGIES.map((s) => s.id)).toEqual(["block_ptr_layout", "tma_descriptors", "warp_specialization", "persistent_kernel"]);
  });

  test("each returns null once applied; warp specialization forces at least 2 stages", () => {
    const [blockPtr, tma, ws, persistent] = STRUCTURAL_STRATEGIES;
    expect(blockPtr!.apply({ ...BASE, memory: "block_ptr" })).toBeNull();
    expect(blockPtr!.apply({ ...BASE, memory: "tma" })).toBeNull();
    expect(tma!.apply({ ...BASE, memory: "tma" })).toBeNull();
    expect(tma!.apply({ ...BASE, memory: "block_ptr" })).toMatchObject({ memory: "tma" });
    expect(ws!.apply({ ...BASE, warpSpecialize: true })).toBeNull();
    expect(ws!.apply(BASE)).toMatchObject({ warpSpecialize: true, numStages: 2 });
    expect(persistent!.apply({ ...BASE, persistent: true })).toBeNull();
  });
});

describe("Mutator.proposeLocal", () => {
  test("is deterministic for a fixed seed", async () => {
    const { node, lineage } = parentNode();
    const a = await new Mutator().proposeLocal(node, lineage, 6, createRng(42));
    const b = await new Mutator().proposeLocal(node, lineage, 6, createRng(42));
    expect(a.map((p) => [p.strategy.id, p.input.source])).toEqual(b.map((p) => [p.strategy.id, p.input.source]));
    const c = await new Mutator().proposeLocal(node, lineage, 6, createRng(7));
    expect(c.map((p) => p.strategy.id)).not.toEqual(a.map((p) => p.strategy.id));
  });

  test("respects count and produces well-formed local proposals", async () => {
    const { node, lineage } = parentNode();
    for (const count of [0, 1, 3, 6]) {
      const proposals = await new Mutator().proposeLocal(node, lineage, count, createRng(1));
      expect(proposals).toHaveLength(count);
    }
    const proposals = await new Mutator().proposeLocal(node, lineage, 50, createRng(1));
    expect(proposals.length).toBeGreaterThan(6);
    const sources = new Set<string>();
    for (const p of proposals) {
      expect(p.input.source).not.toBe(node.source);
      expect(p.input.mutation.kind).toBe("local");
      expect(p.input.parentIds).toEqual([node.id]);
      expect(p.input.diff!.length).toBeGreaterThan(0);
      expect(p.input.diff).toMatch(/^[-+]/m);
      expect(p.input.mutation.rationale).toContain("64x32 w4 s1 ->");
      expect(staticCheck(p.input.source)).toEqual([]);
      expect(configFromSource(p.input.source)).toEqual(p.config);
      sources.add(p.input.source);
    }
    expect(sources.size).toBe(proposals.length);
  });

  test("skips sources reported by isKnown", async () => {
    const { node, lineage } = parentNode();
    const first = await new Mutator().proposeLocal(node, lineage, 4, createRng(3));
    const known = new Set(first.map((p) => p.input.source));
    const second = await new Mutator({ isKnown: (s) => known.has(s) }).proposeLocal(node, lineage, 4, createRng(3));
    expect(second.length).toBeGreaterThan(0);
    for (const p of second) expect(known.has(p.input.source)).toBe(false);
  });

  test("never re-proposes the parent config even when a pair cancels out", async () => {
    const { node, lineage } = parentNode();
    for (let seed = 0; seed < 20; seed++) {
      const proposals = await new Mutator().proposeLocal(node, lineage, 50, createRng(seed));
      for (const p of proposals) expect(p.input.source).not.toBe(node.source);
    }
  });
});

describe("Mutator.proposeParadigmShift", () => {
  test("throws on an unknown strategy id", async () => {
    const { node, lineage } = parentNode();
    await expect(new Mutator().proposeParadigmShift(node, lineage, "warps_up", 4)).rejects.toThrow("unknown structural strategy warps_up");
  });

  test("returns [] when the rewrite is already applied", async () => {
    const { node, lineage } = parentNode(renderKernel({ ...BASE, persistent: true }));
    expect(await new Mutator().proposeParadigmShift(node, lineage, "persistent_kernel", 4)).toEqual([]);
  });

  test.each(STRUCTURAL_STRATEGIES.map((s) => s.id))("%s applies its structural flag to every proposal", async (id) => {
    const { node, lineage } = parentNode();
    const strategy = STRUCTURAL_STRATEGIES.find((s) => s.id === id)!;
    const proposals = await new Mutator().proposeParadigmShift(node, lineage, id, 10, "kev-27b said so");
    expect(proposals.length).toBeGreaterThan(1);
    expect(proposals.length).toBeLessThanOrEqual(5);
    const expected = strategy.apply(BASE)!;
    for (const p of proposals) {
      expect(p.input.mutation.kind).toBe("paradigm_shift");
      expect(p.config.memory).toBe(expected.memory);
      expect(p.config.warpSpecialize).toBe(expected.warpSpecialize);
      expect(p.config.persistent).toBe(expected.persistent);
      expect(configFromSource(p.input.source)).toEqual(p.config);
      expect(staticCheck(p.input.source)).toEqual([]);
    }
    expect(new Set(proposals.map((p) => p.input.source)).size).toBe(proposals.length);
  });

  test("respects count", async () => {
    const { node, lineage } = parentNode();
    expect(await new Mutator().proposeParadigmShift(node, lineage, "tma_descriptors", 2)).toHaveLength(2);
  });
});

describe("LlmGenerator", () => {
  test("renders a prompt with the contract and lineage, and extracts the code block", async () => {
    const prompts: string[] = [];
    const target = renderKernel({ ...BASE, numWarps: 8 });
    const llm = new LlmGenerator(async (prompt) => {
      prompts.push(prompt);
      return `Sure, here it is:\n\`\`\`python\n${target}\n\`\`\`\nDone.`;
    });
    const { node, lineage } = parentNode();
    const proposals = await new Mutator({ generator: llm }).proposeLocal(node, lineage, 1, createRng(0));
    expect(proposals).toHaveLength(1);
    expect(proposals[0]!.input.source).toBe(target);
    const prompt = prompts[0]!;
    expect(prompt).toContain("Apply exactly one micro-optimization");
    expect(prompt).toContain("Current kernel config: 64x32 w4 s1");
    for (const rule of KERNEL_CONTRACT) expect(prompt).toContain(rule);
    expect(prompt).toContain(node.source);
    expect(prompt).toContain("gen 0 baseline");
  });

  test("structural proposals use the structural template and pass the directive", async () => {
    const prompts: string[] = [];
    const llm = new LlmGenerator(async (prompt) => {
      prompts.push(prompt);
      return "no code here";
    });
    const { node, lineage } = parentNode();
    const proposals = await new Mutator({ generator: llm }).proposeParadigmShift(node, lineage, "tma_descriptors", 2, "plateau at 89 TFLOPS");
    expect(proposals).toEqual([]);
    expect(prompts.length).toBeGreaterThan(0);
    expect(prompts[0]).toContain('structural rewrite ("Paradigm Shift")');
    expect(prompts[0]).toContain("Why now: plateau at 89 TFLOPS");
    expect(prompts[0]).toContain("Directive: TMA tensor descriptors");
  });

  test("the template generator is the default", () => {
    expect(new Mutator().generator).toBeInstanceOf(TemplateGenerator);
    expect(new Mutator().generator.kind).toBe("template");
  });
});

describe("extractCodeBlock", () => {
  test.each([
    ["python fence", "x\n```python\nprint(1)\n```\ny", "print(1)\n"],
    ["py fence", "```py\na = 1\n\n```", "a = 1\n"],
    ["bare fence", "```\nb = 2\n```", "b = 2\n"],
    ["first block wins", "```python\nfirst\n```\n```python\nsecond\n```", "first\n"],
  ])("%s", (_label, reply, expected) => {
    expect(extractCodeBlock(reply)).toBe(expected);
  });

  test("returns undefined without a fenced block", () => {
    expect(extractCodeBlock("def f(): pass")).toBeUndefined();
  });
});

describe("lineDiff", () => {
  test("a one-line edit", () => {
    expect(lineDiff("a\nb\nc", "a\nX\nc")).toBe("-b\n+X");
  });

  test("insertions, deletions, and identical inputs", () => {
    expect(lineDiff("a\nc", "a\nb\nc")).toBe("+b");
    expect(lineDiff("a\nb\nc", "a\nc")).toBe("-b");
    expect(lineDiff("same\ntext", "same\ntext")).toBe("");
    expect(lineDiff("", "x")).toBe("-\n+x");
  });

  test("diff between rendered kernels only touches changed constants", () => {
    const d = lineDiff(renderKernel(BASE), renderKernel({ ...BASE, numWarps: 8 }));
    expect(d).toBe("-NUM_WARPS = 4\n+NUM_WARPS = 8");
  });
});
