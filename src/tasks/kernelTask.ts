/**
 * The original task: optimize a Triton flash-attention forward kernel.
 * Wraps the existing evaluator, mutator, kernel template, and Kev wording.
 * The Kev text it produces is byte-identical to the pre-task version, so
 * recorded demo caches keep replaying.
 */

import { configFromSource, describeConfig } from "../agents/kernelTemplate";
import { STRUCTURAL_STRATEGIES, type Mutator } from "../agents/mutator";
import { TIER_COST_GPU_SECONDS, staticCheck, type Evaluator } from "../harness/evaluator";
import { KERNEL_OBJECTIVE, KERNEL_SHIFT_INSTRUCTIONS, pruneQuestion } from "../prompts/kevQuestions";
import type { BenchmarkSpec } from "../types";
import type { ResearchTask } from "./types";

export const describeKernel = (source: string): string => describeConfig(configFromSource(source));

/**
 * `evaluator` and `mutator` are optional so the paradigm-shift engine can use
 * the structural menu without a full evaluation stack.
 */
export function createKernelTask(benchmark: BenchmarkSpec, evaluator?: Evaluator, mutator?: Mutator): ResearchTask {
  const need = <T>(x: T | undefined, what: string): T => {
    if (!x) throw new Error(`kernel task created without a ${what}`);
    return x;
  };
  const backend = evaluator?.backend ?? "simulated";
  return {
    id: benchmark.name,
    name: "Triton flash-attention forward kernel",
    description: benchmark.description,
    objective: KERNEL_OBJECTIVE,
    noun: "kernel",
    metric: { name: "throughput", unit: "TFLOPS" },
    baselineScore: benchmark.baselineTflops,
    targetScore: benchmark.targetTflops,
    baselineSource: benchmark.baselineSource,
    seedRationale: "Reference Triton kernel",
    backend,
    backendLabel: backend === "gpu" ? "GPU" : "SIMULATED H100",
    tierCosts: TIER_COST_GPU_SECONDS,
    describe: describeKernel,
    staticCheck,
    evaluate: (source, o) => need(evaluator, "evaluator").evaluate(source, { promotionScore: o.promotionScore }),
    proposeLocal: (parent, lineage, count, rng) => need(mutator, "mutator").proposeLocal(parent, lineage, count, rng),
    structuralOptions(source) {
      const config = configFromSource(source);
      return STRUCTURAL_STRATEGIES.filter((s) => s.apply(config) !== null).map(({ id, label, description }) => ({ id, label, description }));
    },
    appliedStructural(source) {
      const config = configFromSource(source);
      return STRUCTURAL_STRATEGIES.filter((s) => s.apply(config) === null).map((s) => s.label);
    },
    proposeShift: (parent, lineage, optionId, count, rationale) => need(mutator, "mutator").proposeParadigmShift(parent, lineage, optionId, count, rationale),
    kev: {
      pruneInstructions: (label, rationale, diff) => pruneQuestion(label, rationale, diff).instructions!,
      shiftInstructions: KERNEL_SHIFT_INSTRUCTIONS,
    },
  };
}
