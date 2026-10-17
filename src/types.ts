export interface ProblemShape {
  batch: number;
  heads: number;
  seqLen: number;
  headDim: number;
  dtype: "float16" | "bfloat16";
  causal: boolean;
}

export interface HardwareSpec {
  name: string;
  sharedMemBytes: number;
}

export interface BenchmarkSpec {
  name: string;
  description: string;
  problem: ProblemShape;
  hardware: HardwareSpec;
  baselineTflops: number;
  targetTflops: number;
  /** Max absolute error allowed against the reference output. */
  errorTolerance: number;
  baselineSource: string;
  referenceSource: string;
}
