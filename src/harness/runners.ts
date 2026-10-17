/**
 * Kernel runners: the execution backend behind evaluator Tiers 2 and 3.
 *
 * The Evaluator owns tier orchestration and Tier 1 static checks. A
 * KernelRunner owns "compile, run, compare against reference, time it".
 * Without a runner the Evaluator uses its built-in simulated H100 model.
 * PythonGpuRunner executes candidates on a real CUDA device through
 * benchmarks/<name>/harness.py.
 *
 * Sandbox notes for PythonGpuRunner:
 * - Each run gets a fresh temp directory as cwd; the candidate is written there.
 * - The child gets a minimal environment. Secrets such as KEV_API_KEY are never
 *   forwarded.
 * - Python runs with -E -P (ignores PYTHON* env vars; no script dir on sys.path).
 * - Hard wall-clock timeout per tier; the process is killed on expiry.
 * This is process isolation, not a security boundary. Tier 1 rejects obvious
 * dangerous imports, but untrusted kernels should run inside a container.
 */

import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ProblemShape, BenchmarkSpec } from "../types";
import type { KernelFeatures } from "./evaluator";

export type HardwareTier = 2 | 3;

export interface RunRequest {
  tier: HardwareTier;
  source: string;
  features: KernelFeatures;
  spec: BenchmarkSpec;
}

/**
 * Issues use a "<kind>: message" prefix ("compile:", "runtime:", "numerical:",
 * "timeout:") so the evaluator can classify failures.
 */
export interface RunOutcome {
  issues: string[];
  score?: number;
  latencyMs?: number;
  maxAbsError?: number;
  /** Measured device time. When absent the evaluator uses its tier cost table. */
  costGpuSeconds?: number;
}

export interface KernelRunner {
  /** "simulated" numbers must be labeled as such anywhere they are displayed. */
  readonly kind: "simulated" | "gpu";
  run(request: RunRequest): Promise<RunOutcome>;
}

// ---------------------------------------------------------------------------
// Real hardware runner
// ---------------------------------------------------------------------------

export interface PythonGpuRunnerOptions {
  /** Benchmark directory containing harness.py and the reference module. */
  benchmarkDir: string;
  python?: string;
  /** Per-tier timeout. Defaults: Tier 2 60 s, Tier 3 300 s. */
  timeoutMs?: Partial<Record<HardwareTier, number>>;
  /**
   * Max abs error allowed on hardware. Real fp16 kernels are rarely bit-exact
   * with the PyTorch reference, so this defaults to spec.errorTolerance only
   * when that is nonzero, else 2e-2.
   */
  errorTolerance?: number;
  /** Tier 2 runs this reduced problem. Defaults to batch 1, 4 heads, seqLen 1024. */
  miniProblem?: Partial<ProblemShape>;
}

const DEFAULT_TIMEOUT_MS: Readonly<Record<HardwareTier, number>> = { 2: 60_000, 3: 300_000 };
const FORWARDED_ENV = ["PATH", "SYSTEMROOT", "TEMP", "TMP", "CUDA_PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"];

function minimalEnv(): Record<string, string> {
  const env: Record<string, string> = { PYTHONDONTWRITEBYTECODE: "1" };
  for (const key of FORWARDED_ENV) {
    const value = process.env[key];
    if (value !== undefined) env[key] = value;
  }
  return env;
}

async function runProcess(cmd: string[], cwd: string, timeoutMs: number): Promise<{ stdout: string; stderr: string; code: number | null; timedOut: boolean }> {
  const proc = Bun.spawn(cmd, { cwd, env: minimalEnv(), stdout: "pipe", stderr: "pipe", stdin: "ignore" });
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    proc.kill();
  }, timeoutMs);
  try {
    const [stdout, stderr, code] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text(), proc.exited]);
    return { stdout, stderr, code, timedOut };
  } finally {
    clearTimeout(timer);
  }
}

/** Parses the last JSON line printed by harness.py. */
export function parseHarnessOutput(stdout: string): RunOutcome | undefined {
  const line = stdout.trim().split(/\r?\n/).reverse().find((l) => l.startsWith("{"));
  if (!line) return undefined;
  try {
    const raw = JSON.parse(line) as Record<string, unknown>;
    if (!Array.isArray(raw.issues)) return undefined;
    const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : undefined);
    return {
      issues: raw.issues.map(String),
      score: num(raw.score),
      latencyMs: num(raw.latency_ms),
      maxAbsError: num(raw.max_abs_error),
    };
  } catch {
    return undefined;
  }
}

export class PythonGpuRunner implements KernelRunner {
  readonly kind = "gpu" as const;
  private readonly python: string;

  constructor(private readonly options: PythonGpuRunnerOptions) {
    this.python = options.python ?? "python";
  }

  async run({ tier, source, spec }: RunRequest): Promise<RunOutcome> {
    const problem: ProblemShape =
      tier === 2 ? { ...spec.problem, batch: 1, heads: Math.min(4, spec.problem.heads), seqLen: Math.min(1024, spec.problem.seqLen), ...this.options.miniProblem } : spec.problem;
    const tolerance = this.options.errorTolerance ?? (spec.errorTolerance > 0 ? spec.errorTolerance : 2e-2);
    const timeoutMs = this.options.timeoutMs?.[tier] ?? DEFAULT_TIMEOUT_MS[tier];

    const workDir = mkdtempSync(join(tmpdir(), "kev-evolve-"));
    try {
      const candidatePath = join(workDir, "candidate.py");
      writeFileSync(candidatePath, source);
      const cmd = [
        this.python, "-E", "-P", join(this.options.benchmarkDir, "harness.py"),
        "--candidate", candidatePath,
        "--reference", join(this.options.benchmarkDir, "reference.py"),
        "--problem", JSON.stringify(problem),
        "--tolerance", String(tolerance),
        "--tier", String(tier),
      ];
      const started = performance.now();
      const res = await runProcess(cmd, workDir, timeoutMs);
      const elapsedSeconds = (performance.now() - started) / 1000;
      if (res.timedOut) return { issues: [`timeout: tier ${tier} exceeded ${timeoutMs} ms`], costGpuSeconds: elapsedSeconds };
      const parsed = parseHarnessOutput(res.stdout);
      if (!parsed) {
        const tail = res.stderr.trim().split(/\r?\n/).slice(-1)[0] ?? "";
        return { issues: [`runtime: harness exited ${res.code} without a result. ${tail}`.trim()], costGpuSeconds: elapsedSeconds };
      }
      return { ...parsed, costGpuSeconds: elapsedSeconds };
    } finally {
      rmSync(workDir, { recursive: true, force: true });
    }
  }
}

/**
 * Returns a PythonGpuRunner when torch, triton, and a CUDA device are all
 * available, else undefined (callers fall back to the simulated model).
 */
export async function detectGpuRunner(options: PythonGpuRunnerOptions): Promise<PythonGpuRunner | undefined> {
  const probe = "import torch, triton; print('CUDA_OK' if torch.cuda.is_available() else 'NO_CUDA')";
  try {
    const res = await runProcess([options.python ?? "python", "-E", "-P", "-c", probe], options.benchmarkDir, 30_000);
    return res.stdout.includes("CUDA_OK") ? new PythonGpuRunner(options) : undefined;
  } catch {
    return undefined;
  }
}
