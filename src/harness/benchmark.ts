import { join } from "node:path";
import type { BenchmarkSpec } from "../types";

export const BENCHMARKS_DIR = join(import.meta.dir, "..", "..", "benchmarks");

export async function loadBenchmark(name: string, root = BENCHMARKS_DIR): Promise<BenchmarkSpec> {
  if (!/^[\w-]+$/.test(name)) throw new Error(`Invalid benchmark name: ${name}`);
  const dir = join(root, name);
  const manifest = Bun.file(join(dir, "benchmark.json"));
  if (!(await manifest.exists())) throw new Error(`Unknown benchmark: ${name}`);
  const { baseline, reference, ...meta } = await manifest.json();
  const read = async (file: string) => (await Bun.file(join(dir, file)).text()).replace(/\r\n/g, "\n");
  return { ...meta, baselineSource: await read(baseline), referenceSource: await read(reference) };
}
