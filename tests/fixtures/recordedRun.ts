import { Mutator } from "../../src/agents/mutator";
import { Reflector } from "../../src/agents/reflector";
import { loadBenchmark } from "../../src/harness/benchmark";
import { Evaluator } from "../../src/harness/evaluator";
import type { GraphEvent } from "../../src/harness/graphStore";
import { InMemoryGraphStore } from "../../src/harness/inMemoryGraphStore";
import { ParadigmShiftEngine, PlateauDetector } from "../../src/harness/plateauDetector";
import { evolve, type EvolveEvent, type EvolveSummary } from "../../src/harness/runner";

export interface RecordedRun {
  /** Interleaved in arrival order, as a live dashboard would see them. */
  timeline: Array<{ kind: "evolve"; event: EvolveEvent } | { kind: "graph"; event: GraphEvent }>;
  evolveEvents: EvolveEvent[];
  summary: EvolveSummary;
  store: InMemoryGraphStore;
}

/** One deterministic offline run (simulated backend, no Kev, no network). */
export async function recordRun(maxCycles = 40, seed = 0): Promise<RecordedRun> {
  const spec = await loadBenchmark("triton_flash_attention");
  const store = new InMemoryGraphStore({ now: () => 0 });
  const known = new Set<string>();
  const timeline: RecordedRun["timeline"] = [];
  const evolveEvents: EvolveEvent[] = [];
  store.subscribe((event) => {
    if (event.type === "node_added") known.add(event.node.source);
    timeline.push({ kind: "graph", event });
  });
  const summary = await evolve(
    {
      benchmark: spec, store, evaluator: new Evaluator(spec, { seed }), reflector: new Reflector(),
      mutator: new Mutator({ isKnown: (s) => known.has(s) }), plateau: new PlateauDetector(),
      shiftEngine: new ParadigmShiftEngine({ store, benchmark: spec, kev: null }), kev: null,
    },
    {
      seed,
      maxCycles,
      onEvent: (event) => {
        evolveEvents.push(event);
        timeline.push({ kind: "evolve", event });
      },
    },
  );
  return { timeline, evolveEvents, summary, store };
}
