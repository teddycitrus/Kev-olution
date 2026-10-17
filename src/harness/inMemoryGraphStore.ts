import {
  MAX_FAILURE_SUMMARY_CHARS,
  type ExecutionMetrics,
  type FailureSummary,
  type FrontierOptions,
  type GraphEvent,
  type GraphEventListener,
  type GraphSnapshot,
  type GraphStore,
  type GraphStoreReplayer,
  type KevVerdict,
  type LineageContext,
  type MutationKind,
  type NewNodeInput,
  type NodeId,
  type NodeStatus,
  type TrajectoryEdge,
  type TrajectoryNode,
} from "./graphStore";

type DistributiveOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;
type EventInput = DistributiveOmit<GraphEvent, "seq" | "at">;

export interface InMemoryGraphStoreOptions {
  now?: () => number;
  idPrefix?: string;
}

const STATUSES: readonly NodeStatus[] = ["pending", "pruned", "evaluated", "failed", "purged"];
const EXCLUDED_FROM_FRONTIER = new Set<NodeStatus>(["pruned", "purged", "failed"]);

const isChampionEligible = (m: ExecutionMetrics | undefined): m is ExecutionMetrics & { score: number } =>
  m !== undefined && typeof m.score === "number" && Number.isFinite(m.score) && m.maxAbsError === 0;

function validateParentCount(kind: MutationKind, count: number): void {
  const ok = kind === "seed" ? count === 0 : kind === "local" ? count === 1 : kind === "crossover" ? count >= 2 : count >= 1;
  if (!ok) throw new Error(`mutation kind "${kind}" cannot have ${count} parent(s)`);
}

export class InMemoryGraphStore implements GraphStore {
  private readonly nodes = new Map<NodeId, TrajectoryNode>();
  private readonly order: NodeId[] = [];
  private readonly childIds = new Map<NodeId, NodeId[]>();
  private readonly pruneReasons = new Map<NodeId, string>();
  private readonly log: GraphEvent[] = [];
  private readonly listeners = new Set<GraphEventListener>();
  private championId?: NodeId;
  private seq = 0;
  private readonly now: () => number;
  private readonly idPrefix: string;

  constructor(options: InMemoryGraphStoreOptions = {}) {
    this.now = options.now ?? Date.now;
    this.idPrefix = options.idPrefix ?? "n";
  }

  // -------------------------------------------------------------------------
  // Writes
  // -------------------------------------------------------------------------

  addNode(input: NewNodeInput): TrajectoryNode {
    const parentIds = [...input.parentIds];
    if (new Set(parentIds).size !== parentIds.length) throw new Error("duplicate parent ids");
    validateParentCount(input.mutation.kind, parentIds.length);
    const parents = parentIds.map((id) => this.require(id));
    const id = `${this.idPrefix}${this.order.length}`;
    if (this.nodes.has(id) || parentIds.includes(id)) throw new Error(`adding ${id} would create a cycle`);
    const node: TrajectoryNode = {
      id,
      parentIds,
      generation: parents.length ? Math.max(...parents.map((p) => p.generation)) + 1 : 0,
      createdAt: this.now(),
      source: input.source,
      ...(input.diff !== undefined && { diff: input.diff }),
      mutation: { ...input.mutation },
      status: "pending",
    };
    this.emit({ type: "node_added", node });
    return this.require(id);
  }

  recordVerdict(id: NodeId, verdict: KevVerdict): void {
    this.require(id);
    if (!(verdict.score >= 0 && verdict.score <= 1)) throw new Error(`Kev score ${verdict.score} outside [0, 1]`);
    this.emit({ type: "node_scored", id, verdict: { ...verdict } });
  }

  markPruned(id: NodeId, reason: string): void {
    this.require(id);
    this.emit({ type: "node_pruned", id, reason });
  }

  recordEvaluation(id: NodeId, metrics: ExecutionMetrics): void {
    const node = this.require(id);
    this.emit({ type: "node_evaluated", id, metrics: { ...metrics } });
    if (node.status === "purged" || !isChampionEligible(metrics)) return;
    const current = this.champion()?.metrics?.score ?? Number.NEGATIVE_INFINITY;
    if (metrics.score > current) {
      this.emit({
        type: "champion_changed",
        id,
        ...(this.championId !== undefined && { previousId: this.championId }),
        score: metrics.score,
      });
    }
  }

  recordFailure(id: NodeId, failure: FailureSummary): void {
    this.require(id);
    const summary =
      failure.summary.length > MAX_FAILURE_SUMMARY_CHARS
        ? `${failure.summary.slice(0, MAX_FAILURE_SUMMARY_CHARS - 3)}...`
        : failure.summary;
    this.emit({ type: "node_failed", id, failure: { ...failure, summary } });
  }

  /**
   * The champion and all of its ancestors (every parent, not only first
   * parents) are never purged, so a Paradigm Shift can branch from the
   * champion. Their descendants are still traversed and purged.
   */
  purgeBranch(rootId: NodeId, reason: string): NodeId[] {
    this.require(rootId);
    const protectedIds = this.ancestorsOf(this.championId);
    const purgedIds: NodeId[] = [];
    const seen = new Set<NodeId>([rootId]);
    const queue = [...(this.childIds.get(rootId) ?? [])];
    while (queue.length) {
      const id = queue.shift()!;
      if (seen.has(id)) continue;
      seen.add(id);
      if (!protectedIds.has(id) && this.require(id).status !== "purged") purgedIds.push(id);
      queue.push(...(this.childIds.get(id) ?? []));
    }
    this.emit({ type: "branch_purged", rootId, purgedIds, reason });
    return [...purgedIds];
  }

  // -------------------------------------------------------------------------
  // Reads
  // -------------------------------------------------------------------------

  getNode(id: NodeId): TrajectoryNode | undefined {
    return this.nodes.get(id);
  }

  children(id: NodeId): TrajectoryNode[] {
    return (this.childIds.get(id) ?? []).map((c) => this.require(c));
  }

  parents(id: NodeId): TrajectoryNode[] {
    return this.require(id).parentIds.map((p) => this.require(p));
  }

  lineage(id: NodeId): TrajectoryNode[] {
    const path: TrajectoryNode[] = [];
    for (let node: TrajectoryNode | undefined = this.require(id); node; node = node.parentIds[0] ? this.nodes.get(node.parentIds[0]) : undefined) {
      path.push(node);
    }
    return path.reverse();
  }

  frontier(options: FrontierOptions): TrajectoryNode[] {
    const candidates = this.order
      .map((id) => this.require(id))
      .filter((n) => !EXCLUDED_FROM_FRONTIER.has(n.status) && (options.includePending || n.status !== "pending"));
    const ranked = options.rankBy
      ? candidates.map((n) => ({ n, s: options.rankBy!(n) })).sort((a, b) => b.s - a.s).map((x) => x.n)
      : candidates.sort(
          (a, b) =>
            (b.metrics?.score ?? Number.NEGATIVE_INFINITY) - (a.metrics?.score ?? Number.NEGATIVE_INFINITY) ||
            (b.kev?.score ?? Number.NEGATIVE_INFINITY) - (a.kev?.score ?? Number.NEGATIVE_INFINITY),
        );
    return ranked.slice(0, Math.max(0, options.limit));
  }

  champion(): TrajectoryNode | undefined {
    return this.championId === undefined ? undefined : this.nodes.get(this.championId);
  }

  bestByGeneration(): Array<{ generation: number; score: number }> {
    // Derived from champion changes so the curve can never disagree with champion().
    const bestAt = new Map<number, number>();
    for (const e of this.log) {
      if (e.type !== "champion_changed") continue;
      const g = this.require(e.id).generation;
      bestAt.set(g, Math.max(bestAt.get(g) ?? Number.NEGATIVE_INFINITY, e.score));
    }
    const maxGen = this.order.reduce((m, id) => Math.max(m, this.require(id).generation), -1);
    const out: Array<{ generation: number; score: number }> = [];
    let running = Number.NEGATIVE_INFINITY;
    for (let g = 0; g <= maxGen; g++) {
      running = Math.max(running, bestAt.get(g) ?? Number.NEGATIVE_INFINITY);
      if (Number.isFinite(running)) out.push({ generation: g, score: running });
    }
    return out;
  }

  lineageContext(id: NodeId, maxDeadEnds = 10): LineageContext {
    const path = this.lineage(id);
    const onPath = new Set(path.map((n) => n.id));
    const index = new Map(this.order.map((nid, i) => [nid, i]));
    const deadEnds = path
      .flatMap((a) => this.children(a.id))
      .filter((c) => !onPath.has(c.id) && (c.status === "failed" || c.status === "pruned"))
      .filter((c, i, arr) => arr.findIndex((x) => x.id === c.id) === i)
      .sort((a, b) => b.createdAt - a.createdAt || index.get(b.id)! - index.get(a.id)!)
      .slice(0, Math.max(0, maxDeadEnds))
      .map((c) => ({
        id: c.id,
        strategy: c.mutation.strategy,
        ...(c.failure && { failure: { ...c.failure } }),
        ...(this.pruneReasons.has(c.id) && { pruneReason: this.pruneReasons.get(c.id)! }),
      }));
    const ancestors = path.map((n) => ({
      id: n.id,
      generation: n.generation,
      mutation: { ...n.mutation },
      ...(n.metrics && { metrics: { ...n.metrics } }),
    }));
    return {
      target: id,
      ancestors,
      deadEnds,
      approxTokens: Math.ceil(JSON.stringify({ ancestors, deadEnds }).length / 4),
    };
  }

  snapshot(): GraphSnapshot {
    const nodes = this.order.map((id) => structuredClone(this.require(id)));
    const edges: TrajectoryEdge[] = nodes.flatMap((n) =>
      n.mutation.kind === "seed" ? [] : n.parentIds.map((from) => ({ from, to: n.id, kind: n.mutation.kind as Exclude<MutationKind, "seed"> })),
    );
    const counts = Object.fromEntries(STATUSES.map((s) => [s, 0])) as Record<NodeStatus, number>;
    for (const n of nodes) counts[n.status] += 1;
    return {
      nodes,
      edges,
      ...(this.championId !== undefined && { championId: this.championId }),
      generation: nodes.reduce((g, n) => Math.max(g, n.generation), 0),
      counts,
      lastSeq: this.seq,
    };
  }

  // -------------------------------------------------------------------------
  // Event log
  // -------------------------------------------------------------------------

  events(sinceSeq = 0): readonly GraphEvent[] {
    return this.log.filter((e) => e.seq > sinceSeq).map((e) => structuredClone(e));
  }

  subscribe(listener: GraphEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Applies a recorded event without notifying listeners. Used by replay. */
  replay(event: GraphEvent): void {
    if (event.seq <= this.seq) throw new Error(`event seq ${event.seq} is not after ${this.seq}`);
    this.apply(event);
    this.seq = event.seq;
    this.log.push(structuredClone(event));
  }

  private emit(input: EventInput): void {
    const event = { ...structuredClone(input), seq: this.seq + 1, at: this.now() } as GraphEvent;
    this.apply(event);
    this.seq = event.seq;
    this.log.push(structuredClone(event));
    for (const listener of this.listeners) listener(structuredClone(event));
  }

  private apply(event: GraphEvent): void {
    switch (event.type) {
      case "node_added": {
        const node = structuredClone(event.node);
        if (this.nodes.has(node.id)) throw new Error(`duplicate node ${node.id}`);
        for (const p of node.parentIds) this.require(p);
        this.nodes.set(node.id, node);
        this.order.push(node.id);
        for (const p of node.parentIds) {
          const list = this.childIds.get(p) ?? [];
          list.push(node.id);
          this.childIds.set(p, list);
        }
        return;
      }
      case "node_scored":
        this.require(event.id).kev = { ...event.verdict };
        return;
      case "node_pruned":
        this.setStatus(event.id, "pruned");
        this.pruneReasons.set(event.id, event.reason);
        return;
      case "node_evaluated":
        this.setStatus(event.id, "evaluated").metrics = { ...event.metrics };
        return;
      case "node_failed":
        this.setStatus(event.id, "failed").failure = { ...event.failure };
        return;
      case "branch_purged":
        for (const id of event.purgedIds) this.require(id).status = "purged";
        return;
      case "champion_changed":
        this.require(event.id);
        this.championId = event.id;
        return;
    }
  }

  /** Purged is terminal: late results are stored but the status stays purged. */
  private setStatus(id: NodeId, status: NodeStatus): TrajectoryNode {
    const node = this.require(id);
    if (node.status !== "purged") node.status = status;
    return node;
  }

  /** The node itself plus every transitive parent. Empty for undefined. */
  private ancestorsOf(id: NodeId | undefined): Set<NodeId> {
    const out = new Set<NodeId>();
    const stack = id === undefined ? [] : [id];
    while (stack.length) {
      const next = stack.pop()!;
      if (out.has(next)) continue;
      out.add(next);
      stack.push(...this.require(next).parentIds);
    }
    return out;
  }

  private require(id: NodeId): TrajectoryNode {
    const node = this.nodes.get(id);
    if (!node) throw new Error(`unknown node ${id}`);
    return node;
  }
}

export const replayGraphStore: GraphStoreReplayer = (events) => {
  const store = new InMemoryGraphStore();
  for (const event of events) store.replay(event);
  return store;
};
