/**
 * DAG Trajectory Memory Store: public contract.
 *
 * Replaces a flat, linear context window with a directed acyclic graph of code
 * iterations. Each node is one candidate program; edges point from parent to
 * child. Failed executions are stored as compressed summaries, never raw logs,
 * so agents can build prompts from a node's lineage without context bloat.
 *
 * State changes are event sourced: every mutation of the graph emits a
 * GraphEvent, and a store must be rebuildable by replaying its event log.
 */

import type { KevModel } from "./kevClient";

export type NodeId = string;

/** Multi-fidelity evaluation tiers (see src/harness/evaluator.ts). */
export type FidelityTier = 1 | 2 | 3;

export type NodeStatus =
  /** Created, not yet scored or evaluated. */
  | "pending"
  /** Rejected by the Kev router before any sandbox run. */
  | "pruned"
  /** Passed at least one evaluation tier. */
  | "evaluated"
  /** Failed an evaluation tier; see TrajectoryNode.failure. */
  | "failed"
  /** Removed by a Paradigm Shift purge; kept for audit, excluded from search. */
  | "purged";

export type MutationKind =
  /** Root program, usually the PyTorch reference or a hand-written baseline. */
  | "seed"
  /** Small, local edit to the parent (tile size, unroll factor, num_warps). */
  | "local"
  /** Combines two or more parents. */
  | "crossover"
  /** Large structural rewrite triggered by the Plateau Detector. */
  | "paradigm_shift";

export interface MutationInfo {
  kind: MutationKind;
  /** Short strategy label, e.g. "warp-level tiling", "swap K/V loop order". */
  strategy: string;
  /** One or two sentence rationale from the proposing agent. */
  rationale: string;
}

export interface ExecutionMetrics {
  /** Highest tier this candidate has passed. */
  tier: FidelityTier;
  /** Throughput; undefined for Tier 1 (static checks only). */
  score?: number;
  /** Kernel latency in milliseconds. */
  latencyMs?: number;
  /** Max absolute error against the reference output. */
  maxAbsError?: number;
  /** Wall-clock time spent in the sandbox, all tiers combined. */
  evalWallMs: number;
}

export type FailureKind =
  | "syntax"
  | "compile"
  | "runtime"
  | "numerical"
  | "timeout"
  | "regression";

/** Compressed failure record. Raw traces stay on disk, never in the graph. */
export interface FailureSummary {
  kind: FailureKind;
  tier: FidelityTier;
  /** One-line summary, capped at MAX_FAILURE_SUMMARY_CHARS. */
  summary: string;
  /** Optional path to the full sandbox log for human debugging. */
  logPath?: string;
}

export const MAX_FAILURE_SUMMARY_CHARS = 280;

/** Kev router verdict on a candidate, recorded before any sandbox run. */
export interface KevVerdict {
  model: KevModel;
  /** Predicted promise of the candidate in [0, 1]. Higher is better. */
  score: number;
  /** Round-trip decision latency in milliseconds. */
  latencyMs: number;
}

export interface TrajectoryNode {
  readonly id: NodeId;
  /** Empty for seeds; one entry for local mutations; two or more for crossover. */
  readonly parentIds: readonly NodeId[];
  /** Generation index; seeds are 0, children are max(parent generation) + 1. */
  readonly generation: number;
  readonly createdAt: number;
  /** Full candidate source (Triton kernel, CUDA, etc). */
  readonly source: string;
  /** Unified diff against the first parent; absent for seeds. */
  readonly diff?: string;
  readonly mutation: MutationInfo;
  status: NodeStatus;
  kev?: KevVerdict;
  metrics?: ExecutionMetrics;
  failure?: FailureSummary;
}

export interface TrajectoryEdge {
  readonly from: NodeId;
  readonly to: NodeId;
  readonly kind: Exclude<MutationKind, "seed">;
}

/** Input for GraphStore.addNode. The store assigns id, generation, createdAt. */
export interface NewNodeInput {
  parentIds: readonly NodeId[];
  source: string;
  diff?: string;
  mutation: MutationInfo;
}

// ---------------------------------------------------------------------------
// Events (event sourcing)
// ---------------------------------------------------------------------------

interface BaseEvent {
  /** Monotonic sequence number within one store. */
  seq: number;
  at: number;
}

export type GraphEvent =
  | (BaseEvent & { type: "node_added"; node: TrajectoryNode })
  | (BaseEvent & { type: "node_scored"; id: NodeId; verdict: KevVerdict })
  | (BaseEvent & { type: "node_pruned"; id: NodeId; reason: string })
  | (BaseEvent & { type: "node_evaluated"; id: NodeId; metrics: ExecutionMetrics })
  | (BaseEvent & { type: "node_failed"; id: NodeId; failure: FailureSummary })
  | (BaseEvent & { type: "branch_purged"; rootId: NodeId; purgedIds: NodeId[]; reason: string })
  | (BaseEvent & { type: "champion_changed"; id: NodeId; previousId?: NodeId; score: number });

export type GraphEventType = GraphEvent["type"];

export type GraphEventListener = (event: GraphEvent) => void;

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/** Scoring function used to rank the search frontier. */
export type FrontierRanker = (node: TrajectoryNode) => number;

export interface FrontierOptions {
  /** Max nodes to return. */
  limit: number;
  /** Defaults to ranking by metrics.score, then kev.score. */
  rankBy?: FrontierRanker;
  /** Include nodes still in "pending" status. Default false. */
  includePending?: boolean;
}

/**
 * Compact, prompt-ready view of a node's ancestry. Agents use this in place of
 * raw history: successful ancestors carry metrics, failed siblings carry only
 * their FailureSummary.
 */
export interface LineageContext {
  target: NodeId;
  /** Root first, target last. */
  ancestors: Array<Pick<TrajectoryNode, "id" | "generation" | "mutation" | "metrics">>;
  /** Failed or pruned children of ancestors, most recent first. */
  deadEnds: Array<{ id: NodeId; strategy: string; failure?: FailureSummary; pruneReason?: string }>;
  /** Rough token estimate of the rendered context. */
  approxTokens: number;
}

/** Plain, serializable view consumed by the dashboard. */
export interface GraphSnapshot {
  nodes: TrajectoryNode[];
  edges: TrajectoryEdge[];
  championId?: NodeId;
  generation: number;
  counts: Record<NodeStatus, number>;
  lastSeq: number;
}

// ---------------------------------------------------------------------------
// Store contract
// ---------------------------------------------------------------------------

export interface GraphStore {
  // Writes. Each emits exactly one GraphEvent (champion_changed may follow
  // recordEvaluation as a second event).

  /** Adds a node. Throws if any parentId is unknown or the graph would cycle. */
  addNode(input: NewNodeInput): TrajectoryNode;
  recordVerdict(id: NodeId, verdict: KevVerdict): void;
  markPruned(id: NodeId, reason: string): void;
  /** Sets status "evaluated" and updates the champion if score improved with zero error. */
  recordEvaluation(id: NodeId, metrics: ExecutionMetrics): void;
  /** Truncates failure.summary to MAX_FAILURE_SUMMARY_CHARS before storing. */
  recordFailure(id: NodeId, failure: FailureSummary): void;
  /**
   * Marks every descendant of rootId (excluding rootId) as "purged".
   * Used by the Paradigm Shift agent to abandon a stalled local basin.
   * Returns the purged ids.
   */
  purgeBranch(rootId: NodeId, reason: string): NodeId[];

  // Reads.
  getNode(id: NodeId): TrajectoryNode | undefined;
  children(id: NodeId): TrajectoryNode[];
  parents(id: NodeId): TrajectoryNode[];
  /** Root-to-node path following first parents. */
  lineage(id: NodeId): TrajectoryNode[];
  frontier(options: FrontierOptions): TrajectoryNode[];
  champion(): TrajectoryNode | undefined;
  /** Best champion score per generation, for plateau detection and graphs. */
  bestByGeneration(): Array<{ generation: number; score: number }>;
  lineageContext(id: NodeId, maxDeadEnds?: number): LineageContext;
  snapshot(): GraphSnapshot;

  // Event log.
  events(sinceSeq?: number): readonly GraphEvent[];
  subscribe(listener: GraphEventListener): () => void;
}

/** Rebuilds a store from an event log. Implementations must be deterministic. */
export type GraphStoreReplayer = (events: readonly GraphEvent[]) => GraphStore;
