/**
 * Speculative response cache for Kev SystemOne, used in --demo mode.
 *
 * Goal: routing decisions return in under 5 ms during a live pitch. Three
 * sources feed the cache:
 *   1. Rehearsal snapshots: a prior run with the same seed is saved to disk and
 *      loaded at startup, so a deterministic demo replays with zero network
 *      calls and zero quota.
 *   2. Speculation: the harness calls KevClient.speculate() for requests it
 *      expects to make soon (e.g. scoring the next generation while the
 *      current one is in the sandbox). Results land in the cache.
 *   3. Write-through: every network response is stored.
 *
 * Only exact matches hit: the key is a hash of model + state + questions with
 * object keys sorted. A hit returns the response Kev actually gave for that
 * input; nothing is synthesized. Callers see `source` on every result so the
 * dashboard can report cache latency separately from Kev network latency.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { KevModel, KevQuestion, KevState, SystemOneResponse } from "./kevClient";

export type KevResultSource = "network" | "cache" | "inflight";

export interface DemoCacheOptions {
  /** LRU capacity. Default 2048 entries. */
  maxEntries?: number;
  /** Max concurrent speculative requests. Default 4. */
  maxSpeculative?: number;
  /** Rehearsal snapshot to load at startup and write with saveSnapshot(). */
  snapshotPath?: string;
}

export interface CacheStats {
  hits: number;
  inflightHits: number;
  misses: number;
  speculative: number;
  entries: number;
}

interface SnapshotFile {
  version: 1;
  entries: Array<[string, SystemOneResponse]>;
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).filter((k) => obj[k] !== undefined).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalJson(obj[k])}`).join(",")}}`;
}

export function cacheKey(model: KevModel, state: KevState, questions: Record<string, KevQuestion>): string {
  return createHash("sha1").update(canonicalJson({ model, state, questions })).digest("hex");
}

export class KevResponseCache {
  readonly maxSpeculative: number;
  private readonly maxEntries: number;
  private readonly entries = new Map<string, SystemOneResponse>();
  private readonly inflight = new Map<string, Promise<SystemOneResponse>>();
  private readonly counters = { hits: 0, inflightHits: 0, misses: 0, speculative: 0 };
  private speculativeActive = 0;

  constructor(private readonly options: DemoCacheOptions = {}) {
    this.maxEntries = options.maxEntries ?? 2048;
    this.maxSpeculative = options.maxSpeculative ?? 4;
    if (options.snapshotPath) this.loadSnapshot(options.snapshotPath);
  }

  /** Returns a cached response (refreshing LRU order) and counts the hit. */
  get(key: string): SystemOneResponse | undefined {
    const hit = this.entries.get(key);
    if (hit) {
      this.entries.delete(key);
      this.entries.set(key, hit);
      this.counters.hits += 1;
    }
    return hit;
  }

  getInflight(key: string): Promise<SystemOneResponse> | undefined {
    const pending = this.inflight.get(key);
    if (pending) this.counters.inflightHits += 1;
    return pending;
  }

  recordMiss(): void {
    this.counters.misses += 1;
  }

  set(key: string, response: SystemOneResponse): void {
    this.entries.delete(key);
    this.entries.set(key, response);
    while (this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  /** Registers a pending request so concurrent identical asks share it. */
  track(key: string, pending: Promise<SystemOneResponse>, speculative = false): void {
    this.inflight.set(key, pending);
    if (speculative) {
      this.speculativeActive += 1;
      this.counters.speculative += 1;
    }
    pending
      .then((response) => this.set(key, response))
      .catch(() => undefined)
      .finally(() => {
        this.inflight.delete(key);
        if (speculative) this.speculativeActive -= 1;
      });
  }

  has(key: string): boolean {
    return this.entries.has(key) || this.inflight.has(key);
  }

  canSpeculate(): boolean {
    return this.speculativeActive < this.maxSpeculative;
  }

  stats(): CacheStats {
    return { ...this.counters, entries: this.entries.size };
  }

  /** Writes all entries to disk for a later rehearsal replay. */
  saveSnapshot(path = this.options.snapshotPath): void {
    if (!path) throw new Error("saveSnapshot requires a path");
    const file: SnapshotFile = { version: 1, entries: [...this.entries] };
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(`${path}.tmp`, JSON.stringify(file));
    renameSync(`${path}.tmp`, path);
  }

  private loadSnapshot(path: string): void {
    if (!existsSync(path)) return;
    try {
      const file = JSON.parse(readFileSync(path, "utf8")) as Partial<SnapshotFile>;
      if (file.version !== 1 || !Array.isArray(file.entries)) return;
      for (const entry of file.entries) {
        if (Array.isArray(entry) && typeof entry[0] === "string" && entry[1] && typeof entry[1] === "object") {
          this.set(entry[0], entry[1]);
        }
      }
    } catch {
      // A corrupt snapshot only costs cache hits; never block startup on it.
    }
  }
}
