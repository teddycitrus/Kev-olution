/**
 * Daily quota tracking for Kev SystemOne (kev-9b: 1000 calls/day,
 * kev-27b: 200 calls/day). Re-exported from kevClient.ts.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { KevModel, ServerUsage } from "./kevClient";

export const KEV_DAILY_LIMITS: Readonly<Record<KevModel, number>> = {
  "kev-9b": 1000,
  "kev-27b": 200,
};

export class KevQuotaExceededError extends Error {
  constructor(readonly model: KevModel, readonly limit: number) {
    super(`Kev daily quota exhausted for ${model} (${limit} calls)`);
    this.name = "KevQuotaExceededError";
  }
}

export interface QuotaState {
  /** UTC date, YYYY-MM-DD. Counters reset when this changes. */
  date: string;
  used: Record<KevModel, number>;
}

export interface QuotaStore {
  load(): QuotaState | undefined;
  save(state: QuotaState): void;
}

export interface QuotaSnapshot {
  date: string;
  used: Record<KevModel, number>;
  remaining: Record<KevModel, number>;
}

export class MemoryQuotaStore implements QuotaStore {
  private state?: QuotaState;
  load(): QuotaState | undefined {
    return this.state && structuredClone(this.state);
  }
  save(state: QuotaState): void {
    this.state = structuredClone(state);
  }
}

/** Persists usage to a JSON file. Writes atomically via rename. */
export class FileQuotaStore implements QuotaStore {
  constructor(private readonly path = ".kev/usage.json") {}

  load(): QuotaState | undefined {
    if (!existsSync(this.path)) return undefined;
    try {
      const parsed = JSON.parse(readFileSync(this.path, "utf8")) as Partial<QuotaState>;
      if (typeof parsed.date !== "string" || !parsed.used) return undefined;
      return {
        date: parsed.date,
        used: {
          "kev-9b": Number(parsed.used["kev-9b"]) || 0,
          "kev-27b": Number(parsed.used["kev-27b"]) || 0,
        },
      };
    } catch {
      return undefined;
    }
  }

  save(state: QuotaState): void {
    mkdirSync(dirname(this.path), { recursive: true });
    const tmp = `${this.path}.tmp`;
    writeFileSync(tmp, JSON.stringify(state, null, 2));
    renameSync(tmp, this.path);
  }
}

const utcDate = (now: Date): string => now.toISOString().slice(0, 10);

export class QuotaTracker {
  private state: QuotaState;

  constructor(
    private readonly store: QuotaStore,
    private readonly limits: Readonly<Record<KevModel, number>> = KEV_DAILY_LIMITS,
    private readonly now: () => Date = () => new Date(),
  ) {
    this.state = this.freshOrLoaded();
  }

  /** Reserves one call. Throws KevQuotaExceededError when the limit is reached. */
  reserve(model: KevModel): void {
    this.rollover();
    if (this.state.used[model] >= this.limits[model]) {
      throw new KevQuotaExceededError(model, this.limits[model]);
    }
    this.state.used[model] += 1;
    this.store.save(this.state);
  }

  /** Returns a reservation for a call the server did not bill. */
  release(model: KevModel): void {
    this.rollover();
    this.state.used[model] = Math.max(0, this.state.used[model] - 1);
    this.store.save(this.state);
  }

  /** Adopts server counts when they are higher (other processes share the key). */
  reconcile(server: ServerUsage): void {
    this.rollover();
    if (server.date !== this.state.date) return;
    for (const model of Object.keys(this.limits) as KevModel[]) {
      const remote = server.usage[model]?.used;
      if (typeof remote === "number" && remote > this.state.used[model]) {
        this.state.used[model] = remote;
      }
    }
    this.store.save(this.state);
  }

  remaining(model: KevModel): number {
    this.rollover();
    return Math.max(0, this.limits[model] - this.state.used[model]);
  }

  snapshot(): QuotaSnapshot {
    this.rollover();
    return {
      date: this.state.date,
      used: { ...this.state.used },
      remaining: { "kev-9b": this.remaining("kev-9b"), "kev-27b": this.remaining("kev-27b") },
    };
  }

  private freshOrLoaded(): QuotaState {
    const today = utcDate(this.now());
    const loaded = this.store.load();
    if (loaded && loaded.date === today) return loaded;
    return { date: today, used: { "kev-9b": 0, "kev-27b": 0 } };
  }

  private rollover(): void {
    if (this.state.date !== utcDate(this.now())) {
      this.state = { date: utcDate(this.now()), used: { "kev-9b": 0, "kev-27b": 0 } };
      this.store.save(this.state);
    }
  }
}
