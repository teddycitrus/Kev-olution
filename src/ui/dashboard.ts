/**
 * Live terminal dashboard. Thin I/O shell around the pure reducer (state.ts)
 * and renderer (render.ts): subscribes to GraphStore events and EvolveEvents,
 * redraws at most `fps` times per second on the alternate screen, and prints a
 * plain summary to normal scrollback when it closes.
 *
 * Dependency-free ANSI (no Ink/Blessed) so the live demo has no install or
 * reconciler risk on Windows terminals. Non-TTY output should use the plain
 * logger in src/index.ts instead.
 */

import type { GraphStore } from "../harness/graphStore";
import type { EvolveEvent } from "../harness/runner";
import { renderDashboard, renderFinalSummary } from "./render";
import { initialState, reduceEvolve, reduceGraph, type DashboardState } from "./state";

export interface DashboardStream {
  write(chunk: string): unknown;
  columns?: number;
  rows?: number;
}

export interface DashboardOptions {
  stream?: DashboardStream;
  /** Max redraws per second. Default 12. */
  fps?: number;
  color?: boolean;
  now?: () => number;
}

const ESC = "\x1b[";
const ENTER_ALT = `${ESC}?1049h${ESC}?25l`;
const LEAVE_ALT = `${ESC}?25h${ESC}?1049l`;
const HOME = `${ESC}H`;
/** Synchronized output (DEC 2026): the terminal paints the whole frame at once. Ignored where unsupported. */
const SYNC_BEGIN = `${ESC}?2026h`;
const SYNC_END = `${ESC}?2026l`;

export class Dashboard {
  readonly state: DashboardState;
  private readonly stream: DashboardStream;
  private readonly frameMs: number;
  private readonly color: boolean;
  private readonly now: () => number;
  private timer?: ReturnType<typeof setTimeout>;
  private lastDraw = 0;
  private active = false;
  private unsubscribe?: () => void;
  private readonly onSigint = () => {
    this.stop();
    process.exit(130);
  };

  constructor(options: DashboardOptions = {}) {
    this.stream = options.stream ?? process.stdout;
    this.frameMs = 1000 / (options.fps ?? 12);
    this.color = options.color ?? true;
    this.now = options.now ?? Date.now;
    this.state = initialState(this.now());
  }

  /** Enters the alternate screen and starts listening to the graph store. */
  start(store?: GraphStore): void {
    if (this.active) return;
    this.active = true;
    if (store) this.unsubscribe = store.subscribe((e) => {
      reduceGraph(this.state, e);
      this.schedule();
    });
    process.once("SIGINT", this.onSigint);
    this.stream.write(ENTER_ALT);
    this.draw();
  }

  /** EvolveEvent sink; pass as `onEvent` to evolve(). */
  readonly onEvent = (event: EvolveEvent): void => {
    reduceEvolve(this.state, event, this.now());
    if (event.type === "finished" || event.type === "paradigm_shift" || event.type === "champion") this.draw();
    else this.schedule();
  };

  /**
   * Keeps the final frame on screen for `ms` (the demo's freeze), redrawing so
   * the badge stays up, then stops.
   */
  async hold(ms: number): Promise<void> {
    const until = this.now() + ms;
    while (this.active && this.now() < until) {
      this.draw();
      await new Promise((r) => setTimeout(r, Math.min(250, Math.max(0, until - this.now()))));
    }
    this.stop();
  }

  /** Draws the final frame, restores the terminal, and prints the summary. */
  stop(): void {
    if (!this.active) return;
    if (this.timer) clearTimeout(this.timer);
    this.timer = undefined;
    this.unsubscribe?.();
    process.removeListener("SIGINT", this.onSigint);
    this.draw();
    this.active = false;
    this.stream.write(LEAVE_ALT);
    this.stream.write(`${renderFinalSummary(this.state).join("\n")}\n`);
  }

  /** Current frame as text, for tests and snapshots. */
  frame(width = this.stream.columns ?? 120, height = this.stream.rows ?? 40): string[] {
    return renderDashboard({ ...this.state, now: this.now() }, { width, height, color: this.color });
  }

  private schedule(): void {
    if (!this.active || this.timer) return;
    const wait = Math.max(0, this.frameMs - (this.now() - this.lastDraw));
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.draw();
    }, wait);
  }

  private draw(): void {
    if (!this.active) return;
    this.lastDraw = this.now();
    this.stream.write(`${SYNC_BEGIN}${HOME}${this.frame().join("\n")}${SYNC_END}`);
  }
}
