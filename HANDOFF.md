# AGENT HANDOFF PROTOCOL

## Current Goal
DEMO READY. The PRD Definition of Done was verified live on 2026-10-17: kev-9b routing and kev-27b Paradigm Shift choices came from the self-hosted endpoints, the run reached 176.1 TFLOPS (simulated H100), and a 45 s deterministic cache replay works. Remaining work is small: CLI wiring and test coverage for the new telemetry bar.

## Role Division
- **Claude Code (Terminal 1):** System Architecture, API Schemas, Refactoring, & Code Review.
- **Devin (Terminal 2):** Feature Implementation, Test Suites, Boilerplate, & Build Fixes.

---

## Active State & Checklist

- [x] Steps 1-6: Harness core, Kev client, evaluator, DAG store, agents, loop, CLI, and tests.
- [x] Step 7: Claude Code built the live terminal dashboard.
- [x] Step 8: Devin wrote the dashboard and endpoint tests, plus `scripts/demo.ts` / `bun run demo`, and fixed three UI issues.
- [x] Step 9: Claude Code reviewed Step 8 (all three fixes accepted), ran the live DoD pass, added the Kev telemetry bar and Paradigm Shift banner, and set demo pacing to 45 s. `bun run typecheck` is clean and `bun test` passes 483/483.
- [x] Step 10: Devin added telemetry, state, and runner tests, plus `demo:offline` and `--pace` passthrough docs. `bun test` 507/507 across 17 files, `bun run typecheck` clean.
- [x] Step 11: Claude Code reviewed the `runDemo` `--offline` guard (13:23) and accepted it. `--offline` skips warmup and routes heuristically, with no false cache claim. `bun test` 507/507 and `bun run typecheck` are clean. Keepalive (`bun run keepalive`) is pinging both Modal containers every 3 min until about 14:16.
- [x] Step 12: Claude Code refactored the dashboard for the 45 s pitch. `bun run evolve --demo` = about 38 s paced run plus a 7 s frozen final frame. `bun run typecheck` is clean and `bun test` passes 507/507.
- [ ] Step 13: **[Devin]** Tests for the new pitch features (list below). Do not change `src/` behavior before the demo.
- [x] Step 14: Claude Code added a localhost web visualizer (`bun run web`, http://localhost:4173). It is new files only, plus one `package.json` script; the terminal dashboard and every existing `bun run` demo are unchanged. `bun test` 507/507, `bun run typecheck` clean.
- [ ] Step 15: **[Devin]** Tests for the web server (list below). Do not modify the terminal dashboard or `src/index.ts`.
- [x] Step 17: Claude Code wrote `README.md` in the project root (project name **Kev-olution**, othneildrew Best-README-Template, ASCII only). Repo links assume `teddycitrus/kev-olution`. The README references a `LICENSE` file that does not exist yet. @Devin: no action needed.
- [x] Step 16: Claude Code redesigned the web dashboard (sleek dark layout) and added the "Time saved by Kev" modal, backed by a measured no-Kev counterfactual (see Step 16 notes). Terminal demos are unchanged. `bun run typecheck` is clean.

---

## Step 16 notes (Last updated by: Claude Code, 14:17)

### What Was Just Done:
- **`src/web/server.ts`.**
  - New `RunWork` (cycles, candidates, compiles = evaluations reaching Tier 2, benchmarks = reaching Tier 3, simulated GPU-s, Kev calls, best), with `emptyWork()` and `tallyWork(w, event)`.
  - `noKevBaseline()` runs the identical search (seed 0, same task and evaluator, `kev: null`) once, unpaced (about 1 s), and caches it. It is precomputed at server start.
  - Every run publishes `{ kind: "baseline", work }` right after the `run` event.
  - Measured: no-Kev = 26 cycles, 130 candidates, 79 compiles, 15 benchmarks, 419 GPU-s. Kev demo run = 25, 119, 47, 15, 371 GPU-s, 28 Kev calls. Both reach 176.1 TFLOPS.
- **`src/web/index.html` (rewritten).**
  - Dark dashboard in the style of the user's reference: sidebar (brand, nav, run buttons, SIMULATED H100 pill); header with mode, elapsed, and a clock button.
  - KPI cards with icon tiles: best, speedup, Kev decisions/pruned, est. time saved.
  - Throughput chart with a gradient area, shift markers, and target/baseline lines.
  - Quick stats with progress bars: target progress, % pruned, compiles avoided vs no-Kev, simulated compute avoided, shifts.
  - Search DAG, Top kernels, Recent activity (icon tiles with "Ns ago"), and a Kev routing panel.
  - Shift alert and done alert. The page is still empty on load.
- **"Time saved by Kev" modal.** Opens from the clock button, the time-saved KPI card, or the done alert, and opens automatically 1.6 s after a Kev run finishes.
  - Shows a with/without table of measured work counts, plus wall-clock = candidates x gen + compiles x compile + benchmarks x bench + Kev calls x kev.
  - The upper-end default costs (per the user's request to take the high end) are editable inputs: 40 s LLM-written kernel, 35 s compile + check, 20 s benchmark, 1.3 s Kev call.
  - Result for the demo: about 1.9 h with Kev vs 2.3 h without, **about 25 min saved (18%)**, nearly all from 32 fewer compiles.
- Verified: `node --check` on the page JS, a fresh page gets 0 events, and a full demo stream replayed through the page logic gives the numbers above.

### Actionable Next Task for Next Agent:
- @Devin: extend Step 15 (`tests/webServer.test.ts`):
  1. `tallyWork` counts compiles (`evaluated.tier >= 2`) and benchmarks (`tier >= 3`) and copies the `finished` stats.
  2. `noKevBaseline()` returns 26 cycles, 79 compiles, and 15 benchmarks at seed 0, and the same promise on a second call.
  3. Every run emits a `baseline` event immediately after `run`.

## Step 14 notes (Last updated by: Claude Code, 13:44)

### What Was Just Done:
- **`src/web/server.ts` (new).** A `Bun.serve` server on 127.0.0.1:4173 (`--port` to change).
  - `GET /` serves the page.
  - `GET /events` is Server-Sent Events: the full history of the current run, then live events.
  - `POST /run?mode=demo|live|offline&pace=MS` starts a run. It returns 409 while a run is in progress, rejects bad modes with 400, and clamps pace to 0 to 1000.
  - The loop wiring mirrors `src/index.ts` (seed 0) but does not import or change it.
  - Graph events are trimmed with `trimGraphEvent`: `node_added` sends id, parents, generation, kind, strategy, and the config label, never kernel source.
  - Modes:
    - `demo` replays `.kev/demo-cache.json`.
    - `live` uses an empty in-memory cache, so every decision goes to Kev, and saves to `.kev/web-live-cache.json`. It never touches the demo cache, so the terminal demos stay deterministic.
    - `offline` makes no Kev calls.
- **`src/web/index.html` (new).** A single file with no external dependencies (works offline on stage).
  - Header: task line (benchmark shape and hardware) and the SIMULATED H100 tag.
  - KPI cards: best, speedup, cycle, active model, routing p50 and model p50, Kev decisions and pruned count, compute avoided % with $ est., and shifts.
  - Throughput chart (SVG): best-so-far step line, every candidate as a dot (green ok, red rejected), target and baseline lines, gold shift markers labeled with the rewrite.
  - Search DAG (SVG): every node by generation, green viable, red pruned or failed, gold shift, gray purged, with the champion lineage highlighted and hover tooltips showing config, TFLOPS, Kev score, and failure summary.
  - Shift banner (3.5 s, animated) showing kev-27b's choice probabilities.
  - Final verification badge, worded by backend (same rule as the terminal badge).
  - Rolling event log.
- **`package.json`.** Added the `"web"` script.
- **Empty start (13:58).** `/events` replays history only while a run is in progress, so a fresh page load starts empty and a page opened mid-run catches up. Verified: 0 events before a run, 60 events for a mid-run join, 0 after the run finishes. @Devin: in Step 15, test 3 must assert that a subscriber arriving after `finished` receives nothing, and that a mid-run subscriber receives the history.
- **Verified.**
  - The page serves (24 KB) and its JS passes `node --check`.
  - A full demo run streamed 543 events. Replaying them through the page's own handlers with a DOM stub gives 176.1 TFLOPS, 4.27x, 3 shifts, Kev routing p50 0.10 ms (cache) and model p50 129 ms, 113 decisions with 73 pruned, 86% avoided, and the badge shown.
  - Not checked in a real browser: the Chrome extension is not connected in this session.

### Actionable Next Task for Next Agent:
- @Devin (Step 15), tests only, in `tests/webServer.test.ts`. Use `createServer(0)` for a random port, `mode=offline`, and `pace=0` so there is no network and no Kev.
  1. `GET /` returns 200 HTML containing `KEV-EVOLVE`.
  2. `POST /run?mode=bogus` returns 400; a second `POST /run` while running returns 409.
  3. `/events` streams a `run` event first and ends with an `evolve` `finished` event. A mid-run subscriber receives the history so far; a subscriber connecting after `finished` receives nothing (the page starts empty).
  4. `trimGraphEvent` drops `source` from `node_added` and passes other events through unchanged.
  5. Pace is clamped (`pace=99999` gives 1000; `pace=-5` gives 0).
  6. Stop the server in `afterAll`.

---

## Step 12 notes (Last updated by: Claude Code, 13:33)

### What Was Just Done:
- **Layout (`renderDashboard`, width 120 and up).**
  - Top: title `KEV-EVOLVE: Frontier Research & Optimization Harness` with the backend and routing labels, a status line, the telemetry bar, the status or shift banner, and the verification badge on the final frame.
  - Left: the DAG trajectory.
  - Right: the TFLOPS chart, with Kev latency and compute cost below it.
  - Bottom: the event log at full width.
  - Narrow terminals stack chart, Kev, cost, and DAG.
- **Telemetry bar.** decisions/s (labeled by source), `routing latency` p50 with model p50, `cost savings` % with $ (est.), and `active model` (`Kev-9B`; `Kev-27B` during a kev-27b shift cycle; `fallback order`; or `heuristic (offline)`), via the new exported `activeModel(s)`.
- **Shift banner.** It alternates red and yellow backgrounds every 400 ms: ` !! PARADIGM SHIFT TRIGGERED (PLATEAU DETECTED AT CYCLE N - ROUTED TO KEV-27B): <REWRITE> !! `. When kev-27b did not decide, it says `KEV-27B UNAVAILABLE, FALLBACK ORDER`. It is ASCII only, keeping the exact-width contract.
- **DAG colors.** Green = viable, red = pruned or failed, gold (bold yellow) = Paradigm Shift nodes and SHIFT tags, gray = purged or pending. Each lineage line now reads `id TFLOPS [<- champion] [SHIFT] config [tally]`, so truncation only cuts the config.
- **`renderBadge(s)` (new, final frame only).** The green badge `[0.000% NUMERICAL ERROR vs REFERENCE (SIMULATED H100 MODEL) | 176.1 TFLOPS (4.27x)]` uses the champion's real `maxAbsError` (new `DagNode.maxAbsError`, set from `node_evaluated`). On the GPU backend it reads `[VERIFIED ON GPU: max abs error ... vs PYTORCH REFERENCE ...]`. A nonzero simulated error gives a yellow badge.
- **`Dashboard`.**
  - Frames are wrapped in synchronized output (`ESC[?2026h ... ESC[?2026l`) to avoid flicker; terminals without support ignore it.
  - New `hold(ms)` redraws the final frame every 250 ms for `ms`, then calls `stop()`.
  - No new retained state per frame: the log stays capped at 200 lines and timers are cleared on stop.
- **CLI.**
  - `--demo` without `--pace` now defaults to `DEMO_PACE_MS = 100`.
  - With the dashboard in `--demo`, the final frame is held for `DEMO_HOLD_MS = 7000`.
  - `DEMO_ARGS` in `scripts/demo.ts` uses `--pace 100`.
- **Verified.**
  - The cache-replay `bun run evolve --demo --plain` took 38.4 s (plus the 7 s hold in a TTY, about 45 s total), with the same 3 kev-27b shifts and 176.1 TFLOPS.
  - Frames rendered at 140x44 at the shift moment and at the final frame: every line is exactly 140 columns, with 0 cache misses (no network).
  - Measured at the end: 86% compute avoided, $2.42 est., and a recorded Kev model p50 of about 129 ms.
- **Tests I updated to the new format (assertions only):**
  - `tests/uiRender.test.ts`: header, telemetry text, banner text, side-by-side check, DAG line order, champion-marker line lookup.
  - `tests/dashboard.test.ts`: the frame prefix now includes the sync-begin escape, and sync-end is stripped before width checks.
  - `tests/demo.test.ts`: pace 100.
- **Not done, on purpose.** The scripted timeline values from the request (8 ms, 120+ evals/s, $14.20, a 42 -> 88 -> 142 curve) and a "PYTORCH MATCH" claim are not hardcoded. The dashboard shows only measured values, and the badge names the simulated model. The real run follows the same story arc: expansion, a plateau around 83 TFLOPS with red pruned branches, the kev-27b shift banner, the jump to 114/130, then 163 -> 176, and the frozen green badge.

### Actionable Next Task for Next Agent:
- @Devin (Step 13), tests only:
  1. `activeModel`: returns Kev-27B in a kev-27b shift cycle, `fallback order` in a fallback shift cycle, Kev-9B with Kev routing otherwise, and `heuristic (offline)` offline.
  2. Banner: background style alternates with `now` (400 ms buckets) when color is on; it uses the `FALLBACK ORDER` wording when `decidedBy` is fallback.
  3. `renderBadge`: undefined before `finished`; the simulated, zero-error text; the yellow variant for a nonzero error; the GPU wording.
  4. `Dashboard.hold(ms)` with an injected clock redraws and then stops; every frame write starts with `ESC[?2026h` and ends with `ESC[?2026l`.
  5. `parseArgs(["--demo"]).paceMs === 100`; `parseArgs(["--demo", "--pace", "40"]).paceMs === 40`.
  6. The DAG gold style is applied to `paradigm_shift` nodes when color is on.

---

## Devin Step 10 notes (Last updated by: Devin, 13:18)

### What Was Just Done:
- `tests/uiRender.test.ts` (+16):
  - `renderTelemetry`:
    - exact all-`-` empty bar
    - decisions/s value and label for every source mix (inflight counts as cache); `60k` formatting; `-` when routing time is 0
    - model and round-trip p50 sources (heuristic latencies excluded)
    - dollar formula and `est.` label
    - banner for the shift cycle and the next one, then stagnating, improving, and the stop reason winning over the banner
    - the 4-line header with exact widths in color
  - `sparkline`: empty, constant, scaled, last-`width` samples, and all zeros.
- `tests/uiState.test.ts` (+3):
  - `decisions`/`decisionMs`/`modelLatency` accumulation (`modelLatency` only when present)
  - `stalledCycles` (<= 1% counts, reset on an improvement, repeated cycle numbers)
  - `paradigm_shift` resets the stall count, sets `lastShiftCycle`, and the shift cycle itself is not counted
- `tests/runner.test.ts` (+2): `modelLatencyMs === 12` on network answers and on demo-cache hits (same client, second run); heuristic decisions omit it.
- **CLI wiring:**
  - `package.json` `"demo:offline": "bun run scripts/demo.ts --offline"`.
  - `scripts/demo.ts` docs list `--plain`, `--pace 140`, and `demo:offline`.
  - **Fix:** `runDemo` used to warm both endpoints whenever keys were set, even with `--offline`, so the no-network fallback could block for up to 6.5 min on cold containers. `--offline` now skips warmup and prints that it is offline. Covered in `tests/demo.test.ts` (+3).
- Also available: `bun run rehearse` (offline, seed 0, `--pace 107`, about 47 s verified).

### Actionable Next Task for Next Agent:
- @Claude Code: nothing blocking. Optionally review the `runDemo` `--offline` guard. Use `bun run demo:offline` as the fallback if the endpoints are cold.

### Files Modified in Last Session (Devin):
- `tests/uiRender.test.ts`, `tests/uiState.test.ts`, `tests/runner.test.ts`, `tests/demo.test.ts`
- `scripts/demo.ts` (`--offline` skips warmup; passthrough docs)
- `package.json` (`demo:offline`)
- `AGENTS.md` (4-minute HANDOFF polling rule), `HANDOFF.md`

---

## Latest Handoff Notes (Last updated by: Claude Code)

### What Was Just Done:
- **Step 8 review: accepted as is.**
  - `stop()` now draws before deactivating, so an aborted run keeps its final frame.
  - `draw()` requires `active`, so nothing leaks into scrollback after stop.
  - The DAG indent is capped at 3, and `<- champion` comes before SHIFT.
  - `scripts/demo.ts` is fine.
- **Live DoD pass (self-hosted Modal endpoints, `bun run demo --plain`).** Logs are in `.kev/dod-run1.log` and `.kev/dod-run2.log`.
  - **Run 1 (network):**
    - Warmup: kev-9b ready in 137 s, kev-27b in 207 s (cold).
    - Result: 25 cycles, converged at **176.1 TFLOPS, 4.27x**, target reached.
    - kev-27b made all three shift choices: **TMA, then warp specialization, then persistent**. It skipped the block_ptr step the fallback order would have tried, so it needed 3 shifts instead of 4.
    - kev-9b: 25 billed calls, pruning 73 of 119 candidates, so only 47 sandbox evaluations were needed (79 offline). kev-9b round trip p50 219 ms, max 1264 ms.
    - The paced loop took about 61 s at pace 140. `.kev/demo-cache.json` was written (8.6 KB).
  - **Run 2 (cache replay, pace 120):**
    - Identical result and shifts.
    - 25/25 Kev decisions served from `cache` (slowest 3.1 ms), with 0 billed calls.
    - **46.0 s wall time**, including the warmup check (0.2 s, already warm).
  - **Measured vs PRD targets.** The Kev-reported model time on these calls is about 120 ms, not under 20 ms: long text states plus 6 questions with diffs per call, and the first request of a new shape runs without a CUDA graph. The dashboard shows the real value.
  - The pre-DoD cache from my earlier 12-cycle test was moved to `.kev/demo-cache.pre-dod.json.bak`, not deleted.
- **`src/ui/render.ts`: telemetry bar** (`renderTelemetry`, inserted under the header, 2 lines). It shows:
  - decisions/s (candidates routed per second of routing time), labeled `(cache)`, `(network)`, `(mixed)`, or `(heuristic)`
  - Kev-reported model latency p50
  - round-trip p50
  - $ saved (est.) = GPU-seconds avoided x `GPU_DOLLARS_PER_HOUR` (3.95, the Modal H100 rate), with the % avoided
  - status: `improving (local search)`, `stagnating N/3 cycles without >1% gain`, `!! PARADIGM SHIFT TRIGGERED: <rewrite> [kev-27b|fallback] !!` (shown during the shift cycle and the one after), or the final stop reason
- **`src/ui/render.ts`, other changes.** `sparkline()` adds a Kev latency sparkline in the Kev panel. Lineage node ids are green when evaluated and red otherwise. All new text is ASCII (no emoji), so the exact-width frame contract and the existing width tests still hold.
- **`src/ui/state.ts`.** New state fields: `modelLatency[]`, `decisions`, `decisionMs`, `stalledCycles` (resets on a shift; 1% threshold, matching PlateauDetector), and `lastShiftCycle`.
- **`src/harness/runner.ts`.** `kev_decision` gains an optional `modelLatencyMs`: Kev's reported `latency_ms`, which on cache hits is the value recorded when Kev first answered. This is additive and no existing fields changed.
- **Demo pacing.** `DEMO_ARGS` in `scripts/demo.ts` now uses `--pace 120` (about 45 s on replay). The three matching assertions in `tests/demo.test.ts` were updated.
- **Not done, on purpose.** The dashboard shows measured values only. I did not hardcode the brief's example numbers (120 decisions/s, under 15 ms, $14.20 / 86%, a 42 -> 88 -> 142 curve). The real curve is 41 -> 83 -> 114/130 -> 163 -> 176, and the real estimated savings are about $1.4 to $2.4 per run at H100 rates.

### Actionable Next Task for Next Agent:
- @Devin (Step 10). Tests only, plus one small CLI addition; no network.
  1. **`tests/uiRender.test.ts`: `renderTelemetry`.**
     - Shows `-` for every metric before any decision.
     - The decisions/s value and its `(cache)`/`(network)`/`(mixed)`/`(heuristic)` label for each source mix; values of 10000 or more render as `Nk`.
     - The dollar estimate equals `(earlyExit + pruning) / 3600 * GPU_DOLLARS_PER_HOUR`, formatted `$x.xx`, and always says `est.`.
     - The status line is the PARADIGM SHIFT banner when `cycle - lastShiftCycle <= 1`, the stagnating line when `stalledCycles > 0`, `improving` otherwise, and the stop reason once `summary` is set.
     - `sparkline`: empty input, constant input, and max-scaled output of the requested width.
     - Re-run the existing exact-width assertions; the header is now 4 lines.
  2. **`tests/uiState.test.ts`.**
     - `kev_decision` accumulates `decisions` (kept + pruned), `decisionMs`, and `modelLatency` (only when `modelLatencyMs` is present).
     - `stalledCycles` increments on cycles with 1% gain or less, resets on an improvement and on `paradigm_shift`.
     - `lastShiftCycle` is set by `paradigm_shift`.
  3. **`tests/runner.test.ts`.** With a mocked Kev whose responses carry `latency_ms: 12`, `kev_decision.modelLatencyMs === 12` for network answers and also for a cache hit (demo mode, same state twice).
  4. **CLI wiring.** Add `--pace` to the `bun run demo` passthrough docs in `scripts/demo.ts` (extra args are already appended, so `bun run demo --pace 140` overrides). Add a `package.json` script `"demo:offline": "bun run scripts/demo.ts --offline"` for a no-network fallback if the endpoints are cold at demo time. Note that the offline run takes 26 cycles at pace 120, about 50 s.
  5. Run `bun test` and `bun run typecheck`. Update this file.

### Demo runbook (for the user):
1. About 7 minutes before the demo, warm both endpoints: `bun run evolve --warmup --cycles 0 --plain`. Cold starts take up to 6.5 min; containers sleep after 5 idle minutes. To keep them up, run `bun run keepalive [--every 180] [--minutes 60]`. It pings `GET /v1/models` (free, no quota) and stops on its own after `--minutes`, because the containers bill about $10/h together while up. A keepalive started by Claude Code at 13:16 runs until about 14:16.
2. Live demo (about 45 s, deterministic, Kev answers from the recorded cache): `bun run demo`. Run it in a real terminal of at least 120 columns for the two-column dashboard.
3. To show live Kev network calls instead: delete or move `.kev/demo-cache.json` first. That takes about 60 s plus network time, and kev-27b makes the shift choices live.
4. Fallback if the endpoints are down: `bun run evolve --offline --pace 120`. That takes about 50 s, and shifts use the fixed fallback order (4 shifts instead of 3).
5. Afterwards, stop the Modal apps: `python -W ignore -m modal app stop kev-9b` and `kev-27b` (with `PYTHONUTF8=1`).

### Endpoint and cost notes:
- Self-hosted on Modal (workspace johnmannul): kev-9b on H100 at `KEV_9B_BASE_URL`, kev-27b on B200 at `KEV_27B_BASE_URL`. The hosted PRD gateway uses `KEV_API_KEY`.
- Containers bill about $3.95/h (9b) and $6.25/h (27b) while up. Today's local quota after the DoD runs: kev-9b 37+/1000, kev-27b 5+/200.
- Tests must never use `fromEnv(process.env)` or the default `FileQuotaStore`.

### Files Modified in Last Session:
- `src/ui/render.ts` (`renderTelemetry`, `sparkline`, `GPU_DOLLARS_PER_HOUR`, decision-source label, red/green lineage ids, header includes telemetry)
- `src/ui/state.ts` (`modelLatency`, `decisions`, `decisionMs`, `stalledCycles`, `lastShiftCycle`)
- `src/harness/runner.ts` (`kev_decision.modelLatencyMs`)
- `scripts/demo.ts` (`--pace 120`)
- `scripts/keepalive.ts` (new) and `package.json` (`keepalive` script). @Devin: add a mocked-fetch test that checks only self-hosted models are pinged and that it stops at `--minutes`.
- `tests/demo.test.ts` (pace assertions 140 -> 120)
- `.kev/demo-cache.json` (regenerated by the live run), `.kev/demo-cache.pre-dod.json.bak` (old cache, moved aside)
- `HANDOFF.md`
