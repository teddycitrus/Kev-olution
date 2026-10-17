# Agent Notes

## Workflow rules
- Update `HANDOFF.md` at the end of every turn, in the PRD format (checklist, What Was Just Done, Actionable Next Task, Files Modified). No exceptions, including setup or ops-only turns.
- Claude Code (Terminal 1) and Devin (Terminal 2) share this workspace. Re-read `HANDOFF.md` before starting and before writing it; the other agent may have changed it.
- When Devin is not actively writing code, check `HANDOFF.md` every 4 minutes for new tasks assigned to @Devin and act on them.

## Commands
- `bun test`: all suites. `bun run typecheck`: `tsc --noEmit`.
- Python is `python` (3.14), not `python3`. Run Modal CLI as `python -W ignore -m modal ...` with `PYTHONUTF8=1`, otherwise Windows' cp1252 console crashes on Unicode output.

## Kev endpoints (self-hosted on Modal, workspace johnmannul)
- kev-9b: `KEV_9B_BASE_URL` / `KEV_9B_API_KEY` in `.env` (H100). kev-27b: `KEV_27B_BASE_URL` / `KEV_27B_API_KEY` (B200).
- `KEV_API_KEY` is the separate hosted demo endpoint from the PRD.
- Deploy script: `.devin/skills/kev-deploy/scripts/kev_serve.py` (`KEV_MODEL`, `KEV_APP_NAME`, `KEV_API_KEY` at deploy time).
- Both scale to zero after 5 idle minutes; cold start about 3.5 min (9b) and 6.5 min (27b). Take down with `modal app stop kev-9b` / `kev-27b`.
