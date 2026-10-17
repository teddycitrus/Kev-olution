import { expect, test } from "bun:test";
import { REHEARSAL_ARGS, rehearsalArgs } from "../scripts/demoRehearsal";
import { parseArgs } from "../src/index";

test("rehearsal is offline, seed 0, paced for ~45 s, and accepts extra flags", () => {
  expect(parseArgs(REHEARSAL_ARGS)).toMatchObject({ offline: true, seed: 0, paceMs: 107, demo: false });
  expect(parseArgs(rehearsalArgs(["--plain"])).plain).toBe(true);
});

test("418.7 simulated GPU-seconds at --pace 107 lands near 45 s", () => {
  expect((418.7 * 107) / 1000).toBeCloseTo(44.8, 1);
});

test("the demo entrypoint flags parse", () => {
  expect(parseArgs(["--demo", "--benchmark", "triton_flash_attention"])).toMatchObject({ demo: true, benchmark: "triton_flash_attention" });
});
