import { describe, expect, test } from "bun:test";
import { parseArgs } from "../src/index";

describe("parseArgs", () => {
  test("defaults", () => {
    expect(parseArgs([])).toEqual({ benchmark: "triton_flash_attention", seed: 0, demo: false, offline: false, gpu: false, paceMs: 0, warmup: false, plain: false });
  });

  test("every flag", () => {
    expect(
      parseArgs(["--benchmark", "other_bench", "--cycles", "12", "--duration", "1.5", "--seed", "7", "--pace", "40", "--demo", "--offline", "--gpu", "--warmup", "--plain"]),
    ).toEqual({ benchmark: "other_bench", cycles: 12, durationMs: 1500, seed: 7, paceMs: 40, demo: true, offline: true, gpu: true, warmup: true, plain: true });
  });

  test("zero is accepted", () => {
    expect(parseArgs(["--cycles", "0", "--duration", "0"])).toMatchObject({ cycles: 0, durationMs: 0 });
  });

  test.each([["--verbose"], ["triton_flash_attention"], ["-c"]])("rejects unknown argument %s", (flag) => {
    expect(() => parseArgs([flag])).toThrow(`unknown argument ${flag}`);
  });

  test.each([
    ["--cycles", "-1"],
    ["--duration", "-0.5"],
    ["--seed", "-3"],
    ["--pace", "-10"],
  ])("rejects negative %s %s", (flag, value) => {
    expect(() => parseArgs([flag, value])).toThrow(`${flag} expects a non-negative number`);
  });

  test.each([["--cycles"], ["--seed"], ["--duration"], ["--pace"]])("rejects %s without a value", (flag) => {
    expect(() => parseArgs([flag])).toThrow("expects a non-negative number");
  });

  test.each([["--cycles", "ten"], ["--seed", "NaN"], ["--pace", "Infinity"]])("rejects non-numeric %s %s", (flag, value) => {
    expect(() => parseArgs([flag, value])).toThrow("expects a non-negative number");
  });
});
