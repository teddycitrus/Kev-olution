"""Hardware harness for evaluator Tiers 2 and 3 (see src/harness/runners.ts).

Loads a candidate kernel module, runs its attention(q, k, v, sm_scale) on a CUDA
device, compares against the PyTorch reference, and times it with
triton.testing.do_bench. Prints exactly one JSON object on the last stdout line:

  {"issues": [...], "tflops": float, "latency_ms": float, "max_abs_error": float}

Issue strings are prefixed with their kind: compile, runtime, or numerical.
"""

import argparse
import importlib.util
import json
import math


def emit(**fields):
    fields.setdefault("issues", [])
    print(json.dumps(fields), flush=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(exc):
    name = type(exc).__name__
    return "compile" if ("Compil" in name or "OutOfResources" in name) else "runtime"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--problem", required=True, help="JSON ProblemShape")
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--tier", type=int, choices=(2, 3), required=True)
    args = parser.parse_args()

    try:
        import torch
        from triton.testing import do_bench
    except ImportError as exc:
        return emit(issues=[f"runtime: {exc}"])
    if not torch.cuda.is_available():
        return emit(issues=["runtime: CUDA device not available"])

    problem = json.loads(args.problem)
    batch, heads, seq_len, head_dim = (problem[k] for k in ("batch", "heads", "seqLen", "headDim"))
    dtype = getattr(torch, problem["dtype"])
    if problem.get("causal"):
        return emit(issues=["runtime: causal reference not implemented in harness"])

    try:
        candidate = load_module(args.candidate, "candidate")
        reference = load_module(args.reference, "reference")
    except Exception as exc:  # noqa: BLE001 - report any import-time failure
        return emit(issues=[f"compile: {type(exc).__name__}: {str(exc)[:200]}"])

    torch.manual_seed(0)
    shape = (batch, heads, seq_len, head_dim)
    q, k, v = (torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3))
    sm_scale = 1.0 / math.sqrt(head_dim)

    try:
        out = candidate.attention(q, k, v, sm_scale)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return emit(issues=[f"{classify(exc)}: {type(exc).__name__}: {str(exc)[:200]}"])

    expected = reference.attention_reference(q, k, v, sm_scale)
    err = (out.float() - expected.float()).abs().max().item()
    if not math.isfinite(err) or err > args.tolerance:
        return emit(issues=[f"numerical: max abs error {err} exceeds tolerance {args.tolerance}"], max_abs_error=err)

    warmup, rep = (5, 20) if args.tier == 2 else (25, 100)
    ms = do_bench(lambda: candidate.attention(q, k, v, sm_scale), warmup=warmup, rep=rep)
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    emit(tflops=flops / (ms * 1e-3) / 1e12, latency_ms=ms, max_abs_error=err)


if __name__ == "__main__":
    main()
