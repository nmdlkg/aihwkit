#!/usr/bin/env python
import argparse
import json
import time

import torch
from torch import nn

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda


SHAPES = {
    "S": (64, 256, 256),
    "M": (32, 512, 512),
    "L": (16, 1024, 1024),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark latency and memory per backend"
    )
    parser.add_argument("--backend", choices=["cuda", "triton"], required=True)
    parser.add_argument(
        "--workload",
        choices=["forward", "forward_backward", "train_step"],
        required=True,
    )
    parser.add_argument("--shape", choices=sorted(SHAPES), default="M")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--nvtx", action="store_true")
    return parser.parse_args()


def make_step_fn(model, opt, crit, x, y, workload: str):
    if workload == "forward":

        def _fn() -> None:
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

        return _fn

    if workload == "forward_backward":

        def _fn() -> None:
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.cuda.synchronize()

        return _fn

    def _fn() -> None:
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    return _fn


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not cuda.is_compiled():
        raise RuntimeError("aihwkit CUDA backend is not compiled")

    use_triton = args.backend == "triton"
    batch, in_features, out_features = SHAPES[args.shape]

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    cfg = SingleRPUConfig(device=ConstantStepDevice(), use_triton=use_triton)
    tile_class = cfg.get_default_tile_module_class().__name__
    model = AnalogLinear(in_features, out_features, rpu_config=cfg).cuda()
    opt = AnalogSGD(model.parameters(), lr=0.01)
    opt.regroup_param_groups(model)
    crit = nn.MSELoss()

    x = torch.randn(batch, in_features, device="cuda")
    y = torch.randn(batch, out_features, device="cuda")

    step_fn = make_step_fn(model, opt, crit, x, y, args.workload)

    for _ in range(args.warmup):
        step_fn()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_alloc = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()

    iter_times_ms = []
    for _ in range(args.iters):
        if args.nvtx:
            torch.cuda.nvtx.range_push(f"{args.backend}_{args.workload}_{args.shape}")

        t0 = time.perf_counter()
        step_fn()
        t1 = time.perf_counter()
        iter_times_ms.append((t1 - t0) * 1000.0)

        if args.nvtx:
            torch.cuda.nvtx.range_pop()

    end_alloc = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    iter_times_sorted = sorted(iter_times_ms)
    median_ms = iter_times_sorted[len(iter_times_sorted) // 2]

    result = {
        "backend": args.backend,
        "tile_class": tile_class,
        "workload": args.workload,
        "shape": args.shape,
        "batch": batch,
        "in_features": in_features,
        "out_features": out_features,
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_median_ms": median_ms,
        "latency_mean_ms": sum(iter_times_ms) / len(iter_times_ms),
        "latency_min_ms": min(iter_times_ms),
        "latency_max_ms": max(iter_times_ms),
        "memory_start_alloc_bytes": int(start_alloc),
        "memory_end_alloc_bytes": int(end_alloc),
        "memory_start_reserved_bytes": int(start_reserved),
        "memory_end_reserved_bytes": int(end_reserved),
        "memory_peak_alloc_bytes": int(peak_alloc),
        "memory_peak_reserved_bytes": int(peak_reserved),
    }

    payload = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
