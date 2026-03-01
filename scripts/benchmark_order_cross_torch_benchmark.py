#!/usr/bin/env python
import json
import math
import os
import statistics

import torch
from torch import nn
from torch.utils.benchmark import Timer

from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda


SHAPES = [
    {"name": "S", "batch": 64, "in": 256, "out": 256},
    {"name": "M", "batch": 32, "in": 512, "out": 512},
    {"name": "L", "batch": 16, "in": 1024, "out": 1024},
]

WORKLOADS = ["forward", "forward_backward", "train_step"]
ORDERS = [("cuda", "triton"), ("triton", "cuda")]

WARMUP_ITERS = 8
REPEATS = 4
MIN_RUN_TIME = 0.5


def make_runner(backend: str, shape: dict, workload: str):
    use_triton = backend == "triton"
    cfg = SingleRPUConfig(device=ConstantStepDevice(), use_triton=use_triton)

    batch = shape["batch"]
    in_f = shape["in"]
    out_f = shape["out"]

    model = AnalogLinear(in_f, out_f, rpu_config=cfg).cuda()
    opt = AnalogSGD(model.parameters(), lr=0.01)
    opt.regroup_param_groups(model)
    crit = nn.MSELoss()

    x = torch.randn(batch, in_f, device="cuda")
    y = torch.randn(batch, out_f, device="cuda")

    if workload == "forward":

        def _fn():
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize()

        return _fn, cfg.get_default_tile_module_class().__name__

    if workload == "forward_backward":

        def _fn():
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            torch.cuda.synchronize()

        return _fn, cfg.get_default_tile_module_class().__name__

    def _fn():
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()

    return _fn, cfg.get_default_tile_module_class().__name__


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if not cuda.is_compiled():
        raise RuntimeError("aihwkit CUDA extension is not compiled")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    measurements = []

    for shape in SHAPES:
        for workload in WORKLOADS:
            for order in ORDERS:
                for repeat in range(REPEATS):
                    for backend in order:
                        fn, tile_class = make_runner(backend, shape, workload)

                        for _ in range(WARMUP_ITERS):
                            fn()

                        timer = Timer(stmt="fn()", globals={"fn": fn}, num_threads=1)
                        m = timer.blocked_autorange(min_run_time=MIN_RUN_TIME)

                        measurements.append(
                            {
                                "shape": shape["name"],
                                "batch": shape["batch"],
                                "in": shape["in"],
                                "out": shape["out"],
                                "workload": workload,
                                "order": f"{order[0]}_then_{order[1]}",
                                "repeat": repeat,
                                "backend": backend,
                                "tile_class": tile_class,
                                "median_s": float(m.median),
                                "iqr_s": float(m.iqr),
                                "number_per_run": int(m.number_per_run),
                            }
                        )

    summary = []
    for shape in ["S", "M", "L"]:
        for workload in WORKLOADS:
            for order in ["cuda_then_triton", "triton_then_cuda"]:
                cuda_rows = [
                    r
                    for r in measurements
                    if r["shape"] == shape
                    and r["workload"] == workload
                    and r["order"] == order
                    and r["backend"] == "cuda"
                ]
                triton_rows = [
                    r
                    for r in measurements
                    if r["shape"] == shape
                    and r["workload"] == workload
                    and r["order"] == order
                    and r["backend"] == "triton"
                ]

                c_med = statistics.median(r["median_s"] for r in cuda_rows)
                t_med = statistics.median(r["median_s"] for r in triton_rows)

                summary.append(
                    {
                        "shape": shape,
                        "workload": workload,
                        "order": order,
                        "cuda_median_ms": c_med * 1e3,
                        "triton_median_ms": t_med * 1e3,
                        "speedup_x": c_med / t_med if t_med > 0 else math.nan,
                        "delta_pct": ((c_med - t_med) / c_med * 100.0)
                        if c_med > 0
                        else math.nan,
                    }
                )

    order_bias = []
    for shape in ["S", "M", "L"]:
        for workload in WORKLOADS:
            row_ct = next(
                r
                for r in summary
                if r["shape"] == shape
                and r["workload"] == workload
                and r["order"] == "cuda_then_triton"
            )
            row_tc = next(
                r
                for r in summary
                if r["shape"] == shape
                and r["workload"] == workload
                and r["order"] == "triton_then_cuda"
            )

            order_bias.append(
                {
                    "shape": shape,
                    "workload": workload,
                    "speedup_cuda_then_triton_x": row_ct["speedup_x"],
                    "speedup_triton_then_cuda_x": row_tc["speedup_x"],
                    "speedup_order_bias_pct": (
                        (row_ct["speedup_x"] - row_tc["speedup_x"])
                        / row_tc["speedup_x"]
                        * 100.0
                    )
                    if row_tc["speedup_x"] > 0
                    else math.nan,
                }
            )

    out = {
        "env": {
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "aihwkit_cuda_compiled": bool(cuda.is_compiled()),
            "warmup_iters": WARMUP_ITERS,
            "repeats": REPEATS,
            "min_run_time": MIN_RUN_TIME,
        },
        "measurements": measurements,
        "summary": summary,
        "order_bias": order_bias,
    }

    out_path = "/home/sinnce/workspace/aihwkit/.sisyphus/evidence/triton_vs_cuda_order_cross.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("WROTE", out_path)
    for row in summary:
        print(
            f"{row['shape']} {row['workload']:<16} {row['order']:<17} "
            f"cuda={row['cuda_median_ms']:.3f}ms triton={row['triton_median_ms']:.3f}ms "
            f"speedup={row['speedup_x']:.3f}x delta={row['delta_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
