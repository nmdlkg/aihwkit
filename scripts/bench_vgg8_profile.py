#!/usr/bin/env python

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor, nn
from torchvision import datasets, transforms

from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.presets import GokmenVlasovPreset

PATH_DATASET = os.path.join("data", "DATASET")
N_CLASSES = 10
WEIGHT_SCALING_OMEGA = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile VGG8 SVHN training: CUDA backend vs Triton backend "
            "(approx and binomial update)"
        )
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader worker processes"
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=3,
        help="Warmup batches before each measured run",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="If > 0, limit train batches per epoch for quick runs",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="If > 0, limit validation batches during evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for both CUDA and Triton runs",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="/tmp/aihwkit_bench/vgg8_profile.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def load_images() -> Tuple[Any, Any]:
    mean = Tensor([0.4377, 0.4438, 0.4728])
    std = Tensor([0.1980, 0.2010, 0.1970])
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    train_set = datasets.SVHN(
        PATH_DATASET, download=True, split="train", transform=transform
    )
    val_set = datasets.SVHN(
        PATH_DATASET, download=True, split="test", transform=transform
    )

    return train_set, val_set


def create_dataloaders(
    train_set: Any, val_set: Any, batch_size: int, num_workers: int, seed: int
) -> Tuple[Any, Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_data = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_data = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_data, val_data


def create_rpu_config(use_triton: bool, use_triton_update: bool):
    mapping = MappingParameter(weight_scaling_omega=WEIGHT_SCALING_OMEGA)
    rpu_config = GokmenVlasovPreset(mapping=mapping)
    rpu_config.runtime.offload_gradient = True
    rpu_config.runtime.offload_input = True
    rpu_config.use_triton = use_triton
    rpu_config.use_triton_update = use_triton_update
    return rpu_config


def create_model(
    use_triton: bool, use_triton_update: bool, device: torch.device
) -> AnalogSequential:
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    rpu_config = create_rpu_config(use_triton, use_triton_update)

    model = AnalogSequential(
        nn.Conv2d(
            in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[0],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=rpu_config,
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=rpu_config,
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[1],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=rpu_config,
        ),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(
            in_channels=channel[1],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=rpu_config,
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[2],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=rpu_config,
        ),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        AnalogLinear(
            in_features=16 * channel[2], out_features=fc_size, rpu_config=rpu_config
        ),
        nn.ReLU(),
        nn.Linear(in_features=fc_size, out_features=N_CLASSES),
        nn.LogSoftmax(dim=1),
    )
    model.to(device)
    return model


def run_warmup(
    model: AnalogSequential,
    train_data: Any,
    device: torch.device,
    learning_rate: float,
    warmup_batches: int,
) -> None:
    criterion = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    model.train()
    for batch_idx, (images, labels) in enumerate(train_data):
        if batch_idx >= warmup_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()


def train_profiled(
    model: AnalogSequential,
    train_data: Any,
    device: torch.device,
    learning_rate: float,
    epochs: int,
    max_train_batches: int,
) -> Tuple[float, List[Dict[str, float]]]:
    criterion = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    epoch_results: List[Dict[str, float]] = []
    t_total_start = time.perf_counter()

    for epoch in range(epochs):
        phases = defaultdict(float)
        total_loss = 0.0
        num_batches = 0
        model.train()

        torch.cuda.synchronize()
        t_ep_start = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(train_data):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            phases["data"] += t1 - t0

            optimizer.zero_grad()
            t2 = time.perf_counter()
            output = model(images)
            loss = criterion(output, labels)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            phases["forward"] += t3 - t2

            t4 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            phases["backward"] += t5 - t4

            t6 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            t7 = time.perf_counter()
            phases["opt_step"] += t7 - t6

            total_loss += loss.item()
            num_batches += 1

        torch.cuda.synchronize()
        t_ep_end = time.perf_counter()
        ep_time = t_ep_end - t_ep_start
        avg_loss = total_loss / num_batches if num_batches > 0 else float("nan")

        epoch_results.append(
            {
                "epoch": float(epoch),
                "time_s": ep_time,
                "loss": avg_loss,
                "data_s": phases["data"],
                "forward_s": phases["forward"],
                "backward_s": phases["backward"],
                "opt_step_s": phases["opt_step"],
                "num_batches": float(num_batches),
            }
        )

        pct = lambda value: (value / ep_time * 100.0) if ep_time > 0 else 0.0
        print(
            f"  Epoch {epoch}: {ep_time:.3f}s | "
            f"data={phases['data']:.3f}s({pct(phases['data']):.0f}%) "
            f"fwd={phases['forward']:.3f}s({pct(phases['forward']):.0f}%) "
            f"bwd={phases['backward']:.3f}s({pct(phases['backward']):.0f}%) "
            f"opt={phases['opt_step']:.3f}s({pct(phases['opt_step']):.0f}%) "
            f"loss={avg_loss:.4f} batches={num_batches}"
        )

    total_time = time.perf_counter() - t_total_start
    return total_time, epoch_results


def evaluate(
    model: AnalogSequential,
    val_data: Any,
    device: torch.device,
    max_val_batches: int,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_data):
            if max_val_batches > 0 and batch_idx >= max_val_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(images)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0
    return correct / total


def average_epochs(
    epoch_results: List[Dict[str, float]], skip_first: bool
) -> Dict[str, float]:
    start_idx = 1 if skip_first and len(epoch_results) > 1 else 0
    selected = epoch_results[start_idx:]
    count = len(selected)
    keys = ["data_s", "forward_s", "backward_s", "opt_step_s", "time_s"]

    if count == 0:
        return {key: float("nan") for key in keys}

    return {key: sum(ep[key] for ep in selected) / count for key in keys}


def assert_backend(name: str, use_triton: bool, model: AnalogSequential) -> str:
    tiles = list(model.analog_tiles())
    if not tiles:
        raise RuntimeError(f"{name}: no analog tiles found in model")

    tile_class = type(tiles[0]).__name__
    if use_triton and "Triton" not in tile_class:
        raise RuntimeError(
            f"{name}: Triton backend requested but tile class is {tile_class}. "
            "Ensure Triton is installed and CUDA is available."
        )
    if not use_triton and "Triton" in tile_class:
        raise RuntimeError(
            f"{name}: CUDA backend requested but Triton tile class was selected: {tile_class}"
        )
    return tile_class


def run_one(
    name: str,
    use_triton: bool,
    use_triton_update: bool,
    train_set: Any,
    val_set: Any,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(f"  {name}")
    print(f"{'=' * 72}")
    print(f"  use_triton={use_triton}, use_triton_update={use_triton_update}")

    train_data, val_data = create_dataloaders(
        train_set=train_set,
        val_set=val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    warmup_model = create_model(use_triton, use_triton_update, device)
    warmup_tile_class = assert_backend(name + " warmup", use_triton, warmup_model)
    run_warmup(
        model=warmup_model,
        train_data=train_data,
        device=device,
        learning_rate=args.learning_rate,
        warmup_batches=args.warmup_batches,
    )
    del warmup_model
    gc.collect()
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = create_model(use_triton, use_triton_update, device)
    tile_class = assert_backend(name, use_triton, model)
    print(f"  Tile: {tile_class}")
    print(f"  Warmup tile: {warmup_tile_class}")

    total_time, epoch_results = train_profiled(
        model=model,
        train_data=train_data,
        device=device,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
    )
    accuracy = evaluate(
        model=model,
        val_data=val_data,
        device=device,
        max_val_batches=args.max_val_batches,
    )
    print(f"  Total: {total_time:.3f}s, Accuracy: {accuracy * 100:.2f}%")

    avg_skip0 = average_epochs(epoch_results, skip_first=True)
    avg_all = average_epochs(epoch_results, skip_first=False)

    if args.epochs > 1:
        print(f"\n  Avg (epoch 1-{args.epochs - 1}):")
        print(
            f"    epoch={avg_skip0['time_s']:.3f}s | data={avg_skip0['data_s']:.3f}s "
            f"fwd={avg_skip0['forward_s']:.3f}s bwd={avg_skip0['backward_s']:.3f}s "
            f"opt={avg_skip0['opt_step_s']:.3f}s"
        )
    else:
        print("\n  Avg (all epochs):")
        print(
            f"    epoch={avg_all['time_s']:.3f}s | data={avg_all['data_s']:.3f}s "
            f"fwd={avg_all['forward_s']:.3f}s bwd={avg_all['backward_s']:.3f}s "
            f"opt={avg_all['opt_step_s']:.3f}s"
        )

    result = {
        "name": name,
        "use_triton": use_triton,
        "use_triton_update": use_triton_update,
        "tile_class": tile_class,
        "total_time_s": total_time,
        "accuracy": accuracy,
        "epochs": epoch_results,
        "avg_epoch_1_plus": avg_skip0,
        "avg_all_epochs": avg_all,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_comparison(
    base_result: Dict[str, Any], other_result: Dict[str, Any], use_skip0: bool
):
    key = "avg_epoch_1_plus" if use_skip0 else "avg_all_epochs"
    ca = base_result[key]
    ta = other_result[key]

    print(f"\n{'=' * 72}")
    if use_skip0:
        print("  PHASE-BY-PHASE COMPARISON (avg epoch 1+)")
    else:
        print("  PHASE-BY-PHASE COMPARISON (avg all epochs)")
    print(f"{'=' * 72}")
    base_title = base_result["name"][:10]
    other_title = other_result["name"][:12]
    print(
        f"  {'Phase':<12} {base_title + ' (ms)':>10} {other_title + ' (ms)':>12} "
        f"{'Speedup':>8} {'% of base':>10} {'% of other':>12}"
    )
    print(f"  {'-' * 68}")

    for phase_key, label in [
        ("data_s", "data"),
        ("forward_s", "forward"),
        ("backward_s", "backward"),
        ("opt_step_s", "opt_step"),
        ("time_s", "TOTAL"),
    ]:
        cuda_ms = ca[phase_key] * 1000.0
        triton_ms = ta[phase_key] * 1000.0
        speedup = cuda_ms / triton_ms if triton_ms > 0 else float("inf")
        cuda_share = ca[phase_key] / ca["time_s"] * 100.0 if ca["time_s"] > 0 else 0.0
        triton_share = ta[phase_key] / ta["time_s"] * 100.0 if ta["time_s"] > 0 else 0.0
        print(
            f"  {label:<12} {cuda_ms:>10.1f} {triton_ms:>12.1f} {speedup:>7.3f}x "
            f"{cuda_share:>9.1f}% {triton_share:>11.1f}%"
        )

    print(f"\n  {base_result['name']} accuracy:  {base_result['accuracy'] * 100:.2f}%")
    print(f"  {other_result['name']} accuracy: {other_result['accuracy'] * 100:.2f}%")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    print(
        f"GPU: {gpu_name} | PyTorch: {torch.__version__} | "
        f"CUDA runtime: {torch.version.cuda}"
    )
    print(
        "Model: VGG8-inspired SVHN network | "
        f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.learning_rate}"
    )
    print(
        "Limits: "
        f"max_train_batches={args.max_train_batches}, "
        f"max_val_batches={args.max_val_batches}, warmup_batches={args.warmup_batches}"
    )

    train_set, val_set = load_images()

    _ = torch.randn(128, 128, device=device) @ torch.randn(128, 128, device=device)
    torch.cuda.synchronize()

    cuda_result = run_one(
        name="CUDA Backend",
        use_triton=False,
        use_triton_update=False,
        train_set=train_set,
        val_set=val_set,
        device=device,
        args=args,
    )
    triton_approx_result = run_one(
        name="Triton Approx Update",
        use_triton=True,
        use_triton_update=True,
        train_set=train_set,
        val_set=val_set,
        device=device,
        args=args,
    )
    triton_naive_result = run_one(
        name="Triton Binomial Update",
        use_triton=True,
        use_triton_update=False,
        train_set=train_set,
        val_set=val_set,
        device=device,
        args=args,
    )

    print_comparison(cuda_result, triton_approx_result, use_skip0=args.epochs > 1)
    print_comparison(cuda_result, triton_naive_result, use_skip0=args.epochs > 1)
    print_comparison(
        triton_approx_result, triton_naive_result, use_skip0=args.epochs > 1
    )

    output = {
        "env": {
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "timestamp": datetime.now().isoformat(),
        },
        "args": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "warmup_batches": args.warmup_batches,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
            "seed": args.seed,
        },
        "cuda": cuda_result,
        "triton": triton_approx_result,
        "triton_approx": triton_approx_result,
        "triton_naive": triton_naive_result,
    }

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
