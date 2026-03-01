#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MNIST e2e with per-phase profiling to identify bottleneck.

Network: 784→512→512→512→10 (4 AnalogLinear layers), batch=128, 10 epochs.
Breaks down: data_load, forward, backward, opt_step per batch.
"""

import os, sys, time, json, gc, math
from datetime import datetime
from collections import defaultdict

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice
from aihwkit.simulator.rpu_base import cuda

DEVICE = torch.device("cuda")
PATH_DATASET = os.path.join("data", "DATASET")
INPUT_SIZE = 784
HIDDEN_SIZES = [512, 512, 512]
OUTPUT_SIZE = 10
EPOCHS = 10
BATCH_SIZE = 128


def load_images():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        PATH_DATASET, download=True, train=True, transform=transform
    )
    val_set = datasets.MNIST(
        PATH_DATASET, download=True, train=False, transform=transform
    )
    train_data = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_data = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    return train_data, val_data


def create_model(use_triton):
    rpu = SingleRPUConfig(device=ConstantStepDevice(), use_triton=use_triton, use_triton_update=use_triton)
    layers = [
        AnalogLinear(INPUT_SIZE, HIDDEN_SIZES[0], True, rpu_config=rpu),
        nn.Sigmoid(),
    ]
    for i in range(len(HIDDEN_SIZES) - 1):
        layers.append(
            AnalogLinear(HIDDEN_SIZES[i], HIDDEN_SIZES[i + 1], True, rpu_config=rpu)
        )
        layers.append(nn.Sigmoid())
    layers.append(AnalogLinear(HIDDEN_SIZES[-1], OUTPUT_SIZE, True, rpu_config=rpu))
    layers.append(nn.LogSoftmax(dim=1))
    model = AnalogSequential(*layers)
    model.to(DEVICE)
    return model


def train_profiled(model, train_data):
    classifier = nn.NLLLoss()
    optimizer = AnalogSGD(model.parameters(), lr=0.05)
    optimizer.regroup_param_groups(model)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    epoch_results = []
    t_total_start = time.perf_counter()

    for epoch in range(EPOCHS):
        phases = defaultdict(float)
        total_loss = 0
        model.train()

        t_ep_start = time.perf_counter()
        t_iter_start = time.perf_counter()

        for images, labels in train_data:
            # --- data transfer ---
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            images = images.view(images.shape[0], -1)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            phases["data"] += t1 - t0

            # --- forward ---
            optimizer.zero_grad()
            t2 = time.perf_counter()
            output = model(images)
            loss = classifier(output, labels)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            phases["forward"] += t3 - t2

            # --- backward ---
            t4 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t5 = time.perf_counter()
            phases["backward"] += t5 - t4

            # --- optimizer step (analog update) ---
            t6 = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            t7 = time.perf_counter()
            phases["opt_step"] += t7 - t6

            total_loss += loss.item()

        torch.cuda.synchronize()
        t_ep_end = time.perf_counter()
        ep_time = t_ep_end - t_ep_start
        avg_loss = total_loss / len(train_data)

        epoch_results.append(
            {
                "epoch": epoch,
                "time_s": ep_time,
                "loss": avg_loss,
                "data_s": phases["data"],
                "forward_s": phases["forward"],
                "backward_s": phases["backward"],
                "opt_step_s": phases["opt_step"],
            }
        )

        pct = lambda v: v / ep_time * 100
        print(
            f"  Epoch {epoch}: {ep_time:.3f}s | "
            f"data={phases['data']:.3f}s({pct(phases['data']):.0f}%) "
            f"fwd={phases['forward']:.3f}s({pct(phases['forward']):.0f}%) "
            f"bwd={phases['backward']:.3f}s({pct(phases['backward']):.0f}%) "
            f"opt={phases['opt_step']:.3f}s({pct(phases['opt_step']):.0f}%) "
            f"loss={avg_loss:.4f}"
        )

        scheduler.step()

    total_time = time.perf_counter() - t_total_start
    return total_time, epoch_results


def evaluate(model, val_data):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(DEVICE, non_blocking=True).view(images.shape[0], -1)
            labels = labels.to(DEVICE, non_blocking=True)
            pred = model(images)
            _, predicted = torch.max(pred, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def run_one(name, use_triton, train_data, val_data):
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")

    # JIT warmup
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    warmup_model = create_model(use_triton)
    warmup_opt = AnalogSGD(warmup_model.parameters(), lr=0.05)
    warmup_opt.regroup_param_groups(warmup_model)
    crit = nn.NLLLoss()
    for i, (img, lbl) in enumerate(train_data):
        if i >= 3:
            break
        img = img.to(DEVICE).view(img.shape[0], -1)
        lbl = lbl.to(DEVICE)
        warmup_opt.zero_grad()
        loss = crit(warmup_model(img), lbl)
        loss.backward()
        warmup_opt.step()
    torch.cuda.synchronize()
    del warmup_model, warmup_opt, crit
    gc.collect()
    torch.cuda.empty_cache()

    # Real run
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = create_model(use_triton)
    tiles = list(model.analog_tiles())
    print(f"  Tile: {type(tiles[0]).__name__}, Layers: {len(tiles)} analog tiles")
    arch = " → ".join(
        [f"{INPUT_SIZE}"] + [str(h) for h in HIDDEN_SIZES] + [f"{OUTPUT_SIZE}"]
    )
    print(f"  Arch: {arch}")

    total_time, epoch_results = train_profiled(model, train_data)
    accuracy = evaluate(model, val_data)
    print(f"  Total: {total_time:.3f}s, Accuracy: {accuracy * 100:.2f}%")

    # Averages (skip epoch 0 for fair amortization)
    skip = 1
    avg = {}
    for key in ["data_s", "forward_s", "backward_s", "opt_step_s", "time_s"]:
        avg[key] = sum(e[key] for e in epoch_results[skip:]) / (EPOCHS - skip)

    print(f"\n  Avg (epoch 1-{EPOCHS - 1}):")
    print(
        f"    epoch={avg['time_s']:.3f}s | data={avg['data_s']:.3f}s "
        f"fwd={avg['forward_s']:.3f}s bwd={avg['backward_s']:.3f}s opt={avg['opt_step_s']:.3f}s"
    )

    result = {
        "name": name,
        "use_triton": use_triton,
        "tile_class": type(tiles[0]).__name__,
        "total_time_s": total_time,
        "accuracy": accuracy,
        "epochs": epoch_results,
        "avg_epoch_1_9": avg,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    gpu = torch.cuda.get_device_name(0)
    print(
        f"GPU: {gpu} | PyTorch: {torch.__version__} | CUDA compiled: {cuda.is_compiled()}"
    )
    print(
        f"Network: {INPUT_SIZE} → {' → '.join(str(h) for h in HIDDEN_SIZES)} → {OUTPUT_SIZE}"
    )
    print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")

    train_data, val_data = load_images()
    _ = torch.randn(128, 784, device=DEVICE) @ torch.randn(784, 512, device=DEVICE)
    torch.cuda.synchronize()

    cuda_r = run_one("CUDA Backend", False, train_data, val_data)
    triton_r = run_one("Triton Backend", True, train_data, val_data)

    # Comparison
    print(f"\n{'=' * 70}")
    print("  PHASE-BY-PHASE COMPARISON (avg epoch 1-9)")
    print(f"{'=' * 70}")
    ca, ta = cuda_r["avg_epoch_1_9"], triton_r["avg_epoch_1_9"]
    print(
        f"  {'Phase':<12} {'CUDA (ms)':>10} {'Triton (ms)':>12} {'Speedup':>8} {'% of CUDA':>10} {'% of Triton':>12}"
    )
    print(f"  {'-' * 66}")
    for key, label in [
        ("data_s", "data"),
        ("forward_s", "forward"),
        ("backward_s", "backward"),
        ("opt_step_s", "opt_step"),
        ("time_s", "TOTAL"),
    ]:
        c, t = ca[key] * 1000, ta[key] * 1000
        sp = c / t if t > 0 else float("inf")
        cp = ca[key] / ca["time_s"] * 100
        tp = ta[key] / ta["time_s"] * 100
        print(
            f"  {label:<12} {c:>10.1f} {t:>12.1f} {sp:>7.3f}x {cp:>9.1f}% {tp:>11.1f}%"
        )

    print(f"\n  CUDA accuracy:  {cuda_r['accuracy'] * 100:.2f}%")
    print(f"  Triton accuracy: {triton_r['accuracy'] * 100:.2f}%")

    out = {
        "env": {
            "gpu": gpu,
            "torch": torch.__version__,
            "timestamp": datetime.now().isoformat(),
        },
        "cuda": cuda_r,
        "triton": triton_r,
    }
    path = "/tmp/aihwkit_bench/mnist_profile.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
