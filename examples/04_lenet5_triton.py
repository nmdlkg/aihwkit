# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""aihwkit example: LeNet5 with Triton backend.

Trains a LeNet5-inspired analog CNN on MNIST using the Triton backend
with SingleRPUConfig(device=ConstantStepDevice()).

Run with:
    PYTHONPATH=src AIHWKIT_USE_TRITON=1 python examples/04_lenet5_triton.py
"""

import os
from datetime import datetime

import torch
from torch import nn
from torchvision import datasets, transforms

from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import ConstantStepDevice

# Device setup
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Using device: {DEVICE}")

# Paths
PATH_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "DATASET")

# Training hyperparameters
SEED = 1
N_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.01
N_CLASSES = 10

# RPU config: ConstantStep device (simpler than agad, works with Triton)
RPU_CONFIG = SingleRPUConfig(device=ConstantStepDevice())
print(f"RPU Config: {RPU_CONFIG}")


def load_images():
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    os.makedirs(PATH_DATASET, exist_ok=True)
    train_set = datasets.MNIST(
        PATH_DATASET, download=True, train=True, transform=transform
    )
    val_set = datasets.MNIST(
        PATH_DATASET, download=True, train=False, transform=transform
    )
    train_data = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    val_data = torch.utils.data.DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False
    )
    return train_data, val_data


def create_model():
    """Return a LeNet5-inspired analog model."""
    channel = [16, 32, 512, 128]
    model = AnalogSequential(
        AnalogConv2d(
            in_channels=1,
            out_channels=channel[0],
            kernel_size=5,
            stride=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=5,
            stride=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        AnalogLinear(
            in_features=channel[2], out_features=channel[3], rpu_config=RPU_CONFIG
        ),
        nn.Tanh(),
        AnalogLinear(
            in_features=channel[3], out_features=N_CLASSES, rpu_config=RPU_CONFIG
        ),
        nn.LogSoftmax(dim=1),
    )
    return model


def train_step(train_data, model, criterion, optimizer):
    """Train one epoch."""
    total_loss = 0.0
    model.train()
    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_data.dataset)


def evaluate(val_data, model, criterion):
    """Evaluate model on validation set."""
    total_loss = 0.0
    predicted_ok = 0
    total_images = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_data:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            pred = model(images)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
    epoch_loss = total_loss / len(val_data.dataset)
    accuracy = predicted_ok / total_images * 100
    return epoch_loss, accuracy


if __name__ == "__main__":
    torch.manual_seed(SEED)

    print(
        f"\n{datetime.now().time().replace(microsecond=0)} --- Loading MNIST dataset..."
    )
    train_data, val_data = load_images()

    print(f"{datetime.now().time().replace(microsecond=0)} --- Building model...")
    model = create_model()
    if USE_CUDA:
        model.cuda()
    print(model)

    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(model)
    criterion = nn.NLLLoss()

    print(
        f"\n{datetime.now().time().replace(microsecond=0)} --- Starting LeNet5 Triton training "
        f"({N_EPOCHS} epochs, batch={BATCH_SIZE}, lr={LEARNING_RATE})"
    )
    print("-" * 70)

    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(N_EPOCHS):
        train_loss = train_step(train_data, model, criterion, optimizer)
        val_loss, accuracy = evaluate(val_data, model, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        print(
            f"{datetime.now().time().replace(microsecond=0)} --- "
            f"Epoch {epoch:2d}/{N_EPOCHS - 1} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}%"
        )

    print("-" * 70)
    print(f"\n{datetime.now().time().replace(microsecond=0)} --- Training complete!")
    print(f"Final accuracy: {accuracies[-1]:.2f}%")
    print(f"Loss trend (train): {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
    converged = train_losses[-1] < train_losses[0]
    print(f"Loss decreased: {converged}")
    if converged:
        print("SUCCESS: LeNet5 converged with Triton backend!")
    else:
        print("WARNING: Loss did not decrease!")
