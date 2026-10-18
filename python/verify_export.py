#!/usr/bin/env python3
"""Check that the exported integer model matches the trained float model.

The FPGA only sees the integers written into ModelWeights.vhd: a ternary
rotated W1, integer biases, and an 8-bit W2. This script replays inference
using only those integers, with the same fixed-point arithmetic the hardware
will perform, and reports how often the arg-max agrees with the float model.

An agreement rate below 100% means the export is lossy enough to change the
answer, and the hardware would not be running the model that was trained.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate_checkpoints import load_model


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--parameters", required=True, help="NPZ from export_vhdl_model.py")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _ = load_model(args.checkpoint, device)
    with np.load(args.parameters) as data:
        W1 = torch.from_numpy(data["W1"].astype(np.float32))
        B1 = torch.from_numpy(data["B1"].astype(np.float32))
        W2 = torch.from_numpy(data["W2"].astype(np.float32))
        B2 = torch.from_numpy(data["B2"].astype(np.float32))

    loader = DataLoader(datasets.MNIST(args.data_dir, train=False, download=True,
                                      transform=transforms.ToTensor()), batch_size=500)
    agree = 0
    total = 0
    agree_label = 0
    for images, labels in loader:
        if args.limit and total >= args.limit:
            break
        x = model.prepare_inputs(images)                 # rotation, as the FPGA will do
        hidden = torch.relu(x @ W1.T + B1).clamp(0.0, 127.0)
        hidden = torch.round(hidden)                      # VHDL stores integers
        scores = hidden @ W2.T + B2                       # VHDL accumulates in integers
        hardware = scores.argmax(1)
        software = model(images).argmax(1)
        agree += int((hardware == software).sum())
        agree_label += int((hardware == labels).sum())
        total += labels.numel()

    accuracy = 100.0 * agree / total
    print(f"Samples compared        : {total}")
    print(f"Integer vs float match  : {agree}/{total}  ({accuracy:.2f}%)")
    print(f"Integer accuracy vs MNIST: {100.0 * agree_label / total:.2f}%")
    if accuracy < 100.0:
        print("\nWARNING: the exported integers do not reproduce the trained model.")
        print("The hardware would not be running the model that was evaluated.")
    else:
        print("\nOK: the exported integers reproduce the trained model exactly.")


if __name__ == "__main__":
    main()