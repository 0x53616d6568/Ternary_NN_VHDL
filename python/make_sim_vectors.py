#!/usr/bin/env python3
"""Generate end-to-end test vectors for the full NumberRecognition simulation.

Each line carries the MNIST label, the prediction the trained model makes, and
the 98 packed image bytes. tb_NumberRecognition feeds the bytes through the real
VHDL and compares the digit it produces.

The image protocol is unchanged from the original design: 0xA5 then 98 bytes,
one bit per pixel, least significant bit first.
"""

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from evaluate_checkpoints import load_model


def packed_bytes(image):
    bits = (image.reshape(-1) >= 0.5).to(torch.uint8).tolist()
    if len(bits) != 784:
        raise ValueError(f"Expected 784 pixels, got {len(bits)}")
    return [sum(int(bits[start + b]) << b for b in range(8))
            for start in range(0, 784, 8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--output", default="sim_vectors.txt")
    args = parser.parse_args()

    model, _ = load_model(args.checkpoint, torch.device("cpu"))
    dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                             transform=transforms.ToTensor())
    if not 1 <= args.samples <= len(dataset):
        raise SystemExit(f"--samples must be between 1 and {len(dataset)}")

    with open(args.output, "w", encoding="ascii", newline="\n") as handle:
        for index in range(args.samples):
            image, label = dataset[index]
            with torch.no_grad():
                prediction = int(model(image.unsqueeze(0)).argmax(dim=1).item())
            hex_bytes = " ".join(f"{byte:02X}" for byte in packed_bytes(image))
            handle.write(f"{int(label)} {prediction} {hex_bytes}\n")

    print(f"Wrote {args.samples} vectors to {args.output}")
    print("Format: <label> <software prediction> <98 packed image bytes>")


if __name__ == "__main__":
    main()