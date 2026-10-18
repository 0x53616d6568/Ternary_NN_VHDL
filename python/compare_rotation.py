#!/usr/bin/env python3
"""Cross-check the VHDL Walsh-Hadamard rotation against the trained model.

Writes the same binary images for the ModelSim testbench and then compares the
rotated bus that hardware produced with what prepare_inputs computes in Python.
The two must agree exactly: the trained weights are only meaningful if the FPGA
applies a bit-identical transform to the pixels.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate_checkpoints import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--vector-output", default="rot_images.txt")
    parser.add_argument("--reference-output", default="rot_py.txt")
    parser.add_argument("--hardware-output", default="rot_hw.txt")
    args = parser.parse_args()

    model, checkpoint = load_model(args.checkpoint, torch.device("cpu"))
    if model.hidden_layers != 1:
        raise SystemExit("Expected a single hidden layer.")
    if model.fc1.walsh_block <= 1:
        raise SystemExit("Checkpoint has no rotation; nothing to cross-check.")
    if not torch.equal(model.prepare_inputs(torch.zeros(2, 784)),
                       torch.zeros(2, 784)):
        pass  # zero image is not informative; real comparison happens below

    loader = DataLoader(datasets.MNIST(args.data_dir, train=False, download=True,
                                      transform=transforms.ToTensor()), batch_size=8)

    images = []
    for batch, _ in loader:
        images.append(batch)
        if len(images) * 8 >= args.images:
            break
    images = torch.cat(images)[:args.images]

    with open(args.vector_output, "w", encoding="ascii", newline="\n") as handle:
        for image in images:
            bits = (image.reshape(-1) >= 0.5).to(torch.uint8).tolist()
            handle.write(" ".join(
                f"{sum(int(bits[start + b]) << b for b in range(8)):02X}"
                for start in range(0, 784, 8)) + "\n")

    with torch.no_grad():
        rotated = model.prepare_inputs(images)

    with open(args.reference_output, "w", encoding="ascii", newline="\n") as handle:
        for row in rotated:
            values = [int(v) for v in row.tolist()]
            for start in range(0, len(values), 16):
                handle.write(" ".join(str(v) for v in values[start:start + 16]) + "\n")

    print(f"Wrote {args.vector_output} ({args.images} images)")
    print(f"Wrote {args.reference_output} (Python reference)")

    hardware_path = Path(args.hardware_output)
    if not hardware_path.exists():
        print(f"{args.hardware_output} not present yet; run the ModelSim testbench first.")
        return

    hw = [[int(tok) for tok in line.split()]
          for line in hardware_path.read_text(encoding="ascii").splitlines() if line.strip()]
    ref = [[int(tok) for tok in line.split()]
           for line in Path(args.reference_output).read_text(encoding="ascii").splitlines()
           if line.strip()]

    if len(hw) != len(ref):
        print(f"FAIL: hardware produced {len(hw)} lines, Python produced {len(ref)}")
        return

    mismatches = sum(1 for a, b in zip(hw, ref) if a != b)
    total_values = sum(len(r) for r in ref)
    bad_values = sum(abs(x - y) for ra, rb in zip(hw, ref) for x, y in zip(ra, rb))

    print(f"\nLines compared      : {len(ref)}")
    print(f"Values compared     : {total_values}")
    print(f"Mismatched lines    : {mismatches}")
    print(f"Total absolute error: {bad_values}")
    if mismatches == 0:
        print("\nOK: hardware rotation is bit-identical to the Python model.")
    else:
        print("\nFAIL: hardware rotation differs from the Python model.")


if __name__ == "__main__":
    main()