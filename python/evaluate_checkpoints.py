#!/usr/bin/env python3
"""Evaluate saved ternary checkpoints on the MNIST test set and report cost.

This is the single source of truth for every accuracy and cost number quoted
during optimisation. It reconstructs each model from the settings stored in its
own checkpoint, so it works for any hidden size, pooling factor and rotation.

Cost columns are the deployment figures the VHDL design has to hit:

* weights      : W1 + W2 parameter count
* bits/weight  : 1 + (1 - zero density), i.e. a presence bitmap plus a sign
                 vector, which is how the weights are actually packed
* memory       : packed weight bytes
* multiplies   : 0, because a ternary weight times an input is an add or a
                 subtract
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train_mnist_ternary_v3 import VHDLCompatibleMLP


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    hidden = checkpoint.get("hidden_units") or checkpoint["architecture"][1]
    model = VHDLCompatibleMLP(
        hidden,
        checkpoint.get("threshold_ratio", 0.7),
        gray_inputs=bool(checkpoint.get("gray_inputs", False)),
        walsh_block=int(checkpoint.get("walsh_block", 16)),
        layer2_precision=checkpoint.get("layer2_precision", "full"),
        alive_bias_init=float(checkpoint.get("alive_bias_init", 0.0)),
        pool_factor=int(checkpoint.get("pool_factor", 1)),
        hidden_layers=int(checkpoint.get("hidden_layers", 1)),
        normalise_rotation=bool(checkpoint.get("normalise_rotation", True)),
        rotation_shift=int(checkpoint.get("rotation_shift", 0)),
    )
    state = {k: v for k, v in checkpoint["model_state_dict"].items() if k != "transform"}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def evaluate(model, loader):
    correct = 0
    count = 0
    for images, labels in loader:
        correct += (model(images).argmax(1) == labels).sum().item()
        count += labels.numel()
    return correct / count


def cost_report(model, checkpoint):
    weight, bias = model.export_ternary_weights()
    input_size = model.input_size
    hidden = model.fc1.linear.out_features
    depth = model.hidden_layers
    w1 = int(weight.numel())
    # Extra hidden layers are H x H ternary matrices, the same encoding as W1.
    middle = int(sum(l.linear.weight.numel() for l in model.hidden_extra))
    w2 = int(model.fc2.linear.weight.numel())
    n_weights = w1 + middle + w2
    bits = model.fc1.effective_bits_per_weight()
    zeros = float((weight == 0).float().mean())
    shape = " -> ".join(str(v) for v in
                        [input_size] + [hidden] * depth + [10])
    return {
        "architecture": shape,
        "hidden_layers": depth,
        "weights": n_weights,
        "w1_weights": w1,
        "middle_weights": middle,
        "w2_weights": w2,
        "bits_per_weight": round(bits, 3),
        "zero_density_pct": round(100.0 * zeros, 2),
        "packed_bytes": int(round(n_weights * bits / 8)),
        "macs": n_weights,
        "multiplies": 0,
        "input_size": input_size,
        "pool_factor": model.pool_factor,
        "walsh_block": model.fc1.walsh_block,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(datasets.MNIST(args.data_dir, train=False, download=True,
                                      transform=transforms.ToTensor()), batch_size=1000)

    rows = []
    for path in args.checkpoints:
        model, checkpoint = load_model(path, device)
        accuracy = evaluate(model, loader)
        row = cost_report(model, checkpoint)
        row["checkpoint"] = Path(path).name
        row["test_accuracy_pct"] = round(100.0 * accuracy, 2)
        row["val_accuracy_pct"] = round(100.0 * float(checkpoint.get(
            "best_validation_accuracy", 0.0)), 2)
        rows.append(row)
        print(f"{row['checkpoint']:<34} {row['architecture']:>12}  "
              f"test={row['test_accuracy_pct']:6.2f}%  "
              f"weights={row['weights']:>7,}  "
              f"{row['bits_per_weight']:.2f} b/w  "
              f"{row['packed_bytes']/1024:7.2f} KB  "
              f"mult={row['multiplies']}")

    best = max(rows, key=lambda r: r["test_accuracy_pct"])
    print("\nBest:", best["checkpoint"], f"{best['test_accuracy_pct']:.2f}%",
          f"{best['architecture']}", f"{best['packed_bytes']/1024:.2f} KB")

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print("Wrote", args.json_output)


if __name__ == "__main__":
    main()