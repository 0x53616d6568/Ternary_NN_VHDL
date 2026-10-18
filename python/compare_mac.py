"""Compare the MAC engine's RTL output against the trained model.

tb_TernaryMacEngine drives real MNIST images through TernaryMacEngine and writes
the resulting integer hidden vector to mac_hw.txt, one line per image with
N_HIDDEN space-separated integers. This replays the same images through the
PyTorch model and asserts the two agree exactly.

The testbench references a compare_mac.py that does not exist in the
repository, so the test could run but could never be checked. Agreement here
must be exact, not approximate: these are the integers the rest of the
datapath consumes.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_checkpoints import load_model          # noqa: E402
from synthetic_digits import make_set                # noqa: E402


def main():
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "model/mnist_h32aug.pt"
    hardware = sys.argv[2] if len(sys.argv) > 2 else "mac_hw.txt"
    n_hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    model, _ = load_model(checkpoint, torch.device("cpu"))
    model.eval()

    # Same image set the testbench uses: real MNIST, deterministic.
    from torchvision import datasets, transforms
    ds = datasets.MNIST("./data", train=False, download=True,
                        transform=transforms.ToTensor())

    hw_lines = [l.strip() for l in open(hardware) if l.strip()]
    if not hw_lines:
        raise SystemExit(f"{hardware} is empty -- run the testbench first")

    agree = 0
    mismatches = []
    for index, line in enumerate(hw_lines):
        expected_hw = np.array([int(v) for v in line.split()], dtype=np.int64)
        if expected_hw.size != n_hidden:
            raise SystemExit(f"line {index} has {expected_hw.size} values, "
                             f"expected {n_hidden}")

        image = ds[index][0].reshape(1, 784)
        with torch.no_grad():
            hidden = model.hidden_features(image)
        expected_sw = np.array([int(v) for v in hidden[0].tolist()], dtype=np.int64)

        if np.array_equal(expected_hw, expected_sw):
            agree += 1
        else:
            bad = np.nonzero(expected_hw != expected_sw)[0]
            mismatches.append((index, bad.tolist(),
                               expected_hw[bad].tolist(), expected_sw[bad].tolist()))

    total = len(hw_lines)
    print(f"images compared : {total}")
    print(f"MAC agreement   : {agree}/{total} "
          f"({100.0 * agree / total:.2f}%)")
    if mismatches:
        print("\nfirst mismatches (image, units, hardware, software):")
        for m in mismatches[:5]:
            print("  ", m)
        raise SystemExit("FAIL: MAC engine output differs from the model")
    print("\nOK: the RTL MAC engine reproduces the model's integer hidden "
          "vector exactly.")


if __name__ == "__main__":
    main()
