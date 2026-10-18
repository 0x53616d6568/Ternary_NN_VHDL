#!/usr/bin/env python3
"""Export a v3 ternary checkpoint into the VHDL weight memories.

Replaces the older extract_vhdl_parameters.py + update_vhdl_weight_memory.py
pair, which were hard-coded to the original 784-8-10 network and assumed every
weight was ternary. This exporter handles:

* any hidden size, taken from the checkpoint rather than assumed
* an optional Walsh-Hadamard rotation on the first layer, in which case the
  exported W1 is the rotated matrix and the hardware must apply the identical
  transform to the pixel bus before the first dense layer
* a full-precision second layer. The network is trained with the small output
  matrix left in floating point (Bonsai-style selective precision), so W2 is
  quantised to a signed 8-bit integer with a shared scale rather than being
  forced into {-1, 0, +1}

Scaling W2 and B2 by the same positive constant does not change which class
ArgMax selects, so the hardware can use integer arithmetic throughout and the
scale factor never has to leave the Python side.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate_checkpoints import load_model
import torch


def quantise_second_layer(weight, bias, bits=16):
    """Scale W2 and the already-integer bias by one positive constant.

    The trained model computes scores as hidden @ W2 + round(bias): the bias is
    quantised in absolute units, not relative to the weight scale. The hardware
    has to reproduce that exactly, so the bias is rounded to an integer first
    and only then divided by the shared weight scale. Rounding bias / scale
    directly instead would quantise it on a different grid and change the arg-max
    for images whose top two scores are close.

    Scaling both terms by one positive constant cannot change which class wins.
    The second layer is only 10 x H weights, so widening it is far cheaper than
    the first layer and is where the accuracy is recovered.
    """
    limit = (1 << (bits - 1)) - 1
    peak = float(np.abs(weight).max())
    if peak == 0.0:
        return (np.zeros(weight.shape, dtype=np.int64),
                np.zeros(bias.shape, dtype=np.int64), 1.0, bits)
    scale = peak / limit
    w_int = np.rint(weight / scale).clip(-limit, limit).astype(np.int64)
    b_model = np.rint(bias)                      # exactly what the model uses
    b_int = np.rint(b_model / scale).astype(np.int64)
    return w_int, b_int, scale, bits


def vhdl_vector(name, values, width, unsigned=False):
    words = []
    for value in values:
        value = int(value)
        if unsigned:
            low, high = 0, (1 << width) - 1
        else:
            low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if not low <= value <= high:
            raise ValueError(f"{value} does not fit in {'unsigned' if unsigned else 'signed'} {width} bits")
        words.append(f"{value & ((1 << width) - 1):0{width // 4}X}")
    # The lowest-indexed memory item occupies the least-significant word.
    stream = "".join(reversed(words))
    body = " &\n    ".join(
        f'x"{stream[i:i + 64]}"' for i in range(0, len(stream), 64)
    )
    return (f"  constant {name}_INIT : "
            f"std_logic_vector({len(values) * width - 1} downto 0) :=\n    {body};\n")


def pack_ternary_transposed(matrix):
    """Pack a (N_OUT, N_IN) ternary matrix into 2-bit words for the MAC engine.

    The engine reads one word per input cycle and needs every output unit's
    weight for that input, so the matrix is transposed and packed two bits per
    weight. Two-bit two's complement encodes the ternary set with no decoder:
    00 = 0, 01 = +1, 11 = -1.

    For 784 x 16 that is 784 words of 32 bits, which is 25,088 bits and fits in a
    single block RAM, against 100,352 bits for the byte-per-weight form.
    """
    codes = np.zeros(matrix.shape, dtype=np.uint8)
    codes[matrix == 1] = 0b01
    codes[matrix == -1] = 0b11
    words = []
    for column in range(matrix.shape[1]):
        word = 0
        for row in range(matrix.shape[0]):
            word |= int(codes[row, column]) << (2 * row)
        words.append(word)
    return np.array(words, dtype=np.uint64), codes


def ternary_of(layer):
    """Exactly {-1,0,+1} weights of a TernaryLinear, including any rotation."""
    weight = layer.ternary_weight().detach()
    return torch.round(weight).to(torch.int8).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="output/vhdl_parameters.npz")
    parser.add_argument("--vhdl-output", default="rtl/ModelWeights.vhd")
    parser.add_argument("--w2-bits", type=int, default=16, choices=[8,12,16,24],
                        help="Width of the selective-precision output layer")
    parser.add_argument("--samples", type=int, default=0,
                        help="Also write this many MNIST test vectors for simulation")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, checkpoint = load_model(args.checkpoint, device)

    depth = model.hidden_layers
    if depth < 1:
        raise SystemExit("At least one hidden layer is required.")
    if depth > 2:
        raise SystemExit(
            "The VHDL supports one or two hidden layers; deeper models need "
            "further MAC engine instances.")

    w1, b1 = model.export_ternary_weights()
    w1 = w1.numpy().astype(np.int8)
    b1 = b1.numpy().astype(np.int64)

    w2_raw = model.fc2.linear.weight.detach().numpy().astype(np.float64)
    b2_raw = model.fc2.linear.bias.detach().numpy().astype(np.float64)
    w2, b2, scale, w2_bits = quantise_second_layer(w2_raw, b2_raw, args.w2_bits)

    input_size = model.input_size
    hidden = model.fc1.linear.out_features
    if w1.shape != (hidden, input_size):
        raise SystemExit(f"W1 is {w1.shape}; expected {(hidden, input_size)}")
    if not np.isin(w1, [-1, 0, 1]).all():
        raise SystemExit("W1 must be exactly {-1, 0, +1}.")

    arrays = {"W1": w1, "B1": b1, "W2": w2, "B2": b2}
    packed, codes = pack_ternary_transposed(w1)
    arrays["W1_PACKED"] = packed

    # Naming: W1/W2 are the first hidden layer and the output layer, matching
    # the original design. A second hidden layer is exported as W3/B3.
    if depth == 2:
        extra = model.hidden_extra[0]
        w3 = ternary_of(extra)
        b3 = torch.round(extra.linear.bias.detach()).to(torch.int64).numpy()
        packed3, _ = pack_ternary_transposed(w3)
        arrays["W3"] = w3
        arrays["B3"] = b3
        arrays["W3_PACKED"] = packed3
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    report = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "architecture": [input_size] + [hidden] * depth + [10],
        "hidden_layers": depth,
        "walsh_block": model.fc1.walsh_block,
        "gray_inputs": model.gray_inputs,
        "pool_factor": model.pool_factor,
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "w1_zero_density_pct": round(100.0 * float((w1 == 0).mean()), 3),
        "w1_bits_per_weight": round(model.fc1.effective_bits_per_weight(), 3),
        "w2_full_precision": model.fc2.ternary is False,
        "w2_bits": w2_bits,
        "w2_scale_not_deployed": scale,
        "bias_format": "signed 32-bit integer, rounded from trained bias",
    }

    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    content = ["library ieee;", "use ieee.std_logic_1164.all;", "", "package ModelWeights is"]
    content.append(vhdl_vector("W1", w1.reshape(-1).tolist(), 8).rstrip())
    content.append(vhdl_vector("W1_PACKED", packed.reshape(-1).tolist(), 2 * hidden, unsigned=True).rstrip())
    content.append(vhdl_vector("B1", b1.reshape(-1).tolist(), 32).rstrip())
    if depth == 2:
        content.append(vhdl_vector("W3", arrays["W3"].reshape(-1).tolist(), 8).rstrip())
        content.append(vhdl_vector("W3_PACKED", arrays["W3_PACKED"].reshape(-1).tolist(),
                                   2 * hidden, unsigned=True).rstrip())
        content.append(vhdl_vector("B3", arrays["B3"].reshape(-1).tolist(), 32).rstrip())
    content.append(vhdl_vector("W2", w2.reshape(-1).tolist(), w2_bits).rstrip())
    content.append(vhdl_vector("B2", b2.reshape(-1).tolist(), 32).rstrip())
    content.extend(["end package ModelWeights;", ""])

    vhdl_path = Path(args.vhdl_output)
    vhdl_path.write_text("\n".join(content), encoding="utf-8")

    print("Exported " + "-".join(str(v) for v in report["architecture"]))
    print(f"  H1 {w1.shape} ternary, {report['w1_zero_density_pct']:.1f}% zeros, "
          f"{report['w1_bits_per_weight']:.2f} bits/weight")
    if depth == 2:
        w3 = arrays["W3"]
        print(f"  H2 {w3.shape} ternary, {100.0 * float((w3 == 0).mean()):.1f}% zeros")
    print(f"  W2 {w2.shape} signed {w2_bits}-bit (selective precision layer)")
    print(f"  Walsh-Hadamard block: {model.fc1.walsh_block or 'disabled'}")
    print(f"  Parameters  -> {output.resolve()}")
    print(f"  VHDL package-> {vhdl_path.resolve()}")


if __name__ == "__main__":
    main()