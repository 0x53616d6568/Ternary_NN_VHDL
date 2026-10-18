"""Load a classic float MLP checkpoint (fc1/fc2.*) for GUI use.

Why this exists
---------------
Two separate faults made the float teacher look terrible when loaded through
`evaluate_checkpoints.load_model`:

1. **Silent load failure.** `load_state_dict(strict=False)` discarded all four
   weight tensors, because the old checkpoint names them `fc1.weight` while the
   current `VHDLCompatibleMLP` nests them as `fc1.linear.weight`. Nothing was
   reported, and the resulting randomly initialised network scored **8.1%** on
   MNIST -- which reads like a bad model rather than a broken load.

2. **Ternary wrapper on float weights.** Forcing the float weights through
   `VHDLCompatibleMLP` caps them at 90.8%, because that class applies integer
   rounding, a fixed-point activation clamp and a Walsh-Hadamard rotation that
   the float teacher was never trained with. Measured directly:

       plain float MLP, original weights    98.20%   (recorded 98.24%)
       forced through VHDLCompatibleMLP     90.80%
       silently unloaded, random weights     8.10%

The honest conclusion is that a classic float model cannot be inspected through
the ternary hardware's data path -- the whole point of that path is the integer
arithmetic the float model does not use. So this loader rebuilds the original
plain architecture and adapts it to the interface `digit_console` expects.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


class ClassicFloatMLP(nn.Module):
    """784 -> ReLU -> hidden -> 10, exactly as the float teacher was trained.

    Deliberately NOT a VHDLCompatibleMLP. The ternary class rounds
    activations to integers and rotates the pixel bus; doing that to float
    weights costs about 7 accuracy points, which is the cost of pushing a
    classical model through a hardware data path it was never designed for.
    """

    def __init__(self, hidden):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

    def prepare_inputs(self, x):
        """No rotation and no rounding: the float model consumes greyscale."""
        return x

    def hidden_features(self, x):
        return torch.relu(self.fc1(x))

    def zero_density(self):
        """Meaningless for float weights, reported so callers can proceed."""
        return 0.0


def load_float_checkpoint(path, device):
    """Return (model, checkpoint) for a classic float MLP, weights verified."""
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    architecture = checkpoint["architecture"]
    state = checkpoint["model_state_dict"]

    hidden = architecture[1]
    model = ClassicFloatMLP(hidden)

    # Map both naming conventions explicitly. Every tensor must land, or this
    # raises rather than returning a randomly initialised network -- the exact
    # failure this function was written to prevent.
    def pick(*names):
        for name in names:
            if name in state:
                return state[name]
        return None

    pairs = [
        (model.fc1.weight, pick("fc1.weight", "fc1.linear.weight")),
        (model.fc1.bias,   pick("fc1.bias",   "fc1.linear.bias")),
        (model.fc2.weight, pick("fc2.weight", "fc2.linear.weight")),
        (model.fc2.bias,   pick("fc2.bias",   "fc2.linear.bias")),
    ]
    absent = [name for name, (_, value) in
              zip(("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"), pairs)
              if value is None]
    if absent:
        raise ValueError(
            f"{Path(path).name}: could not find {absent} in the checkpoint "
            f"(it has {sorted(state)}). Refusing to run random weights.")

    for tensor, value in pairs:
        if tuple(tensor.shape) != tuple(value.shape):
            raise ValueError(
                f"{Path(path).name}: shape mismatch, model wants "
                f"{tuple(tensor.shape)} but checkpoint has {tuple(value.shape)}")
        tensor.data = value.float()

    if float(model.fc1.weight.detach().abs().sum()) == 0.0:
        raise ValueError(f"{Path(path).name}: first-layer weights are all zero.")

    return model.to(device).eval(), checkpoint


if __name__ == "__main__":
    from torchvision import datasets, transforms
    target = (sys.argv[1] if len(sys.argv) > 1
              else r"..\ternary_mnist_fpga_project\output\float_teacher.pt")
    m, c = load_float_checkpoint(target, torch.device("cpu"))
    print(f"loaded {target}")
    print(f"  architecture {c['architecture']}  "
          f"recorded val {100 * c['best_validation_accuracy']:.2f}%")
    ds = datasets.MNIST("./data", train=False, download=True,
                        transform=transforms.ToTensor())
    ok = 0
    N = 1000
    with torch.no_grad():
        for i in range(N):
            ok += int(m(ds[i][0].reshape(1, 784)).argmax(1)) == int(ds[i][1])
    print(f"  measured MNIST {100.0 * ok / N:.2f}%")
