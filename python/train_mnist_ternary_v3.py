#!/usr/bin/env python3
"""Ternary MNIST trainer applying Bonsai-style cost-reduction techniques.

Network: 784 -> H ReLU units -> 10 class scores (one hidden layer).

Forward weights are exactly -1, 0 or +1 and biases are integers, so the model
matches the ternary weight memories and integer accumulators in the VHDL design.
Hidden activations are rounded to integers and saturated to 0..127.

Techniques borrowed from the Bonsai 2 / BitNet b1.58 line of work, adapted to a
784-input MLP where the only binding hardware constraint is LUTs and cycles:

* Blockwise Walsh-Hadamard rotation of the input dimension. The rotation is
  orthogonal, so rotating the weights on the right and the inputs on the left is
  an identity transform. It spreads weight energy so that rounding to {-1,0,+1}
  destroys much less signal, which lets the sparsity threshold rise without
  losing accuracy. On hardware this is log2(block) stages of add/subtract and
  needs no multipliers.
* Selective precision: the small second layer (10 x H weights) is kept in full
  precision, matching Bonsai 2's choice to leave a small fraction of tensors at
  high precision. It costs about 1.3% of the weights and lets the large matrix
  stay ternary.
* Per-neuron hidden scaling alpha, which is free in hardware because
  ReLU(alpha * x) = alpha * ReLU(x), so alpha folds into the second layer.
* Closed-loop sparsity control: the actual zero density is measured every epoch
  and the threshold ratio is adjusted to hold a target density. Above roughly
  37.5% zeros the weights pack below two bits per weight.
* Positive bias initialisation, which removes the dead-unit problem at the
  source instead of reviving units after the damage is done.
"""

import argparse
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import (DataLoader, WeightedRandomSampler,
                              random_split)
from torchvision import datasets, transforms


def hadamard_matrix(size, device=None, dtype=torch.float32, normalise=True):
    """Sylvester-constructed Hadamard matrix, orthonormal when normalise is set.

    With normalise=False the entries stay exactly +1 and -1, so rotating a bus
    of integer pixel values produces an integer result. That is what makes the
    hardware match the model bit for bit: no fractional intermediate can land
    on the wrong side of the rounding boundary. The cost is that activations
    grow by sqrt(size), which the trained biases and the clamp absorb.
    """
    if size & (size - 1) != 0:
        raise ValueError("Walsh-Hadamard block size must be a power of two")
    matrix = torch.ones(1, 1, device=device, dtype=dtype)
    while matrix.shape[0] < size:
        top = torch.cat([matrix, matrix], dim=1)
        bottom = torch.cat([matrix, -matrix], dim=1)
        matrix = torch.cat([top, bottom], dim=0)
    return matrix / math.sqrt(size) if normalise else matrix


def rotate_blocks(tensor, block, transform):
    """Apply a blockwise orthogonal transform to the last dimension."""
    if block <= 1:
        return tensor
    length = tensor.shape[-1]
    if length % block != 0:
        raise ValueError(f"Last dimension {length} is not divisible by block {block}")
    reshaped = tensor.reshape(-1, length // block, block)
    return (reshaped @ transform).reshape(tensor.shape)


class TernaryLinear(nn.Module):
    """Linear layer whose forward pass always uses weights in {-1, 0, +1}."""

    def __init__(self, in_features, out_features, threshold_ratio, clip=True, ternary=True,
                 walsh_block=1, normalise_rotation=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ternary = ternary
        self.clip = clip
        self.walsh_block = walsh_block
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)
        if walsh_block > 1:
            self.register_buffer(
                "transform",
                hadamard_matrix(walsh_block, normalise=normalise_rotation),
                persistent=False,
            )
        self.set_threshold_ratio(threshold_ratio)

    def set_threshold_ratio(self, value):
        self._ratio = value

    def latent_weight(self):
        weight = self.linear.weight
        weight = torch.clamp(weight, -1.0, 1.0) if self.clip else weight
        if self.walsh_block > 1:
            # Rotate first, quantise second: rotating a ternary matrix is not a
            # no-op, so the rotation has to precede the ternarisation.
            weight = rotate_blocks(weight, self.walsh_block, self.transform)
        return weight

    def ternary_weight(self):
        latent = self.latent_weight()
        weight = latent.detach()
        threshold = self._ratio * weight.abs().mean()
        hard = torch.where(weight.abs() >= threshold, weight.sign(), torch.zeros_like(weight))
        # Straight-through estimator with an exact forward value. Writing it as
        # hard + (latent - latent) instead leaves a float rounding error of order
        # 1e-7 in the forward pass, which is enough to flip an integer rounding
        # near a boundary and make the hardware disagree with the model.
        return hard.detach() + (latent - latent.detach())

    def quantized_bias(self, bias):
        return torch.round(bias).detach() + (bias - bias.detach())

    def forward(self, x):
        weight = self.ternary_weight() if self.ternary else self.linear.weight
        return F.linear(x, weight, self.quantized_bias(self.linear.bias))

    @torch.no_grad()
    def zero_density(self):
        if not self.ternary:
            return 0.0
        weight = self.latent_weight().detach()
        threshold = self._ratio * weight.abs().mean()
        return float((weight.abs() < threshold).float().mean())

    def effective_bits_per_weight(self):
        """Bits needed for a presence bitmap plus a sign vector."""
        density = self.zero_density()
        if self.ternary and density > 0.0:
            return 1.0 + (1.0 - density)
        return 2.0


def random_morphology(x, strength):
    """Randomly dilate or erode each binary sample by up to `strength` pixels.

    MNIST digits are thresholded grayscale, so their strokes average about 96
    pixels. A mouse stroke on the 28x28 grid is far thicker, and a classifier
    trained only on MNIST degrades sharply outside that distribution. Applying a
    random dilation or erosion during training covers the stroke-thickness shift
    without changing the integer-exact forward path, since the result is still
    binary and still feeds the same arithmetic the hardware performs.
    """
    if strength <= 0:
        return x
    batch = x.shape[0]
    choice = torch.randint(-strength, strength + 1, (batch,))
    out = x.clone()
    for radius in range(-strength, strength + 1):
        if radius == 0:
            continue
        rows = (choice == radius).nonzero().flatten()
        if rows.numel() == 0:
            continue
        block = x[rows].reshape(-1, 1, 28, 28)
        if radius > 0:
            size = 2 * radius + 1
            padded = F.pad(block, (radius,) * 4, value=0.0)
            warped = F.max_pool2d(padded, size, stride=1)
        else:
            pad = -radius
            size = 2 * pad + 1
            padded = F.pad(-block, (pad,) * 4, value=0.0)
            warped = -F.max_pool2d(padded, size, stride=1)
        out[rows] = warped.reshape(rows.numel(), -1).to(out.dtype)
    return out


def random_affine(x, scale_range, shift_px, rot_deg):
    """Random per-sample scale, translation and rotation of a binary image.

    The classifier is highly sensitive to position and scale: measured on MNIST,
    a one pixel shift costs 3 to 9 accuracy points and a 25% scale change costs
    about 30. MNIST is pre-normalised, so without this the network never sees a
    digit anywhere else. Applying the transform to the binary image and
    re-thresholding keeps every value on the forward path an integer, so the
    exported model stays bit-exact with the hardware.
    """
    if scale_range is None and not shift_px and not rot_deg:
        return x
    batch = x.shape[0]
    image = x.reshape(batch, 1, 28, 28)

    if scale_range is None:
        scales = torch.ones(batch)
    else:
        low, high = scale_range
        scales = low + (high - low) * torch.rand(batch)

    if rot_deg:
        limit = math.radians(rot_deg)
        angles = (torch.rand(batch) * 2 - 1) * limit
    else:
        angles = torch.zeros(batch)

    if shift_px:
        # affine_grid works in normalised coordinates, where 2.0 spans the grid.
        limit_x = (2.0 * shift_px) / 28.0
        tx = (torch.rand(batch) * 2 - 1) * limit_x
        ty = (torch.rand(batch) * 2 - 1) * limit_x
    else:
        tx = torch.zeros(batch)
        ty = torch.zeros(batch)

    cos = torch.cos(angles) * scales
    sin = torch.sin(angles) * scales
    theta = torch.stack([
        torch.stack([cos, -sin, -tx], dim=1),
        torch.stack([sin, cos, -ty], dim=1),
    ], dim=1)

    # affine_grid turns the 2x3 matrices into sampling coordinates.
    coords = F.affine_grid(theta, (batch, 1, 28, 28), align_corners=False)
    sample = F.grid_sample(image, coords, align_corners=False,
                           padding_mode="zeros")
    return (sample[:, 0].reshape(batch, -1) >= 0.5).to(x.dtype)


def pool_image(x, factor):
    """Average-pool a flattened 28x28 batch by an integer factor per side."""
    if factor == 1:
        return x
    batch = x.shape[0]
    side = 28 // factor
    if side * factor != 28:
        raise ValueError("Pooling factor must divide 28")
    image = x.reshape(batch, 28, 28)
    pooled = image.reshape(batch, side, factor, side, factor).mean(dim=(2, 4))
    return pooled.reshape(batch, side * side)


class VHDLCompatibleMLP(nn.Module):
    def __init__(self, hidden_units, threshold_ratio=0.7, clip=True, gray_inputs=False,
                 walsh_block=16, layer2_precision="full", alive_bias_init=1.0,
                 pool_factor=1, hidden_layers=1, normalise_rotation=True, rotation_shift=0,
                 morph_strength=0, affine_range=None, affine_shift=0,
                 affine_rot=0):
        super().__init__()
        self.input_size = 784 // (pool_factor * pool_factor)
        self.pool_factor = pool_factor
        self.hidden_units = hidden_units
        self.hidden_layers = hidden_layers
        # In integer-exact mode the unnormalised rotation grows values by
        # sqrt(block); dropping that many low bits restores the scale. It is an
        # arithmetic shift, so it stays exact and costs no gates in hardware.
        self.rotation_shift = 0 if normalise_rotation else rotation_shift
        self.fc1 = TernaryLinear(self.input_size, hidden_units, threshold_ratio, clip,
                                 walsh_block=walsh_block,
                                 normalise_rotation=normalise_rotation)
        # Extra hidden layers carry no rotation: they consume integer activations,
        # not the pixel bus, so the transform is applied exactly once on input.
        self.hidden_extra = nn.ModuleList([
            TernaryLinear(hidden_units, hidden_units, threshold_ratio, clip)
            for _ in range(hidden_layers - 1)
        ])
        self.fc2 = TernaryLinear(hidden_units, 10, threshold_ratio, clip,
                                 ternary=(layer2_precision == "ternary"))
        self.gray_inputs = gray_inputs
        self.walsh_block = walsh_block
        self.alive_bias_init = alive_bias_init
        # Only consulted while training, so evaluation and export stay
        # deterministic and keep the exact integer forward path.
        self.morph_strength = morph_strength
        self.affine_range = affine_range
        self.affine_shift = affine_shift
        self.affine_rot = affine_rot
        if alive_bias_init:
            with torch.no_grad():
                self.fc1.linear.bias.fill_(float(alive_bias_init))
                for layer in self.hidden_extra:
                    layer.linear.bias.fill_(float(alive_bias_init))

    @staticmethod
    def _integer_relu(x):
        hidden = F.relu(x).clamp(0.0, 127.0)
        # Exact integer value in the forward pass, gradient passed straight
        # through, for the same reason as the weight quantiser above.
        rounded = torch.round(hidden)
        return rounded.detach() + (hidden - hidden.detach())

    def set_threshold_ratio(self, value):
        self.fc1.set_threshold_ratio(value)
        for layer in self.hidden_extra:
            layer.set_threshold_ratio(value)
        self.fc2.set_threshold_ratio(value)

    def prepare_inputs(self, image):
        x = image.reshape(image.shape[0], 784)
        if not self.gray_inputs:
            x = (x >= 0.5).to(image.dtype)
        if self.training:
            if self.morph_strength > 0:
                x = random_morphology(x, self.morph_strength)
            x = random_affine(x, self.affine_range, self.affine_shift,
                              self.affine_rot)
        if self.pool_factor > 1:
            # Hardware does this with one adder and a shift in the input path.
            x = pool_image(x, self.pool_factor)
        # The activation rotation must use the same block size and basis as fc1,
        # because the hardware applies the identical transform to the pixel bus.
        if self.fc1.walsh_block > 1:
            x = rotate_blocks(x, self.fc1.walsh_block, self.fc1.transform)
            if self.rotation_shift:
                x = torch.bitwise_right_shift(
                    torch.round(x).to(torch.int32), self.rotation_shift
                ).to(x.dtype)
        return x

    def forward(self, image):
        hidden = self._integer_relu(self.fc1(self.prepare_inputs(image)))
        for layer in self.hidden_extra:
            hidden = self._integer_relu(layer(hidden))
        return self.fc2(hidden)

    @torch.no_grad()
    def hidden_features(self, image):
        hidden = self._integer_relu(self.fc1(self.prepare_inputs(image)))
        for layer in self.hidden_extra:
            hidden = self._integer_relu(layer(hidden))
        return torch.round(hidden)

    @torch.no_grad()
    def export_ternary_weights(self):
        """Rotated, ternary first-layer weights ready for the VHDL memories."""
        if self.walsh_block <= 1:
            raise SystemExit("Walsh-Hadamard rotation must be enabled to export rotated weights")
        weight = self.fc1.ternary_weight().detach().round().to(torch.int8)
        return weight, torch.round(self.fc1.linear.bias.detach()).to(torch.int64)


class TeacherMLP(nn.Module):
    def __init__(self, hidden, input_size=784, pool_factor=1):
        super().__init__()
        self.pool_factor = pool_factor
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, image):
        x = image.reshape(image.shape[0], 784)
        if self.pool_factor > 1:
            x = pool_image(x, self.pool_factor)
        return self.fc2(F.relu(self.fc1(x)))


def build_sampler(train_set, full_train, focus, boost):
    """Return a sampler that oversamples the digits named in `focus`.

    A WeightedRandomSampler is used rather than physically duplicating indices
    so the epoch length, and therefore the learning-rate schedule, stays exactly
    the same as an unweighted run. `focus` is a string of digit characters, e.g.
    "0236789"; an empty string disables rebalancing entirely.
    """
    if not focus:
        return None
    digits = sorted({int(c) for c in focus if c.isdigit()})
    if not digits:
        return None

    # random_split returns Subset objects, so reach through to the labels of the
    # indices this split actually owns.
    targets = torch.tensor([int(full_train[i][1]) for i in train_set.indices])
    weights = torch.ones(len(targets), dtype=torch.double)
    for digit in digits:
        weights[targets == digit] = boost
    total = float(weights.sum())
    print(f"Focus digits {digits} boosted {boost}x each; "
          f"effective share {100.0 * float(weights[targets != 0].sum()) / total:.0f}% "
          f"across the focused classes")
    return WeightedRandomSampler(weights, num_samples=len(targets), replacement=True)


def build_transform(augment):
    if not augment:
        return transforms.ToTensor()
    return transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    count = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        scores = model(images)
        loss_sum += F.cross_entropy(scores, labels, reduction="sum").item()
        correct += (scores.argmax(dim=1) == labels).sum().item()
        count += labels.numel()
    return loss_sum / count, correct / count


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def distillation_loss(student, teacher, labels, temperature, alpha):
    soft_targets = F.softmax(teacher / temperature, dim=1)
    soft = F.kl_div(F.log_softmax(student / temperature, dim=1), soft_targets,
                    reduction="batchmean") * (temperature ** 2)
    return alpha * soft + (1.0 - alpha) * F.cross_entropy(student, labels)


def load_teacher(path, device, input_size=784, pool_factor=1):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture")
    if not architecture or architecture[0] != input_size:
        raise SystemExit(f"Teacher checkpoint must have {input_size} inputs.")
    teacher = TeacherMLP(architecture[1], input_size, pool_factor)
    state = checkpoint["model_state_dict"]
    prefix = "fc1." if "fc1.weight" in state else "fc1.linear."
    teacher.fc1.weight.data = state[prefix + "weight"].float()
    teacher.fc1.bias.data = state[prefix + "bias"].float()
    teacher.fc2.weight.data = state[prefix.replace("fc1", "fc2") + "weight"].float()
    teacher.fc2.bias.data = state[prefix.replace("fc1", "fc2") + "bias"].float()
    return teacher.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default="mnist_ternary_v3.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--threshold-ratio", type=float, default=0.7)
    parser.add_argument("--target-zeros", type=float, default=None,
                        help="Target zero density for sub-2-bit packing (0.4 enables it)")
    parser.add_argument("--sparsity-step", type=float, default=0.05)
    parser.add_argument("--walsh-block", type=int, default=16, help="0 disables rotation")
    parser.add_argument("--pool-factor", type=int, default=1,
                        help="Average-pool factor per side; 2 means 28x28 -> 14x14")
    parser.add_argument("--hidden-layers", type=int, default=1,
                        help="Number of hidden layers; 2 gives the classic 784-H-H-10 shape")
    parser.add_argument("--integer-exact", action="store_true",
                        help="Unnormalised rotation + binary inputs so every value on the "
                             "forward path is an integer, matching the VHDL bit for bit")
    parser.add_argument("--layer2-precision", choices=["full", "ternary"], default="full")
    parser.add_argument("--alive-bias-init", type=float, default=1.0)
    parser.add_argument("--morph-aug", type=int, default=0,
                        help="Random dilate/erode by up to N pixels during training; "
                             "covers the stroke thickness of freehand input")
    parser.add_argument("--focus-digits", default="",
                        help="Digits to oversample during training, e.g. '0236789'. "
                             "Targets the measured 8->6 loop confusion plus the "
                             "7->2 and 3 confusions.")
    parser.add_argument("--focus-boost", type=float, default=3.0,
                        help="Sampling weight multiplier for the focused digits")
    parser.add_argument("--aug-scale-min", type=float, default=None)
    parser.add_argument("--aug-scale-max", type=float, default=None)
    parser.add_argument("--aug-shift", type=float, default=0.0,
                        help="Random translation in pixels during training")
    parser.add_argument("--aug-rotate", type=float, default=0.0,
                        help="Random rotation in degrees during training")
    parser.add_argument("--gray-inputs", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--distill-alpha", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = build_transform(args.augment)
    full_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True,
                              transform=transforms.ToTensor())
    train_set, val_set = random_split(full_train, [55000, 5000],
                                      generator=torch.Generator().manual_seed(args.seed))
    # Per-digit oversampling.
    #
    # The confusion matrices show the damage is concentrated in a small set of
    # digits rather than spread evenly:
    #
    #   hand-drawn, 8 -> 6 : 25 of 40 (8 scored 35.0%, the worst digit by far)
    #   hand-drawn, 5 -> 8 :  8      7 -> 2 : 8      0 -> 9 : 8
    #   real MNIST, 4 -> 9 : 10      8 -> 3 : 6      5 -> 3 : 6
    #   fonts,      7 -> 1 :  4      7 -> 2 : 3
    #
    # 6/8/9/0 form a loop-count family -- the same bowl shape distinguished only
    # by whether it closes, which a 16-unit ternary layer has very little room to
    # encode. 2 and 7 are a stroke-topology pair, and 3 appears as both a source
    # and a target on real MNIST. Uniform sampling spends capacity evenly on
    # digits that are already at 97% while the weak ones stay weak.
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              sampler=build_sampler(train_set, full_train,
                                                    args.focus_digits, args.focus_boost),
                              num_workers=args.workers,
                              pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=1000, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=1000, num_workers=args.workers)

    input_size = 784 // (args.pool_factor ** 2)
    if args.walsh_block > 1 and input_size % args.walsh_block != 0:
        raise SystemExit(
            f"Walsh block {args.walsh_block} does not divide the pooled input size {input_size}")
    if args.integer_exact:
        # Binary pixels and an unnormalised rotation keep every intermediate an
        # integer, so the exported model is reproducible in hardware exactly.
        args.gray_inputs = False
        args.alive_bias_init = args.alive_bias_init if args.alive_bias_init else 4.0

    model = VHDLCompatibleMLP(
        args.hidden_units, args.threshold_ratio,
        gray_inputs=args.gray_inputs, walsh_block=args.walsh_block,
        layer2_precision=args.layer2_precision, alive_bias_init=args.alive_bias_init,
        pool_factor=args.pool_factor, hidden_layers=args.hidden_layers,
        morph_strength=args.morph_aug,
        affine_range=((args.aug_scale_min, args.aug_scale_max)
                      if args.aug_scale_min is not None else None),
        affine_shift=args.aug_shift,
        affine_rot=args.aug_rotate,
        normalise_rotation=not args.integer_exact,
        rotation_shift=(args.walsh_block.bit_length() - 1) // 2
                        if args.integer_exact else 0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = cosine_with_warmup(optimizer, 0.05 * args.epochs * len(train_loader),
                                   args.epochs * len(train_loader))
    teacher = load_teacher(args.teacher, device, input_size, args.pool_factor) if args.teacher else None
    if teacher is not None:
        print(f"Distilling from teacher: {args.teacher}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    ratio = args.threshold_ratio

    shape = " -> ".join(str(v) for v in [input_size] + [args.hidden_units] * args.hidden_layers + [10])
    print(f"Device: {device}; {shape}; "
          f"walsh_block={args.walsh_block or 'off'}; pool={args.pool_factor}; "
          f"layer2={args.layer2_precision}; "
          f"inputs={'gray8' if args.gray_inputs else 'binary'}; augment={args.augment}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            if teacher is not None:
                loss = distillation_loss(scores, teacher(images), labels,
                                         args.temperature, args.distill_alpha)
            else:
                loss = F.cross_entropy(scores, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * labels.numel()
            seen += labels.numel()

        density = model.fc1.zero_density()
        if args.target_zeros:
            error = density - args.target_zeros
            ratio = min(2.0, max(0.1, ratio + args.sparsity_step * error / 0.05))
            model.set_threshold_ratio(ratio)

        val_loss, val_accuracy = evaluate(model, val_loader, device)
        bits = model.fc1.effective_bits_per_weight()
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={loss_sum / seen:.4f} "
              f"val_loss={val_loss:.4f} val_accuracy={100.0 * val_accuracy:.2f}% "
              f"zeros={100.0 * density:.1f}% ratio={ratio:.2f} bits/w={bits:.2f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()
                                         if k != "transform"},
                    "threshold_ratio": ratio,
                    "hidden_units": args.hidden_units,
                    "architecture": [input_size] + [args.hidden_units] * args.hidden_layers + [10],
                    "pool_factor": args.pool_factor,
                    "hidden_layers": args.hidden_layers,
                    "normalise_rotation": not args.integer_exact,
                    "integer_exact": args.integer_exact,
                    "morph_aug": args.morph_aug,
                    "aug_scale": [args.aug_scale_min, args.aug_scale_max],
                    "aug_shift": args.aug_shift,
                    "aug_rotate": args.aug_rotate,
                    "rotation_shift": (args.walsh_block.bit_length() - 1) // 2
                                     if args.integer_exact else 0,
                    "walsh_block": args.walsh_block,
                    "layer2_precision": args.layer2_precision,
                    "alive_bias_init": args.alive_bias_init,
                    "gray_inputs": args.gray_inputs,
                    "zero_density": density,
                    "bits_per_weight": bits,
                    "hidden_activation": "integer ReLU saturated to 0..127",
                    "ternary_values": [-1, 0, 1],
                    "best_validation_accuracy": best_accuracy,
                    "epoch": epoch,
                },
                output_path,
            )

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Best validation accuracy: {100.0 * best_accuracy:.2f}%")
    print(f"Held-out test accuracy: {100.0 * test_accuracy:.2f}% (loss {test_loss:.4f})")
    print(f"First-layer zero density: {100.0 * checkpoint['zero_density']:.2f}% "
          f"({checkpoint['bits_per_weight']:.2f} bits/weight)")
    print(f"Checkpoint saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()