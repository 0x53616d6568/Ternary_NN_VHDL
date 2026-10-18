#!/usr/bin/env python3
"""Measure accuracy against stroke thickness and drawing size.

The classifier is trained on MNIST, whose thresholded digits average about 96 lit
pixels and are already size-normalised and centred. Freehand input differs on
both axes, and this script quantifies how much the model tolerates each.

Run it on two checkpoints to see the effect of training-time augmentation:

    python python/evaluate_robustness.py model/mnist_ternary_784_16_10.pt
    python python/evaluate_robustness.py model/mnist_morph_h16.pt
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate_checkpoints import load_model

SIDE = 28


def morph(image, radius):
    """Dilate (radius > 0) or erode (radius < 0) a boolean 28x28 image.

    Erosion is expressed as 1 - dilate(1 - x), so both directions use a dilation
    and the background pads to 0 in the inverted image.
    """
    if radius == 0:
        return image
    block = torch.from_numpy(image.astype(np.float32))[None, None]
    pad = abs(radius)
    size = 2 * pad + 1
    if radius > 0:
        out = F.max_pool2d(F.pad(block, (pad,) * 4, value=0.0), size, stride=1)
    else:
        inverse = 1.0 - block
        out = 1.0 - F.max_pool2d(F.pad(inverse, (pad,) * 4, value=0.0), size, stride=1)
    return out[0, 0].numpy() > 0.5


def rescale(image, span):
    """Scale the inked bounding box to fit `span` pixels and centre it.

    The aspect ratio is preserved: forcing the box into a square stretches
    non-square digits, which inflates the stroke and measures the resampler
    rather than the model. Area averaging is used so thick strokes stay
    connected; nearest neighbour shatters them.
    """
    rows, cols = np.nonzero(image)
    if len(rows) == 0:
        return image
    height = rows.max() - rows.min() + 1
    width = cols.max() - cols.min() + 1

    scale = span / max(height, width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    block = torch.from_numpy(image.astype(np.float32))[None, None]
    crop = block[:, :, rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    small = F.interpolate(crop, size=(new_h, new_w), mode="area")
    binary = (small >= 0.35).float()

    out = torch.zeros(1, 1, SIDE, SIDE)
    oy = (SIDE - new_h) // 2
    ox = (SIDE - new_w) // 2
    out[:, :, oy:oy + new_h, ox:ox + new_w] = binary
    return out[0, 0].numpy() > 0.5


@torch.no_grad()
def score(model, images):
    stacked = np.stack(images).reshape(len(images), -1)
    batch = torch.from_numpy(stacked.astype(np.float32))
    return model(batch).argmax(1).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--spans", default="14,17,20,24,28")
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _ = load_model(args.checkpoint, device)
    model.eval()

    dataset = datasets.MNIST(args.data_dir, train=False, download=True,
                             transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=250)

    originals, labels = [], []
    for images, batch_labels in loader:
        for image, label in zip(images, batch_labels):
            # MNIST tensors arrive as [1, 28, 28]; keep a 2-D grid so
            # morphology operates on the image and not a flattened strip.
            originals.append(image.squeeze().numpy() >= 0.5)
            labels.append(int(label))
            if len(originals) >= args.samples:
                break
        if len(originals) >= args.samples:
            break

    labels = np.array(labels)
    spans = [int(v) for v in args.spans.split(",")]

    print(f"checkpoint : {args.checkpoint}")
    print(f"samples    : {len(originals)}   morph_aug = "
          f"{model.morph_strength if hasattr(model, 'morph_strength') else 'n/a'}")
    print(f"mean ink   : {np.mean([img.sum() for img in originals]):.0f} pixels\n")

    print("Stroke thickness (bounding box untouched):")
    print(f"  {'radius':<10}{'ink':>7}{'accuracy':>12}")
    print("  " + "-" * 29)
    for radius in (-1, 0, 1, 2, 3):
        warped = [morph(img, radius) for img in originals]
        accuracy = 100.0 * np.mean(np.array(score(model, warped)) == labels)
        ink = np.mean([img.sum() for img in warped])
        tag = "  <- MNIST" if radius == 0 else ""
        print(f"  {radius:+d}{'':<7}{ink:>7.0f}{accuracy:>11.2f}%{tag}")

    print("\nDrawing size (stroke thickness preserved, scaled to span):")
    print(f"  {'span':<10}{'ink':>7}{'accuracy':>12}")
    print("  " + "-" * 29)
    for span in spans:
        warped = [rescale(img, span) for img in originals]
        accuracy = 100.0 * np.mean(np.array(score(model, warped)) == labels)
        ink = np.mean([img.sum() for img in warped])
        tag = "  <- MNIST typical" if span == 20 else ""
        print(f"  {span:<10}{ink:>7.0f}{accuracy:>11.2f}%{tag}")


if __name__ == "__main__":
    main()