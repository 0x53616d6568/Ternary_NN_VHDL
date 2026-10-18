#!/usr/bin/env python3
"""Generate mouse-drawn digit images for testing the drawing interface.

The GUI is driven with a mouse, which produces strokes that differ from MNIST in
three measurable ways: they are thicker, they are not size-normalised, and they
carry hand jitter. Measuring accuracy on MNIST alone therefore says nothing about
how the tool behaves in use.

This module renders each digit from stroke definitions the way a mouse would:
a continuous polyline stamped with a round brush, then jittered, offset and
rescaled. That produces a test set with a controlled, measurable shift from the
training distribution.

    python python/synthetic_digits.py --model model/mnist_morph_h16.pt
"""

import argparse
import math
import random

import numpy as np

SIDE = 28

# Each digit is a list of strokes; each stroke a polyline in a unit box.
STROKES = {
    0: [[(0.5, 0.05), (0.78, 0.2), (0.85, 0.5), (0.78, 0.8), (0.5, 0.95),
         (0.22, 0.8), (0.15, 0.5), (0.22, 0.2), (0.5, 0.05)]],
    1: [[(0.5, 0.08), (0.5, 0.95)], [(0.3, 0.28), (0.5, 0.08)]],
    2: [[(0.18, 0.28), (0.35, 0.08), (0.7, 0.08), (0.85, 0.25), (0.8, 0.45),
         (0.55, 0.62), (0.2, 0.92), (0.85, 0.92)]],
    3: [[(0.2, 0.12), (0.6, 0.06), (0.82, 0.22), (0.6, 0.45), (0.85, 0.62),
         (0.65, 0.92), (0.22, 0.88)]],
    4: [[(0.7, 0.08), (0.15, 0.62), (0.88, 0.62)], [(0.7, 0.08), (0.68, 0.95)]],
    5: [[(0.8, 0.08), (0.3, 0.08), (0.25, 0.42), (0.6, 0.38), (0.85, 0.6),
         (0.6, 0.92), (0.2, 0.86)]],
    6: [[(0.75, 0.06), (0.4, 0.2), (0.22, 0.55), (0.3, 0.85), (0.62, 0.93),
         (0.82, 0.72), (0.62, 0.5), (0.3, 0.56)]],
    7: [[(0.15, 0.1), (0.85, 0.1)], [(0.85, 0.1), (0.42, 0.95)]],
    8: [[(0.5, 0.06), (0.25, 0.2), (0.45, 0.45), (0.75, 0.6), (0.72, 0.88),
         (0.45, 0.95), (0.2, 0.8), (0.4, 0.5), (0.62, 0.25), (0.5, 0.06)]],
    9: [[(0.72, 0.12), (0.4, 0.06), (0.22, 0.25), (0.45, 0.46), (0.75, 0.38),
         (0.7, 0.7), (0.45, 0.95)]],
}


def _stamp(canvas, x, y, radius):
    low = max(0, int(x - radius)), max(0, int(y - radius))
    high = min(SIDE, int(x + radius) + 2), min(SIDE, int(y + radius) + 2)
    for yy in range(low[1], high[1]):
        for xx in range(low[0], high[0]):
            if (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2:
                canvas[yy, xx] = True


def render(digit, rng, thickness=2, jitter=0.02, offset=(0.0, 0.0), scale=1.0):
    """Draw one mouse-style digit and return a 28x28 boolean array."""
    canvas = np.zeros((SIDE, SIDE), dtype=bool)
    radius = max(0.7, thickness / 2.0)

    for stroke in STROKES[digit]:
        # Densify the polyline so the brush leaves a continuous line.
        points = []
        for i in range(len(stroke) - 1):
            x0, y0 = stroke[i]
            x1, y1 = stroke[i + 1]
            steps = max(2, int(math.hypot(x1 - x0, y1 - y0) * 60))
            for s in range(steps + 1):
                t = s / steps
                points.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))

        for x, y in points:
            jx = rng.gauss(0, jitter)
            jy = rng.gauss(0, jitter)
            px = (x + jx + offset[0]) * scale
            py = (y + jy + offset[1]) * scale
            _stamp(canvas, px * (SIDE - 1), py * (SIDE - 1), radius)
    return canvas


def make_set(per_digit=40, seed=0, thickness=(1, 4), jitter=0.02,
             offset=0.12, scale_range=(0.85, 1.05)):
    """Build a labelled set of freehand-style digits."""
    rng = random.Random(seed)
    images, labels = [], []
    for digit in range(10):
        for _ in range(per_digit):
            images.append(render(
                digit, rng,
                thickness=rng.uniform(*thickness),
                jitter=rng.uniform(0.0, jitter * 2),
                offset=(rng.uniform(-offset, offset), rng.uniform(-offset, offset)),
                scale=rng.uniform(*scale_range)))
            labels.append(digit)
    return images, labels


def show(image):
    return "\n".join("".join("#" if v else "." for v in row) for row in image)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="model/mnist_morph_h16.pt")
    parser.add_argument("--per-digit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-model", action="store_true",
                        help="Only print a sample, do not evaluate")
    args = parser.parse_args()

    images, labels = make_set(args.per_digit, args.seed)

    if args.no_model:
        for digit in range(10):
            print(f"digit {digit}:")
            print(show(images[digit * args.per_digit]))
            print()
        return

    import sys
    sys.path.insert(0, "python")
    import torch
    from digit_console import NetworkModel

    model = NetworkModel(args.model)
    correct = 0
    per_digit = {d: 0 for d in range(10)}
    for image, label in zip(images, labels):
        pixels = [255 if v else 0 for v in image.reshape(-1)]
        scores = model.analyse(pixels)[2]
        pred = max(range(10), key=lambda d: scores[d])
        per_digit[label] += (pred == label)
        correct += (pred == label)

    print(f"model   : {args.model}")
    print(f"samples : {len(images)} freehand-style digits")
    print(f"ink     : mean {np.mean([i.sum() for i in images]):.0f} px "
          f"(MNIST averages 96)")
    print(f"accuracy: {100.0 * correct / len(images):.2f}%\n")
    for digit in range(10):
        bar = "#" * round(20 * per_digit[digit] / args.per_digit)
        print(f"  {digit}: {per_digit[digit]:2d}/{args.per_digit}  {bar}")


if __name__ == "__main__":
    main()