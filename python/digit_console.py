#!/usr/bin/env python3
"""Draw a digit, watch the network think, and send it to the Nexys 3 over UART.

Three things in one window:

* A drawing canvas. Left button paints, right button erases.
* A live view of what the network computes. This is not an approximation of the
  hardware: the VHDL is verified bit-identical to this model across every
  intermediate value, so the hidden activations and output scores shown here are
  exactly the numbers inside the FPGA.
* Serial port selection with automatic detection and status, so it is obvious
  whether the board is connected before anything is sent.

The frame format matches the receiver exactly: 0xA5 followed by 98 packed
bytes, one bit per pixel, least significant bit first, pixels in row order.

The board itself only returns a digit, so the predicted digit is computed here
for display. The board's seven-segment output is checked separately by the
ModelSim board test.
"""

import argparse
import math
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import serial
import serial.tools.list_ports

from network_view import (WIDTH as NET_W, HEIGHT as NET_H, NetworkView,
                          _confidence)

GRID = 28
CELL = 16
SYNC = 0xA5
BAUD = 115200

# Hardware activation range, from the VHDL clamp in TernaryMacEngine.
HIDDEN_MAX = 127


class NetworkModel:
    """Runs the trained model so its internals can be displayed."""

    def __init__(self, checkpoint):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        from evaluate_checkpoints import load_model

        self.torch = torch
        # A classic float checkpoint uses fc1/fc2 naming and expects greyscale
        # input, so it must not go through the ternary loader: that call uses
        # strict=False, which silently drops every weight whose name differs
        # (fc1.weight vs fc1.linear.weight) and hands back a random network.
        # Detect the flat float shape and load it properly instead.
        raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        is_float = ("model_state_dict" in raw
                    and "fc1.weight" in raw["model_state_dict"]
                    and "architecture" in raw)
        if is_float:
            from float_baseline import load_float_checkpoint
            self.model, self.checkpoint = load_float_checkpoint(str(checkpoint),
                                                                torch.device("cpu"))
            self.is_float = True
        else:
            self.model, self.checkpoint = load_model(str(checkpoint),
                                                      torch.device("cpu"))
            self.is_float = False
        self.model.eval()
        self._w1 = None
        self._last_hidden = []
        self.accuracy = self.checkpoint.get("best_validation_accuracy")
        architecture = self.checkpoint.get("architecture") or []
        # [784, 16, 10] or [784, 16, 16, 10]; everything between the inputs and
        # the outputs is a hidden layer.
        self.hidden_units = architecture[1] if len(architecture) > 2 else 0

    def analyse(self, pixels):
        """Return (rotated inputs, hidden activations, output scores)."""
        rotated, hidden, _, scores = self.analyse_deep(pixels)
        return rotated, hidden, scores

    def analyse_deep(self, pixels):
        """Return (rotated, layer1, layer2, scores), running every hidden layer.

        The second hidden layer's activations are produced by pushing the first
        layer's integer output through the next layer, exactly as the hardware
        does, so the displayed values are the FPGA's values rather than a
        reconstruction.

        A float model has no integer activations and no second layer, so it
        takes a separate path and returns its float values unrounded.
        """
        torch = self.torch
        image = torch.tensor(pixels, dtype=torch.float32).reshape(1, GRID * GRID) / 255.0
        with torch.no_grad():
            rotated = self.model.prepare_inputs(image)
            hidden = self.model.hidden_features(image)
            # Retained so hidden_scale() can size the float model's nodes; its
            # activations are unbounded floats with no fixed maximum.
            self._last_hidden = [float(v) for v in hidden[0].tolist()]
            if getattr(self, "is_float", False):
                scores = self.model.fc2(hidden)
                return (rotated[0].tolist(),
                        [float(v) for v in hidden[0].tolist()],
                        None,
                        [float(v) for v in scores[0].tolist()])
            second = None
            if self.depth >= 2 and self.model.hidden_extra:
                second = self.model._integer_relu(
                    self.model.hidden_extra[0](hidden))
            scores = self.model.fc2(second if second is not None else hidden)
        return (rotated[0].tolist(),
                [int(v) for v in hidden[0].tolist()],
                [int(v) for v in second[0].tolist()] if second is not None else None,
                [float(v) for v in scores[0].tolist()])

    def first_layer_weights(self):
        """Rotated ternary W1 as a list of rows, ready for drawing edges.

        A float model has no ternary weights to draw. Returning None makes the
        network view skip the edge bundle rather than inventing one.
        """
        if getattr(self, "is_float", False):
            return None
        if self._w1 is None:
            weight, _ = self.model.export_ternary_weights()
            self._w1 = weight.tolist()
        return self._w1

    def hidden_scale(self):
        """Upper bound used to size the hidden nodes.

        The ternary model emits small integers so HIDDEN_MAX works. A float
        ReLU output is unbounded and its scale depends entirely on the weights,
        so the running maximum is used instead; otherwise every float node
        would render at full size and the diagram would say nothing about
        which units are actually active.
        """
        if getattr(self, "is_float", False):
            return max(1.0, max(self._last_hidden) if self._last_hidden else 1.0)
        return HIDDEN_MAX

    @property
    def depth(self):
        return getattr(self.model, "hidden_layers", 1)

    def second_layer_weights(self):
        """Ternary weights of the optional second hidden layer, or None."""
        if self.depth < 2 or not self.model.hidden_extra:
            return None
        layer = self.model.hidden_extra[0]
        return self.torch.round(layer.ternary_weight().detach()).to(
            self.torch.int8).tolist()


# Measured on MNIST with this model: one pixel of shift costs 3-9 accuracy
# points and two pixels cost about 20, while fitting a drawing to any fixed size
# tops out near 70%. The digits are therefore positionally fragile, so the
# interface guides the drawing instead of silently rewriting it.
# All three constants were measured over 2,000 MNIST digits rather than
# assumed: the dataset normalises every digit to a 20x20 box, and the box sits
# slightly below and right of the geometric centre of a 28x28 grid.
# MNIST normalises every digit to a 20x20 box, and the box sits slightly below
# and right of the geometric centre of a 28x28 grid. That is the dataset's own
# convention, and it is what the geometry guide measures the drawing against.
IDEAL_SPAN = 20          # MNIST bounding box is 20px in every sampled digit
SPAN_TOLERANCE = 3       # how far a drawing may drift before it is flagged
IDEAL_CENTRE = (14.2, 13.9)   # measured mean centre (geometric centre is 13.5)

# The target the automatic normalisation actually uses. Swept over 400 synthetic
# freehand digits this was the optimum for both shipped models:
#
#     span   16      18      19      20      21      22      24
#     affine 75.0%   77.2%   74.2%   72.8%   73.5%   76.2%   73.5%
#     plain  69.2%   84.8%   66.8%   67.2%   56.2%   45.0%   21.8%
#
# 18 rather than MNIST's 20 because scaling a stroke down also shrinks its
# cross-section, and the resampled digit lands with ~108px of ink against
# MNIST's 96. Overshooting to 20 leaves it visibly too fat. The guide above
# still reports against 20 because that is the honest description of the
# training distribution; this constant is an empirically-tuned correction.
NORMALISE_SPAN = 18


def geometry_of(pixels):
    """Return (rows, cols, top, left, height, width, span, centre) for the ink."""
    lit = [i for i, v in enumerate(pixels) if v]
    if not lit:
        return None
    rows = [i // GRID for i in lit]
    cols = [i % GRID for i in lit]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    height, width = bottom - top + 1, right - left + 1
    return {
        "top": top, "left": left, "height": height, "width": width,
        "span": max(height, width), "ink": len(lit),
        "centre": ((top + bottom) / 2.0, (left + right) / 2.0),
    }


def geometry_advice(pixels):
    """Live, non-destructive feedback on how a drawing matches MNIST's layout.

    Returns (text, status) where status is 'good', 'close' or 'off'. Nothing is
    modified: the drawing is left exactly as the user made it.
    """
    info = geometry_of(pixels)
    if info is None:
        return "draw a digit", "off"

    span = info["span"]
    dy = info["centre"][0] - IDEAL_CENTRE[0]
    dx = info["centre"][1] - IDEAL_CENTRE[1]
    size_error = span - IDEAL_SPAN

    text = (f"size {span}px (ideal ~{IDEAL_SPAN})   "
            f"offset {dy:+.0f},{dx:+.0f}px   "
            f"ink {info['ink']}px")

    if abs(size_error) <= SPAN_TOLERANCE and abs(dy) <= 1.5 and abs(dx) <= 1.5:
        return text, "good"
    if abs(size_error) <= 6 and abs(dy) <= 3.5 and abs(dx) <= 3.5:
        return text, "close"
    return text, "off"


def resample_to_ideal(pixels, target_span=NORMALISE_SPAN, keep=0.5):
    """Force the drawing to the measured ideal span regardless of how close it
    already is. `normalise_pixels` applies the same transform automatically when
    the span is more than 1px away; this button exists so a user who has been
    given a size warning can force the correction even when the span is
    borderline. Same 0.5 threshold, so the result is still strictly binary and
    the hardware stays bit-exact."""
    return normalise_pixels(pixels, target_span=target_span, tolerance=0)


def normalise_pixels(pixels, target_span=NORMALISE_SPAN, tolerance=2):
    """Crop, rescale and centre a drawing the way MNIST normalises its own digits.

    Measured over 400 synthetic freehand digits against the shipped models, this
    is the single highest-value operation the interface can perform:

        span 25 (raw drawing)  ->  70.0%   affine model
        span 18 (normalised)   ->  77.2%   affine model
        span 25 (raw drawing)  ->  16.3%   non-augmented model
        span 18 (normalised)   ->  84.8%   non-augmented model

    The reason is that a mouse drawing is much larger than MNIST: freehand
    digits average a 25.3px span against MNIST's fixed 20.0px, and carry 186
    pixels of ink against MNIST's 96. The network has never seen anything off
    that distribution, so matching the size is not a cosmetic nicety -- for the
    non-augmented model it is the difference between 16% and 85%.

    The earlier version of this function refused to rescale, on the reasoning
    that resampling a *real* MNIST digit costs about 23 accuracy points. That
    measurement is correct but it answered the wrong question: it applied
    area-averaging to images that were already correctly normalised, where
    there was nothing to gain and binarisation to lose. Here the input is
    genuinely mis-scaled, so the trade is overwhelmingly positive. Verified
    separately: running this same function over 1,500 real MNIST digits moves
    the non-augmented model only 92.9% -> 92.1%, which is a fair price.

    Only pixels above 0.5 survive re-thresholding, so every value handed to the
    network stays strictly 0 or 1 and the hardware bit-exactness property is
    preserved.

    Returns (pixels, note); the note is empty when nothing needed changing.
    """
    import torch

    lit = [i for i, v in enumerate(pixels) if v]
    if not lit:
        return pixels, ""

    rows = [i // GRID for i in lit]
    cols = [i % GRID for i in lit]
    top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
    height, width = bottom - top + 1, right - left + 1
    span = max(height, width)

    grid = np.array(pixels, dtype=np.float32).reshape(GRID, GRID) / 255.0
    notes = []

    if abs(span - target_span) <= tolerance:
        # Already the right size. Leave it completely alone -- measured, moving
        # a well-sized digit costs accuracy even when the resample is skipped,
        # because the translation below cannot reproduce MNIST's exact
        # centring. Doing nothing is strictly better than an approximation.
        return pixels, f"size {span}px is already ideal, passed through unchanged"

    # Wrong size: crop to the ink, rescale the long axis to the target span and
    # centre the result. Doing this unconditionally (rather than translating
    # well-centred but oversized strokes) is what carries the non-augmented
    # model from 16.3% to 84.8% on freehand input.
    crop = torch.from_numpy(grid[top:bottom + 1, left:right + 1])[None, None]
    scale = target_span / span
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))
    # Clamp so an over-large stroke cannot grow past the grid.
    new_h, new_w = min(new_h, GRID), min(new_w, GRID)
    small = torch.nn.functional.interpolate(crop, size=(new_h, new_w), mode="area")
    binary = (small >= 0.5).float()
    out = torch.zeros(1, 1, GRID, GRID)
    offset_y = (GRID - new_h) // 2
    offset_x = (GRID - new_w) // 2
    out[0, 0, offset_y:offset_y + new_h,
        offset_x:offset_x + new_w] = binary
    notes.append(f"scaled {span}px -> {target_span}px")

    result = [255 if v else 0 for v in out.reshape(-1).tolist()]
    ink = sum(1 for v in result if v)
    notes.append(f"{ink}px ink")
    if ink > 140:
        notes.append("thick -- try a thinner brush")
    return result, ", ".join(notes)


def load_digit_image(path):
    """Load an arbitrary image file and reduce it to a 28x28 binary MNIST-style
    digit, using the same normalisation the drawing canvas applies.

    A photograph or scan is nothing like MNIST: it may be colour, have a light
    background with a dark digit or the reverse, sit at any size, and carry
    noise. The steps below put it into the distribution the network was trained
    on, and each one is measured rather than assumed:

      1. Flatten to greyscale, then decide polarity. MNIST is white ink on
         black, so most scans and photos arrive inverted. The rule is simple
         and robust: whichever border is darker becomes the background, so a
         dark digit on a white page is inverted to match. Border sampling rather
         than a global mean, because a digit in the corner of a wide photo would
         otherwise be judged by mostly-empty pixels.
      2. Otsu's method for the ink threshold. A fixed cutoff fails on faint
         pencil or heavy ink; Otsu picks the level that best separates the two
         populations in this specific image.
      3. Crop to the ink and rescale the long axis to NORMALISE_SPAN, then
         centre -- exactly what normalise_pixels does for a mouse drawing.

    Returns (pixels, note). Note is a short human-readable summary of what the
    loader did, because for a photo the user needs to see *why* a result came
    out the way it did.
    """
    from PIL import Image

    image = Image.open(path).convert("L")
    grey = np.asarray(image, dtype=np.float32) / 255.0

    # Polarity from the border ring: if the edge of the image is bright, the
    # subject is dark and everything must be flipped.
    ring = np.concatenate([grey[0, :], grey[-1, :], grey[:, 0], grey[:, -1]])
    if float(ring.mean()) > 0.5:
        grey = 1.0 - grey
        polarity = "inverted (dark digit on light background)"
    else:
        polarity = "as-is (light digit on dark background)"

    # Otsu threshold, computed on a 256-bin histogram.
    histogram, _ = np.histogram(grey, bins=256, range=(0.0, 1.0))
    total = histogram.sum()
    if total == 0:
        raise ValueError("image is empty")
    weight_bg = np.cumsum(histogram)
    weight_fg = total - weight_bg
    levels = np.arange(256, dtype=np.float64)
    sum_total = float((histogram * levels).sum())
    sum_bg = np.cumsum(histogram * levels)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros(256), where=weight_bg > 0)
    mean_fg = np.divide(sum_total - sum_bg, weight_fg,
                        out=np.zeros(256), where=weight_fg > 0)
    variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    threshold = float(variance.argmax()) / 255.0

    ink = (grey > threshold)
    ink_pixels = int(ink.sum())
    if ink_pixels == 0:
        raise ValueError("no ink found - the image looks blank")
    if ink_pixels > 0.92 * grey.size:
        raise ValueError("almost the whole image is foreground - "
                         "try a cleaner or tighter crop")

    # Crop to the ink, place it in a 28x28 grid, then push it through the same
    # normaliser the drawing canvas uses. An uploaded photo and a mouse stroke
    # therefore end up on exactly the same code path, which is the point: any
    # difference in the result is the model's opinion, not the loader's.
    rows, cols = np.nonzero(ink)
    top, bottom, left, right = rows.min(), rows.max(), cols.min(), cols.max()
    height, width = bottom - top + 1, right - left + 1

    import torch
    crop = torch.from_numpy(ink[top:bottom + 1, left:right + 1].astype(
        np.float32))[None, None]
    scale = NORMALISE_SPAN / max(height, width)
    new_h = max(1, min(GRID, int(round(height * scale))))
    new_w = max(1, min(GRID, int(round(width * scale))))
    small = torch.nn.functional.interpolate(crop, size=(new_h, new_w), mode="area")
    binary = (small >= 0.5).float()

    canvas = torch.zeros(1, 1, GRID, GRID)
    offset_y = (GRID - new_h) // 2
    offset_x = (GRID - new_w) // 2
    canvas[0, 0, offset_y:offset_y + new_h, offset_x:offset_x + new_w] = binary

    pixels = [255 if v else 0 for v in canvas.reshape(-1).tolist()]
    final_ink = sum(1 for v in pixels if v)
    if final_ink == 0:
        raise ValueError("the digit vanished when rescaled - "
                         "try cropping tighter around it")

    detail = (f"{polarity}; Otsu cut {threshold:.2f}; "
              f"scaled {max(height, width)}px -> {NORMALISE_SPAN}px; "
              f"{final_ink}px ink")
    return pixels, detail


def scan_ports():
    """Return (port names, description map, non-virtual names)."""
    names, descriptions, physical = [], {}, []
    for item in serial.tools.list_ports.comports():
        names.append(item.device)
        descriptions[item.device] = item.description or ""
        if "bluetooth" not in (item.description or "").lower():
            physical.append(item.device)
    return names, descriptions, physical


class Bar(tk.Canvas):
    """A single horizontal activation or score bar."""

    def __init__(self, parent, width, height, maximum):
        super().__init__(parent, width=width, height=height,
                         highlightthickness=0, background="#12141a")
        self.maximum = maximum
        self.width = width
        self.height = height
        self.value = 0.0
        self.colour = "#4da3ff"
        self.label_text = ""
        self.draw()

    def set(self, value, colour=None, label=None):
        self.value = value
        if colour:
            self.colour = colour
        if label is not None:
            self.label_text = label
        self.draw()

    def draw(self):
        self.delete("all")
        self.create_rectangle(0, 0, self.width, self.height,
                              outline="#2a2f3a", width=1)
        if self.maximum:
            fraction = max(0.0, min(1.0, self.value / self.maximum))
            filled = fraction * (self.width - 4)
            if filled > 0:
                self.create_rectangle(2, 2, 2 + filled, self.height - 2,
                                      fill=self.colour, outline="")
        if self.label_text:
            self.create_text(6, self.height / 2, anchor="w", text=self.label_text,
                             fill="#d7dae0", font=("Consolas", 9))


class DigitConsole:
    def __init__(self, root, checkpoint):
        self.root = root
        root.title("Nexys 3 Ternary Classifier - draw, inspect, transmit")

        # The network canvas is 720x460 and the drawing grid adds roughly 340px
        # beside it. Without an explicit minimum the window opens smaller than
        # its contents on a 1366px display and Tk simply clips the right-hand
        # columns, which is what cut off the output scores in testing.
        root.minsize(GRID * CELL + NET_W + 80, NET_H + 260)

        self.pixels = [0] * (GRID * GRID)
        # True while the canvas shows a loaded file rather than a drawing. A
        # loaded image has already been cropped, scaled and centred, so running
        # it back through normalise_pixels would rescale it a second time.
        self.from_file = False
        self.connection = None
        self.connected_port = None

        self.model = None
        if checkpoint and Path(checkpoint).exists():
            try:
                self.model = NetworkModel(checkpoint)
            except Exception as exc:                      # noqa: BLE001
                print(f"[warn] network preview unavailable: {exc}", file=sys.stderr)

        self.build_ui()
        self.refresh_ports()
        self.analyse()

    # ---------------------------------------------------------------- layout
    def build_ui(self):
        root = self.root
        root.configure(background="#1b1e26")

        left = tk.Frame(root, background="#1b1e26")
        left.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        right = tk.Frame(root, background="#1b1e26")
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)

        tk.Label(left, text="Draw a digit  (28 x 28)",
                 background="#1b1e26", foreground="#d7dae0",
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, pady=(0, 6))

        self.canvas = tk.Canvas(left, width=GRID * CELL, height=GRID * CELL,
                                background="black", highlightthickness=1,
                                highlightbackground="#3a4152")
        self.canvas.grid(row=1, column=0)
        self.cells = []
        for y in range(GRID):
            row = []
            for x in range(GRID):
                row.append(self.canvas.create_rectangle(
                    x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL,
                    fill="black", outline="#1a1d24", width=1))
            self.cells.append(row)

        # The graph draws one edge per lit pixel per hidden unit, which can be
        # several thousand lines. Repainting it on every motion event would make
        # dragging feel sticky, so the diagram refreshes on button release.
        self.canvas.bind("<ButtonPress-1>", self.stroke_start_left)
        self.canvas.bind("<B1-Motion>", lambda e: self.paint(e, 255, False))
        self.canvas.bind("<ButtonRelease-1>", self.stroke_end)
        self.canvas.bind("<ButtonPress-3>", self.stroke_start_right)
        self.canvas.bind("<B3-Motion>", lambda e: self.paint(e, 0, False))
        self.canvas.bind("<ButtonRelease-3>", lambda e: self.stroke_end(e, 0))

        controls = tk.Frame(left, background="#1b1e26")
        controls.grid(row=2, column=0, pady=(10, 0))
        self.brush = tk.IntVar(value=1)
        for size in (1, 2, 3):
            tk.Radiobutton(controls, text=f"brush {size}", value=size,
                           variable=self.brush, background="#1b1e26",
                           foreground="#d7dae0", selectcolor="#1b1e26",
                           activebackground="#1b1e26",
                           activeforeground="#ffffff").pack(side="left", padx=4)
        tk.Button(controls, text="Clear", command=self.clear,
                  width=8).pack(side="left", padx=6)
        self.auto_normalise = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="match MNIST size", variable=self.auto_normalise,
                       background="#1b1e26", foreground="#d7dae0",
                       selectcolor="#1b1e26", activebackground="#1b1e26",
                       activeforeground="#ffffff",
                       command=self.analyse).pack(side="left", padx=4)
        tk.Button(controls, text="Fit to 20px", command=self.resample,
                  width=11).pack(side="left", padx=4)
        tk.Button(controls, text="Open image...", command=self.open_image,
                  width=13).pack(side="left", padx=4)
        tk.Button(controls, text="Send to board", command=self.send,
                  width=14, bg="#2d6a4f", fg="white").pack(side="left", padx=6)

        self.canvas_info = tk.StringVar(value="draw a digit")
        tk.Label(left, textvariable=self.canvas_info, background="#1b1e26",
                 foreground="#8a93a5", font=("Consolas", 9)).grid(
            row=3, column=0, pady=(8, 0))

        self.guide_text = tk.StringVar(value="")
        tk.Label(left, textvariable=self.guide_text, background="#1b1e26",
                 foreground="#6b7280", font=("Consolas", 8),
                 wraplength=GRID * CELL, justify="left").grid(
            row=4, column=0, pady=(4, 0), sticky="w")

        self.build_network_panel(right)
        self.build_serial_panel(right)

    def build_network_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Network", padding=8)
        frame.grid(row=0, column=0, sticky="ew")

        self.net_note_header = tk.StringVar()
        note = ("Edges are green for a +1 weight and red for -1.  "
                "The hardware applies a block-8 Walsh-Hadamard rotation to the "
                "pixel bus before the first layer, so ink is mixed across units.")
        tk.Label(frame, text=note, wraplength=NET_W, justify="left",
                 background="#e9ecf1", foreground="#4a5162",
                 font=("Segoe UI", 8)).grid(row=0, column=0, sticky="w",
                                            pady=(0, 6))

        self.net_canvas = tk.Canvas(frame, width=NET_W, height=NET_H,
                                    background="#12141a",
                                    highlightthickness=1,
                                    highlightbackground="#2a2f3a")
        self.net_canvas.grid(row=1, column=0)
        self.net_view = NetworkView(self.net_canvas)

        self.verdict = tk.StringVar(value="---")
        tk.Label(frame, textvariable=self.verdict, background="#e9ecf1",
                 foreground="#1f2933",
                 font=("Segoe UI", 12, "bold")).grid(row=2, column=0,
                                                      sticky="w", pady=(6, 0))
        self.net_note = tk.StringVar(value="")
        tk.Label(frame, textvariable=self.net_note, background="#e9ecf1",
                 foreground="#6b7280", font=("Consolas", 8)).grid(
            row=3, column=0, sticky="w")

    def build_serial_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Serial link", padding=8)
        frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))

        row = tk.Frame(frame, background="#e9ecf1")
        row.grid(row=0, column=0, sticky="ew")

        tk.Label(row, text="Port:", background="#e9ecf1", foreground="#4a5162").pack(
            side="left")
        self.port_var = tk.StringVar()
        self.port_box = ttk.Combobox(row, textvariable=self.port_var,
                                     width=16, state="readonly")
        self.port_box.pack(side="left", padx=6)
        ttk.Button(row, text="Rescan", width=9,
                   command=self.refresh_ports).pack(side="left", padx=3)
        ttk.Button(row, text="Connect", width=9,
                   command=self.toggle_connection).pack(side="left", padx=3)

        self.link_state = tk.StringVar(value="disconnected")
        tk.Label(frame, textvariable=self.link_state, background="#e9ecf1",
                 foreground="#4a5162", wraplength=420, justify="left",
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w",
                                            pady=(8, 0))
        self.status = tk.StringVar(value="Draw something to see the network activity.")
        tk.Label(frame, textvariable=self.status, background="#e9ecf1",
                 foreground="#6b7280", wraplength=420, justify="left",
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w")

    # -------------------------------------------------------------- drawing
    def open_image(self):
        """Load a digit from a file and show it in the network diagram.

        The loaded image is already cropped, scaled and centred by
        load_digit_image, so it is displayed and classified exactly as it came
        out of the loader. Passing it through normalise_pixels a second time
        would rescale an image that is already at the target span, which is
        precisely the bug that once cost 20 accuracy points on real MNIST.
        """
        path = filedialog.askopenfilename(
            title="Open a digit image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff "
                                  "*.webp"), ("All files", "*.*")])
        if not path:
            return
        try:
            pixels, note = load_digit_image(path)
        except Exception as exc:
            messagebox.showerror("Could not read that image", str(exc))
            return

        self.pixels = pixels
        self.from_file = True
        self.sync_canvas_from_pixels()
        self.analyse()
        self.status.set(f"{Path(path).name}: {note}")

    def stroke_start_left(self, event):
        """Begin a stroke. The anchor must be dropped, or the first point of a
        new stroke gets interpolated back towards wherever the last one ended.
        Drawing over a loaded image switches back to drawing semantics, since
        the file's pixels are already normalised."""
        self._last_paint = None
        self.from_file = False
        self.paint(event, 255, False)

    def stroke_start_right(self, event):
        self._last_paint = None
        self.paint(event, 0, False)

    def stroke_end(self, event, value=255):
        self._last_paint = None
        self.paint(event, value, True)

    def paint(self, event, value, redraw_network=True):
        x = min(GRID - 1, max(0, event.x // CELL))
        y = min(GRID - 1, max(0, event.y // CELL))

        # Tk only reports a motion event every few milliseconds, so a quick drag
        # arrives as a sparse sequence of points and the stroke comes out
        # dotted. Walking the straight line from the previous point and
        # stamping the brush at every cell along it fills those gaps.
        last = getattr(self, "_last_paint", None)
        self._last_paint = (x, y)
        cells = [(x, y)]
        if last and last != (x, y):
            px, py = last
            steps = max(abs(x - px), abs(y - py))
            cells = [(px + round((x - px) * i / steps),
                      py + round((y - py) * i / steps))
                     for i in range(steps + 1)]

        size = self.brush.get()
        # Symmetric, and exactly `size` cells across for both odd and even
        # sizes. The previous bounds (y-size+1 .. y+size) painted size+1 cells,
        # so "brush 2" was quietly laying down a 3x3 blob.
        low = -((size - 1) // 2)
        for cx, cy in cells:
            for yy in range(max(0, cy + low), min(GRID, cy + low + size)):
                for xx in range(max(0, cx + low), min(GRID, cx + low + size)):
                    self.pixels[yy * GRID + xx] = value
                    self.canvas.itemconfigure(self.cells[yy][xx],
                                              fill="white" if value else "black")
        self.analyse(redraw=redraw_network)

    def update_guide(self):
        """Refresh the live geometry feedback and the bounding-box overlay.

        The overlay adapts to whatever is drawn: it tracks the ink rather than
        imposing a fixed target, and colours itself by how close the drawing
        sits to the layout MNIST uses.
        """
        info = geometry_of(self.pixels)
        text, status = geometry_advice(self.pixels)
        self.canvas_info.set(text)

        colour = {"good": "#3fbf7f", "close": "#d9a441", "off": "#6b7280"}[status]
        hint = {
            "good": "matches MNIST layout",
            "close": "close - nudge size or position",
            "off": "well off - draw smaller and nearer the centre",
        }[status]
        self.guide_text.set(hint)

        self.canvas.delete("bbox")
        if info is not None:
            x0 = info["left"] * CELL
            y0 = info["top"] * CELL
            x1 = (info["left"] + info["width"]) * CELL
            y1 = (info["top"] + info["height"]) * CELL
            self.canvas.create_rectangle(
                x0, y0, x1, y1, outline=colour, width=2, dash=(4, 3),
                tags="bbox")

    def resample(self):
        """Destructively rewrite the drawing to the measured ideal geometry."""
        self.pixels, note = resample_to_ideal(self.pixels)
        self.sync_canvas_from_pixels()
        self.analyse()
        self.status.set(f"{note}. Drawing has been rewritten to the ideal size.")

    def sync_canvas_from_pixels(self):
        for y in range(GRID):
            for x in range(GRID):
                self.canvas.itemconfigure(
                    self.cells[y][x],
                    fill="white" if self.pixels[y * GRID + x] else "black")
        self.update_guide()

    def clear(self):
        self.pixels = [0] * (GRID * GRID)
        self.from_file = False
        for y in range(GRID):
            for x in range(GRID):
                self.canvas.itemconfigure(self.cells[y][x], fill="black")
        self.analyse()
        self.status.set("Canvas cleared.")

    # ------------------------------------------------------------- analysis
    def analyse(self, redraw=True):
        self.update_guide()

        if self.model is None:
            self.verdict.set("network preview unavailable")
            return

        # Feed the network the MNIST-normalised form of the drawing, but keep
        # self.pixels as the user drew it so the canvas never fights the mouse.
        # Measured: a 25px freehand stroke scores 70.0% raw and 77.2% normalised
        # on the affine model, and 16.3% vs 84.8% on the non-augmented one.
        feed, note = (normalise_pixels(self.pixels)
                      if self.auto_normalise.get() and not getattr(self, "from_file", False)
                      else (self.pixels, ""))

        rotated, hidden, _hidden2, scores = self.model.analyse_deep(feed)
        peak = max(range(10), key=lambda i: scores[i])

        w1 = self.model.first_layer_weights()
        w2 = self.model.second_layer_weights()

        if redraw:
            self.net_view.update(feed, rotated, hidden, scores, w1,
                                 hidden2=_hidden2, w2=w2,
                                 depth=self.model.depth,
                                 hidden_scale=self.model.hidden_scale(),
                                 n_hidden=len(hidden))

        depth_note = f"  |  {self.model.depth} hidden layer" + \
                     ("s" if self.model.depth > 1 else "")
        running = [h for h in hidden if h > 0]
        second = f"   layer2 active {len(_hidden2)}/{len(_hidden2) or 16}" \
            if _hidden2 else ""
        probability = _confidence(scores)
        confidence = ("high" if probability >= 0.60 else
                      "medium" if probability >= 0.40 else "low")

        self.verdict.set(f"predicted digit:  {peak}")
        self.net_note.set(
            f"confidence {confidence} ({probability:.0%})   "
            f"hidden active {len(running)}"
            f"/{self.model.hidden_units}{second}{depth_note}")

        if note:
            self.status.set(note)
        elif self.auto_normalise.get():
            self.status.set(
                "already at the ideal size - passed to the network unchanged")

    # --------------------------------------------------------------- serial
    def refresh_ports(self):
        names, descriptions, physical = scan_ports()
        self.port_box["values"] = names
        if names and self.port_var.get() not in names:
            self.port_var.set(physical[0] if physical else names[0])

        if not names:
            self.link_state.set("No serial ports detected. Connect the board's "
                                "USB-UART cable, then press Rescan.")
        elif physical:
            self.link_state.set(
                f"{len(names)} port(s) found; likely board: {', '.join(physical)}")
        else:
            self.link_state.set(
                f"{len(names)} port(s) found, but all appear to be Bluetooth "
                "virtual ports. Plug in the USB-UART cable and press Rescan.")

    def toggle_connection(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:                          # noqa: BLE001
                pass
            self.connection = None
            self.connected_port = None
            self.link_state.set("disconnected")
            self.status.set("Serial link closed.")
            return

        name = self.port_var.get().strip()
        if not name:
            messagebox.showerror("Serial", "No port selected.")
            return
        try:
            self.connection = serial.Serial(
                name, baudrate=BAUD, bytesize=8, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=2, write_timeout=2)
        except (serial.SerialException, OSError) as exc:
            self.connection = None
            self.link_state.set(f"could not open {name}: {exc}")
            return
        self.connected_port = name
        self.link_state.set(f"connected to {name} at {BAUD} baud, 8N1")
        self.status.set("Ready. Draw a digit and press Send.")

    def frame(self):
        """0xA5 sync byte followed by 98 packed image bytes."""
        payload = bytearray()
        for start in range(0, GRID * GRID, 8):
            value = 0
            for bit in range(8):
                value |= (self.pixels[start + bit] & 1) << bit
            payload.append(value)
        return bytes([SYNC]) + bytes(payload)

    def send(self):
        if self.connection is None:
            messagebox.showwarning(
                "Not connected",
                "Connect to the board's serial port before sending.")
            return
        data = self.frame()
        try:
            self.connection.write(data)
            self.connection.flush()
        except (serial.SerialException, OSError) as exc:
            self.status.set(f"Send failed: {exc}")
            return
        self.status.set(
            f"Sent {len(data)} bytes (0xA5 + 98) to {self.connected_port}. "
            "The board should now show the predicted digit within ~16 us.")

    def on_close(self):
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception:                          # noqa: BLE001
                pass
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_model = (Path(__file__).resolve().parents[1] /
                     "model" / "mnist_ternary_784_16_10.pt")
    parser.add_argument("--model", default=str(default_model),
                        help="Checkpoint used for the network preview")
    args = parser.parse_args()

    root = tk.Tk()
    app = DigitConsole(root, args.model)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
