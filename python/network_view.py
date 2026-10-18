"""Node-and-edge diagram of the ternary classifier.

Draws the actual computational graph rather than activity bars:

    784 inputs -> [16 hidden] x N -> 10 class scores

N is one or two hidden layers, read from the checkpoint, so the same view
renders either architecture.

The input layer is rendered as the 28 x 28 image grid, which makes it legible
as an image while still showing 784 nodes. Edges are drawn only from lit pixels
to the first hidden units that weight them non-zero, coloured by the sign of
the ternary weight: green for +1, red for -1. Later layers are drawn as a dense
web between their own units.

A caveat worth stating plainly: the hardware does not feed raw ink into the
first layer. ImagePacketReceiver delivers binary pixels, WalshHadamard.vhd
applies a block-8 rotation to the pixel bus first, and the stored W1 is the
matching rotated matrix. Ink is therefore mixed across units before the first
multiply, so the drawn edges indicate which drawn pixels matter to which unit
rather than reproducing the rotation term by term.
"""

GRID = 28
HIDDEN_MAX = 127

# Layout. WIDTH/HEIGHT must match the canvas created in digit_console.build_ui.
WIDTH, HEIGHT = 720, 460
IMG_CELL = 12
IMG_X, IMG_Y = 16, 58
OUT_X = 648
NODE_R = 9
N_HIDDEN = 16

# A full drawing lights ~173 pixels and, because W1 is only 31% dense, that
# produces ~914 edges (worst case ~1,467) drawn on every repaint. Stacked in a
# single 1px layer the result is an unreadable mat of lines that hides the
# nodes entirely. Only the strongest connections are kept, and each surviving
# one is drawn twice: a wide, very transparent "halo" underneath and a thin
# bright core on top. That preserves the sense of density while letting the
# nodes and their values stay legible on top of it.
EDGE_LIMIT = 260


def _blend(low, high, fraction):
    """Interpolate between two RGB triples and return a Tk colour string."""
    fraction = max(0.0, min(1.0, fraction))
    r, g, b = (int(low[i] + (high[i] - low[i]) * fraction) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _confidence(scores):
    """Softmax probability of the winning class.

    Kept as a standalone function because digit_console.analyse and
    NetworkView.analyse_cascade both need it and importing the console from the
    view would be circular.
    """
    import math
    top = max(scores)
    denominator = sum(math.exp(s - top) for s in scores)
    return 1.0 / denominator if denominator > 0 else 0.0


class NetworkView:
    """Draws the three-layer network on a Tk canvas."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.pixels = [0] * (GRID * GRID)
        self.rotated = [0] * (GRID * GRID)
        self.n_hidden = 16
        self.hidden = [0] * 16
        self.hidden2 = [0] * 16
        self.scores = [0.0] * 10
        self.w1 = None
        self.w2 = None
        self.hidden_layers = 1
        self.peak = 0
        self.hidden_max = HIDDEN_MAX

        # Column x positions depend on depth, so they are computed once here.
        self.hidden_x = []

        # Vertical room the node columns occupy, in pixels. This is deliberately
        # a different attribute from anything derived from the scores: an
        # earlier version reused one `span` name for both the pixel height and
        # the score range, so the first update() overwrote the geometry with a
        # score difference and squashed all 16 hidden nodes into an 8px band at
        # the top of the canvas. Geometry and data now have separate names.
        self.row_top = IMG_Y
        # Centre the node columns vertically against the image block so the two
        # read as peers rather than the outputs hanging off the bottom.
        self.row_span = HEIGHT - 2 * IMG_Y - 20

    # ------------------------------------------------------------- geometry
    def input_xy(self, index):
        row, col = divmod(index, GRID)
        return IMG_X + col * IMG_CELL + IMG_CELL / 2, IMG_Y + row * IMG_CELL + IMG_CELL / 2

    def set_depth(self, depth):
        """Place the hidden columns for `depth` hidden layers."""
        self.hidden_layers = max(1, min(2, depth))
        usable = OUT_X - (IMG_X + GRID * IMG_CELL) - 60
        step = usable / (self.hidden_layers + 1)
        self.hidden_x = [IMG_X + GRID * IMG_CELL + 40 + step * (i + 1)
                         for i in range(self.hidden_layers)]

    @property
    def n_hidden(self):
        """Number of hidden units in the column being drawn.

        A property rather than a plain attribute so a partially constructed
        instance -- the geometry tests build one via __new__ to avoid needing a
        Tk canvas -- still resolves to the default rather than raising.
        """
        return getattr(self, "_n_hidden", N_HIDDEN)

    @n_hidden.setter
    def n_hidden(self, value):
        self._n_hidden = int(value)

    def hidden_xy(self, index, layer=0):
        step = self.row_span / max(1, self.n_hidden - 1.0)
        return self.hidden_x[layer], self.row_top + step * index

    def output_xy(self, index):
        step = self.row_span / 9.0
        return OUT_X, self.row_top + step * index + step / 2

    # ---------------------------------------------------------------- draw
    def update(self, pixels, rotated, hidden, scores, w1,
               hidden2=None, w2=None, depth=1, hidden_scale=None,
               n_hidden=None):
        self.pixels, self.rotated = pixels, rotated
        self.hidden, self.w1 = hidden, w1
        # The float baseline has 256 hidden units against the ternary model's
        # 16, so the column count follows the model rather than a constant.
        # Node spacing is derived from this at draw time.
        if n_hidden:
            self.n_hidden = n_hidden
        self.hidden2 = hidden2 if hidden2 is not None else [0] * self.n_hidden
        self.w2 = w2
        # Node sizing bound. Defaults to HIDDEN_MAX for the ternary model,
        # whose activations are small integers; a float model supplies its own.
        self.hidden_max = hidden_scale if hidden_scale else HIDDEN_MAX
        self.set_depth(depth)
        lo, hi = min(scores), max(scores)
        score_span = (hi - lo) or 1.0
        self.scores = [((s - lo) / score_span, s) for s in scores]
        self.peak = max(range(10), key=lambda i: scores[i])
        self.redraw()

    def redraw(self):
        c = self.canvas
        c.delete("all")

        self._draw_headers()
        self._draw_edges()
        self._draw_layer_edges()
        self._draw_input_layer()
        for layer in range(self.hidden_layers):
            values = self.hidden if layer == self.hidden_layers - 1 else self.hidden2
            self._draw_hidden_layer(layer, values)
        self._draw_output_layer()

    def _draw_headers(self):
        c = self.canvas
        # Centre each label over the column it names. Anchoring the input
        # header at IMG_X put its midpoint on the canvas edge, so "784 inputs"
        # rendered as "4 inputs" with the first characters clipped away.
        c.create_text(IMG_X + (GRID * IMG_CELL) / 2, 22, text="784 inputs",
                      anchor="center", fill="#c8cedb", font=("Segoe UI", 9, "bold"))
        for layer in range(self.hidden_layers):
            label = f"{self.n_hidden} hidden" if self.hidden_layers == 1 else f"{self.n_hidden} hidden {layer + 1}"
            c.create_text(self.hidden_x[layer], 22, text=label, anchor="center",
                          fill="#c8cedb", font=("Segoe UI", 9, "bold"))
        c.create_text(OUT_X, 22, text="10 outputs", anchor="center",
                      fill="#c8cedb", font=("Segoe UI", 9, "bold"))
        c.create_text(IMG_X + (GRID * IMG_CELL) / 2, IMG_Y + GRID * IMG_CELL + 16,
                      text="drawn pixels  (lit nodes feed the first layer)",
                      anchor="center", fill="#7b8496", font=("Segoe UI", 8))

    def _draw_edges(self):
        """Lit pixel -> hidden unit, coloured by the sign of the ternary weight.

        Connections are ranked by the magnitude of the rotated input and only
        the strongest EDGE_LIMIT are drawn. Drawing all of them was the source
        of the "cluttered" look: ~914 overlapping 1px lines read as a solid
        block, so the hidden nodes and their activation values disappeared
        behind the very edges meant to explain them.
        """
        c = self.canvas
        if self.w1 is None:
            return
        peak_abs = max((abs(v) for v in self.rotated), default=1) or 1

        edges = []
        for index, pixel in enumerate(self.pixels):
            if not pixel:
                continue
            strength = abs(self.rotated[index]) / peak_abs
            if strength <= 0:
                continue
            x1, y1 = self.input_xy(index)
            for h in range(self.n_hidden):
                weight = self.w1[h][index]
                if weight == 0:
                    continue
                edges.append((strength, h, x1, y1, weight))

        # Rank by strength and by which hidden unit is most strongly driven, so
        # the surviving set spreads across units instead of all belonging to
        # whichever one happens to sit in the middle of the column.
        edges.sort(key=lambda e: (-e[0], e[1]))

        for strength, h, x1, y1, weight in edges[:EDGE_LIMIT]:
            x2, y2 = self.hidden_xy(h, 0)
            colour = (_blend((26, 32, 44), (64, 205, 130), strength) if weight > 0
                      else _blend((26, 32, 44), (226, 96, 96), strength))
            # One wide translucent halo and one thin bright core. The halo gives
            # the bundle visual weight; the core keeps it readable. Drawn as two
            # passes over the same coordinates, so the limit counts connections
            # rather than canvas items.
            c.create_line(x1, y1, x2, y2, fill=colour, width=3,
                          stipple="gray75", tags="edgeglow")
            c.create_line(x1, y1, x2, y2, fill=colour, width=1, tags="edgecore")

    def _draw_input_layer(self):
        c = self.canvas
        for index in range(GRID * GRID):
            x, y = self.input_xy(index)
            if self.pixels[index]:
                c.create_oval(x - 4, y - 4, x + 4, y + 4,
                              fill="#e8edf5", outline="#9aa4b8")
            else:
                c.create_oval(x - 2, y - 2, x + 2, y + 2,
                              fill="#2b3242", outline="#39415a")

    def _draw_layer_edges(self):
        """Edges between adjacent hidden layers and from the last one to output."""
        c = self.canvas
        last = self.hidden_layers - 1
        if self.hidden_layers == 2 and self.w2 is not None:
            peak = max((abs(v) for v in self.hidden), default=1) or 1
            for a in range(self.n_hidden):
                x1, y1 = self.hidden_xy(a, 0)
                strength = abs(self.hidden[a]) / peak
                for b in range(self.n_hidden):
                    if self.w2[b][a] == 0:
                        continue
                    x2, y2 = self.hidden_xy(b, 1)
                    colour = (_blend((26, 32, 44), (64, 205, 130), strength)
                              if self.w2[b][a] > 0
                              else _blend((26, 32, 44), (226, 96, 96), strength))
                    c.create_line(x1, y1, x2, y2, fill=colour, width=1)

        peak_out = max((abs(v) for v in self.hidden), default=1) or 1
        for b in range(self.n_hidden):
            x1, y1 = self.hidden_xy(b, last)
            # Drawn faint and thin: these 160 lines are structure, not the
            # story. The picture is supposed to be about the ink and the first
            # layer, so the output web stays in the background.
            for d in range(10):
                x2, y2 = self.output_xy(d)
                c.create_line(x1, y1, x2, y2, fill="#242b39", width=1,
                              stipple="gray50")

    def _draw_hidden_layer(self, layer=0, values=None):
        c = self.canvas
        values = self.hidden if values is None else values
        for h in range(self.n_hidden):
            x, y = self.hidden_xy(h, layer)
            value = values[h]
            fraction = value / self.hidden_max
            radius = NODE_R * (0.55 + 0.45 * fraction)
            colour = _blend((38, 45, 60), (88, 205, 140), fraction)
            c.create_oval(x - radius - 2, y - radius - 2,
                          x + radius + 2, y + radius + 2,
                          fill="#161b25", outline="#39415a")
            c.create_oval(x - radius, y - radius, x + radius, y + radius,
                          fill=colour, outline="")
            c.create_text(x, y + radius + 11, text=f"h{h}", fill="#8a93a5",
                          font=("Consolas", 7))
            # Activation value to the left of the node. With the hidden column
            # sitting at x=510 and node radius up to 9, this lands at ~495 --
            # clear of the node itself, which is what the screenshots showed
            # working, so it is kept rather than moved.
            c.create_text(x - radius - 7, y, text=str(value), anchor="e",
                          fill="#aab3c4", font=("Consolas", 7))

    def _draw_output_layer(self):
        c = self.canvas
        for digit in range(10):
            x, y = self.output_xy(digit)
            fraction, raw = self.scores[digit]
            won = (digit == self.peak)
            radius = NODE_R * (0.55 + 0.45 * fraction)
            if won:
                colour, edge = "#e8a13a", "#f5c46b"
            else:
                colour = _blend((40, 46, 60), (108, 124, 150), fraction)
                edge = "#39415a"
            c.create_oval(x - radius - 2, y - radius - 2,
                          x + radius + 2, y + radius + 2,
                          fill="#161b25", outline=edge)
            c.create_oval(x - radius, y - radius, x + radius, y + radius,
                          fill=colour, outline="")
            c.create_text(x, y + radius + 10, text=str(digit), fill="#8a93a5",
                          font=("Consolas", 8, "bold"))
            c.create_text(x + radius + 6, y, text=f"{raw:.0f}", anchor="w",
                          fill="#aab3c4", font=("Consolas", 7))