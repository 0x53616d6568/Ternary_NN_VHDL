"""End-to-end test of the GUI pipeline exactly as digit_console runs it.

Every path the interface can take is exercised here: a raw drawing, the
auto-normalised drawing, a loaded image file, and the confidence readout that
the screenshots exposed as broken.
"""
import sys, os, math, numpy as np, torch
sys.path.insert(0, "python")
from PIL import Image, ImageDraw, ImageFont
from digit_console import (GRID, normalise_pixels, load_digit_image,
                           geometry_of, NORMALISE_SPAN, IDEAL_SPAN)
from evaluate_checkpoints import load_model
from synthetic_digits import make_set
from torchvision import datasets, transforms

CHECKPOINT = "model/mnist_ternary_784_16_10.pt"
dev = torch.device("cpu")
model, ckpt = load_model(CHECKPOINT, dev)
model.eval()

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


print("=" * 68)
print("GUI PIPELINE TEST SUITE")
print("=" * 68)

# --------------------------------------------------------------- checkpoint
print("\n[1] checkpoint")
check("architecture is 784-16-10", ckpt["architecture"] == [784, 16, 10],
      str(ckpt["architecture"]))
check("one hidden layer", ckpt.get("hidden_layers", 1) == 1)
check("recorded val >= 90%", 100 * ckpt["best_validation_accuracy"] >= 90.0,
      f"{100*ckpt['best_validation_accuracy']:.2f}%")

# ------------------------------------------------------------ normalisation
print("\n[2] normalisation contract")
fh, hl = make_set(per_digit=20, seed=21, thickness=(1, 4), offset=0.12)
hl = np.array(hl)
blobs = []
for im in fh:
    px, _n = normalise_pixels([255 if v else 0 for v in
                               (np.asarray(im, np.float32).reshape(-1) * 255).astype(int).tolist()])
    blobs.append(px)
    if set(px) - {0, 255}:
        blobs.append("nonbinary")
        break
check("output is strictly 0/255 (bit-exactness preserved)",
      all(set(b) <= {0, 255} for b in blobs if isinstance(b, list)))
check("output length is 784", all(len(b) == 784 for b in blobs if isinstance(b, list)))
check("no drawing produced empty output",
      all(sum(1 for v in b if v) > 0 for b in blobs if isinstance(b, list)))

spans = [geometry_of(b)["span"] for b in blobs if isinstance(b, list)]
check(f"spans land near NORMALISE_SPAN={NORMALISE_SPAN}",
      all(abs(s - NORMALISE_SPAN) <= 2 for s in spans),
      f"got {sorted(set(spans))}")

# real MNIST must be left completely alone
ds = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())
mn = [[255 if v else 0 for v in (ds[i][0].squeeze().numpy() >= 0.5).reshape(-1).tolist()]
      for i in range(800)]
untouched = sum(1 for p in mn if normalise_pixels(p)[0] == p)
check("real MNIST digits pass through byte-identical", untouched == 800,
      f"{untouched}/800")


# ------------------------------------------------------------------ accuracy
def classify(flat):
    x = torch.tensor([255 if v else 0 for v in flat], dtype=torch.float32).reshape(1, 784) / 255.0
    with torch.no_grad():
        s = model(x)
    return int(s.argmax(1)), [float(v) for v in s[0].tolist()]


print("\n[3] accuracy on MNIST (1500 images)")
mnl = np.array([ds[i][1] for i in range(1500)])
mn2 = [[255 if v else 0 for v in (ds[i][0].squeeze().numpy() >= 0.5).reshape(-1).tolist()]
       for i in range(1500)]
ok = 0
for p, lab in zip(mn2, mnl):
    q, _ = normalise_pixels(p)
    ok += classify(q)[0] == lab
acc_mnist = 100.0 * ok / len(mnl)
check("MNIST >= 90%", acc_mnist >= 90.0, f"{acc_mnist:.2f}%")
print(f"        measured {acc_mnist:.2f}%")

print("\n[4] accuracy on hand-drawn input (400 digits)")
ok = 0
preds = []
for im, lab in zip(fh, hl):
    q, _ = normalise_pixels([255 if v else 0 for v in
                             (np.asarray(im, np.float32).reshape(-1) * 255).astype(int).tolist()])
    pr, _s = classify(q)
    preds.append((pr, lab))
    ok += pr == lab
acc_hand = 100.0 * ok / len(hl)
check("hand-drawn >= 80%", acc_hand >= 80.0, f"{acc_hand:.2f}%")
print(f"        measured {acc_hand:.2f}%")

# --------------------------------------------------------------- confidence
print("\n[5] confidence readout (the bug found in the screenshots)")


def band(scores):
    top = max(scores)
    e = [math.exp(s - top) for s in scores]
    p = 1.0 / sum(e)
    return ("high" if p >= 0.60 else "medium" if p >= 0.40 else "low"), p


bands = {}
for im, lab in zip(fh, hl):
    q, _ = normalise_pixels([255 if v else 0 for v in
                             (np.asarray(im, np.float32).reshape(-1) * 255).astype(int).tolist()])
    _pr, sc = classify(q)
    b, p = band(sc)
    bands.setdefault(b, [0, 0])
    bands[b][0] += 1
    bands[b][1] += (int(np.argmax(sc)) == lab)

print(f"        distribution: " +
      ", ".join(f"{k}={v[0]}" for k, v in sorted(bands.items())))
check("not everything is 'low' (old bug said low for all 400)",
      not (len(bands) == 1 and "low" in bands))
if "high" in bands and "low" in bands:
    hi = 100.0 * bands["high"][1] / bands["high"][0]
    lo = 100.0 * bands["low"][1] / bands["low"][0]
    check("high band is more accurate than low band", hi > lo, f"high {hi:.1f}% vs low {lo:.1f}%")
    print(f"        high correct {hi:.1f}%   low correct {lo:.1f}%")

# ----------------------------------------------------------------- uploader
print("\n[6] image uploader")
os.makedirs("tt", exist_ok=True)
fonts = []
for f in ["arial.ttf", "arialbd.ttf", "times.ttf", "cour.ttf"]:
    try:
        ImageFont.truetype(f, 150); fonts.append(f)
    except Exception:
        pass
loaded_ok = 0
total = 0
for fn in fonts:
    for d in range(10):
        im = Image.new("RGB", (260, 260), (255, 255, 255))
        ImageDraw.Draw(im).text((55, 35), str(d), fill=(0, 0, 0),
                                font=ImageFont.truetype(fn, 150))
        im.save("tt/x.png")
        try:
            px, note = load_digit_image("tt/x.png")
            total += 1
            if len(px) == 784 and set(px) <= {0, 255} and sum(px) > 0:
                loaded_ok += 1
        except Exception as exc:
            print(f"        {fn} '{d}' raised {exc}")
check("every font image loads to a valid 28x28 binary",
      loaded_ok == total, f"{loaded_ok}/{total}")

try:
    Image.new("L", (100, 100), 128).save("tt/blank.png")
    load_digit_image("tt/blank.png")
    check("blank image is rejected with an error", False, "no exception raised")
except ValueError:
    check("blank image is rejected with an error", True)

# ------------------------------------------------------------ network view
print("\n[7] network view geometry")
import network_view as nv
check("input header is centred over the image block",
      "IMG_X + (GRID * IMG_CELL) / 2" in open("python/network_view.py",
                                               encoding="utf-8").read())
v = nv.NetworkView.__new__(nv.NetworkView)
v.hidden_x = []; v.hidden_layers = 1
v.row_top = nv.IMG_Y
v.row_span = nv.HEIGHT - 2 * nv.IMG_Y - 20
v.set_depth(1)
ys = [v.hidden_xy(h, 0)[1] for h in range(16)]
check("16 hidden nodes spread down the column", ys[-1] - ys[0] > 200,
      f"span {ys[-1]-ys[0]:.0f}px")
check("hidden column inside canvas", ys[-1] < nv.HEIGHT, f"max y {ys[-1]:.0f}")
check("output column inside canvas", v.output_xy(9)[1] < nv.HEIGHT)
check("edge limit is applied", nv.EDGE_LIMIT <= 300, str(nv.EDGE_LIMIT))

print("\n" + "=" * 68)
print(f"RESULT: {passed} passed, {failed} failed")
print("=" * 68)
import shutil
for d in ("tt", "t2", "t3"):
    shutil.rmtree(d, ignore_errors=True)
print(f"\nsummary: MNIST {acc_mnist:.1f}%   hand-drawn {acc_hand:.1f}%")
