# Verified Architecture — Ternary MNIST Classifier for Nexys 3

A multiplier-free ternary neural network accelerator, verified bit-exact against
its PyTorch source model.

| | |
|---|---|
| **Architecture** | 784 → **32** → 10 (one hidden layer) |
| **Accuracy** | **94.61%** MNIST test; 1,871 / 2,000 in hardware simulation |
| **Hardware = software** | **2,000 / 2,000** (100%) |
| **Weight density** | **1.122 bits/weight** (0.926 in W1 alone) |
| **Zero density (W1)** | 65.86% |
| **Multipliers** | **0** |
| **Weights** | 25,418 — 25,088 ternary in W1, 330 in W2/B2 at 16 bits |
| **Latency** | ~1,568 cycles ≈ 15.7 µs @ 100 MHz (784 rotation + 784 MAC) |
| **Target** | Digilent Nexys 3 (Artix-7 XC7A100T) |

Deployed checkpoint `model/mnist_h32mild.pt`; `rtl/ModelWeights.vhd`
SHA-256 `8F3F8D0E0C47B8FA2AC3BB09591F4A6F3779FEB7674C5BCC76F30F751C53A6C1`.

## Cost against the classical reference

The float reference is the project's own 784-256-10 teacher at 98.24% ? a
conventional fully-connected float network, one hidden layer of 256 units.

| | Float 784-256-10 | Ternary 784-16-10 | **Ternary 784-32-10** | ratio vs float |
|---|---|---|---|---|
| Parameters | 203,530 | 12,730 | 25,418 | 8.0x |
| **Multipliers** | 203,530 | **0** | **0** | ? |
| MACs per image | 203,264 | 12,704 | 25,408 | 8.0x |
| Latency at 100 MHz | 2.033 ms | 0.127 ms | 0.254 ms | 8.0x |
| Weight storage | 6,512,960 bit | 15,121 bit | 28,515 bit | **228x** |
| BRAM36 for weights | 24.8 | 0.06 | 0.11 | **225x** |
| Entropy per weight | 32.0 bit | 1.189 bit | **1.122 bit** | 28.5x |
| MNIST test | **98.24%** | 94.19% | 94.61% | -3.6 pts |
| Hand-drawn | 80.5% | 74.8% | **84.8%** | **+4.3** |
| Rendered font glyphs | **91.1%** | 78.9% | 83.3% | -7.8 |

The float network is more accurate on clean MNIST. The ternary design does not
win on accuracy and this document does not claim it does ? it wins on area,
latency and energy, and on hand-drawn input it is the better classifier.

---

Full detail, including all measurements and the record of rejected approaches,
is in **[`HANDBOOK.md`](HANDBOOK.md)**.

---

## 1. Contents

### `rtl/` — hardware

| File | Role | Lines |
|---|---|---|
| `Nexys3Top.vhd` | Board top level | 92 |
| `NumberRecognition.vhd` | Classifier datapath and FSM | 246 |
| `WalshHadamard.vhd` | Streaming block-8 Hadamard rotation of the pixel bus | 160 |
| `TernaryMacEngine.vhd` | Sequential first layer, 32 lanes, 784 cycles | 124 |
| `DenseLayer.vhd` | Second layer 32 × 10, combinational | 74 |
| `ArgMax.vhd` | 10-way argmax over 32-bit scores | 41 |
| `ImagePacketReceiver.vhd` | `0xA5` + 98 packed bytes → 784 pixels | 86 |
| `UARTReceiver.vhd` | Serial receive at 115200 baud | 86 |
| `SevenSegmentDisplay.vhd` | 4-digit multiplexed display | 57 |
| `ModelWeights.vhd` | Generated weight package — **do not hand-edit** | 586 |

Testbenches: `tb_NumberRecognition` (end-to-end), `tb_WalshHadamard`,
`tb_TernaryMacEngine`, `tb_Nexys3Top` (board level), `tb_UARTReceiver`,
`tb_SevenSegmentDisplay`.

### `python/` — training, export, verification

| File | Role |
|---|---|
| `train_mnist_ternary_v3.py` | The trainer. Ternary STE, Walsh–Hadamard rotation, selective precision, closed-loop sparsity, `--integer-exact` mode, augmentation, per-digit rebalancing |
| `export_vhdl_model.py` | Checkpoint → `vhdl_parameters.npz` + `ModelWeights.vhd` |
| `verify_export.py` | Confirms exported integers reproduce the model |
| `make_sim_vectors.py` | Generates MNIST vectors for the end-to-end testbench |
| `make_mac_vectors.py` | Generates `w1_packed.txt` / `b1.txt` for the MAC testbench |
| `compare_mac.py` | Confirms the RTL MAC engine matches Python value-for-value |
| `compare_rotation.py` | Confirms the VHDL rotation matches Python value-for-value |
| `evaluate_checkpoints.py` | Accuracy and cost harness; provides `load_model()` |
| `float_baseline.py` | Loader for classic float checkpoints (784-256-10 reference) |
| `synthetic_digits.py` | Synthetic freehand test set |
| `evaluate_robustness.py` | Thickness and size robustness harness |
| `digit_console.py` | GUI: draw a digit, watch the network, transmit over UART |
| `network_view.py` | Node-link diagram renderer |

### Other

- `test_gui_pipeline.py` — 19-test GUI regression suite
- `model/mnist_h32mild.pt` — **the deployed checkpoint** (94.26% val)
- `scripts/vivado_synth.tcl` — unrun synthesis script
- `constraints/Nexys3Top.ucf` — pin constraints
- `docs/report.tex`, `docs/report.pdf` — full project report
- `docs/measured_figures.json` — measured numbers behind the report

---

## 2. Reproducing the verification

### Requirements

- ModelSim Intel FPGA Edition (tested with 10.5b)
- Python 3.11 with `torch`, `torchvision`, `numpy`, `Pillow`
- MNIST downloaded via torchvision into `./data`
- For the report: `tectonic` (single binary, no installer needed)

### Step 1 — regenerate the VHDL weight package

```bash
python python/export_vhdl_model.py model/mnist_h32mild.pt \
       --vhdl-output rtl/ModelWeights.vhd
```

This reproduces the committed file byte-for-byte (verified by SHA-256).

### Step 2 — confirm the export is lossless

```bash
python python/verify_export.py model/mnist_h32mild.pt \
       --parameters output/vhdl_parameters.npz --limit 2000
```

Expected:

```
Integer vs float match  : 2000/2000  (100.00%)
OK: the exported integers reproduce the trained model exactly.
```

### Step 3 — generate simulation vectors

```bash
python python/make_sim_vectors.py model/mnist_h32mild.pt \
       --samples 2000 --output output/sim_vectors.txt
cp output/sim_vectors.txt sim_vectors.txt
```

### Step 4 — compile and co-simulate

```bash
vlib work
vcom -2008 -work work rtl/ModelWeights.vhd rtl/WalshHadamard.vhd \
    rtl/TernaryMacEngine.vhd rtl/DenseLayer.vhd rtl/ArgMax.vhd \
    rtl/NumberRecognition.vhd rtl/tb_NumberRecognition.vhd
vsim -c -do "run -all; quit -f" work.tb_NumberRecognition
```

Expected summary:

```
Images simulated                    : 2000
Hardware matches software model      : 2000 / 2000
Hardware digit correct vs MNIST label: 1871 / 2000
```

### Step 5 — component tests

```bash
# Rotation
python python/compare_rotation.py model/mnist_h32mild.pt --images 4
vsim -c -do "run -all; quit -f" work.tb_WalshHadamard
python python/compare_rotation.py model/mnist_h32mild.pt --images 4

# MAC engine
python python/make_mac_vectors.py model/mnist_h32mild.pt
vcom -2008 -work work rtl/tb_TernaryMacEngine.vhd
vsim -c -do "run -all; quit -f" work.tb_TernaryMacEngine
python python/compare_mac.py model/mnist_h32mild.pt mac_hw.txt 32
```

`compare_rotation.py` must be run **before** the simulation to emit the
reference, and **after** to compare against the hardware output. Use
`--images 4` to match the testbench's hard-coded `N_IMAGES := 4`.

### Step 6 — board-level test (UART in, seven-segment out)

```bash
vcom -2008 -work work rtl/UARTReceiver.vhd rtl/ImagePacketReceiver.vhd \
    rtl/SevenSegmentDisplay.vhd rtl/Nexys3Top.vhd rtl/tb_Nexys3Top.vhd
vsim -c -do "run -all; quit -f" work.tb_Nexys3Top
```

Expected:

```
Images sent over UART              : 10
Display showed the expected digit : 10 / 10
```

Verifying the segment patterns rather than an internal digit signal is
deliberate: it is the only check that confirms the digit reaches the display.

### Step 7 — peripherals

```bash
vsim -c -do "run -all; quit -f" work.tb_UARTReceiver
vsim -c -do "run -all; quit -f" work.tb_SevenSegmentDisplay
```

Expected: `Passed 256 / Failed 0` and `Passed 16 / Failed 0`.

The UART test needs a concurrent monitor process: `data_valid` is a one-cycle
strobe raised partway through the stop bit, while the next byte is still being
transmitted, so a sender that checks for it afterwards always misses it.

### Step 8 — GUI regression suite

```bash
python test_gui_pipeline.py
```

19 tests covering the normalisation contract, the image uploader, the
confidence readout and the network-view geometry.

### Host tool: drawing, network inspection, and transmission

```bash
python python/digit_console.py --model model/mnist_h32mild.pt
```

* **Drawing canvas** — left paints, right erases, brush sizes 1–3.
* **Network activity** — the 32 hidden ReLU activations and the 10 output
  scores with the arg-max highlighted. Because the VHDL is verified
  bit-identical at every intermediate, these are the FPGA's actual values.
* **Serial link** — port detection with Rescan, explicit Connect, status line.
* **Open image…** — loads a PNG/JPEG/BMP/TIF, detects polarity automatically and
  classifies it.

Frame format: `0xA5` + 98 packed bytes, one bit per pixel, LSB first, row
order. If `--model` points at a missing checkpoint the tool still sends images;
only the preview is disabled.

---

## 3. Retraining from scratch

```bash
python python/train_mnist_ternary_v3.py \
    --epochs 60 --hidden-units 32 --integer-exact --walsh-block 8 \
    --layer2-precision full \
    --morph-aug 1 --aug-scale-min 0.85 --aug-scale-max 1.20 \
    --aug-shift 2.0 --aug-rotate 6 \
    --output model/mnist_h32mild.pt
```

`--integer-exact` is what keeps hardware and model identical: binary pixels, an
unnormalised rotation and a fixed right shift, so every value on the forward
path is an integer. Without it, export verification fails at roughly 99%
agreement.

The augmentation range matters more than expected. Measured on 400 hand-drawn
digits:

| augmentation | MNIST | hand-drawn |
|---|---|---|
| none | 94.7% | 68.8% |
| **mild (above)** | 93.5% | **84.8%** |
| wide (0.7–1.4, ±3px, ±12°, ±2px) | 89.8% | 75.2% |

---

## 4. Known limitations

**The design has not been synthesised.** No synthesis tool was available in the
development environment, so every resource figure in the report is a
hand-derived estimate rather than a tool result.

Two questions remain open and require a Vivado or Quartus run:

1. **DSP inference.** The second layer performs 320 genuine multiplications and
   the device provides 90 DSP slices. Because the hidden operand is narrow, LUT
   implementation is expected, but this is not confirmed. If DSPs are inferred,
   the remedy is to ternarise the second layer, at a direct accuracy cost.
2. **Timing closure.** The combinational second layer and the indexed package
   read form the longest combinational paths and have not been analysed against
   the 100 MHz target.

A ready-to-run script is in `scripts/vivado_synth.tcl`.

**Rotation test coverage.** `tb_WalshHadamard` has `N_IMAGES` hard-coded to 4,
so the rotation is proven over 3,136 values rather than the full 2,000-image
set. The coverage is narrower than the end-to-end figure.

---

## 5. Design note

The first layer is sequential rather than parallel for a concrete reason. A
fully parallel 784 × 32 array needs roughly 50,000 LUTs; the Artix-7 provides
15,850. Streaming one input per cycle needs about 300 LUTs and moves the weights
into two BRAM36 blocks (784 × 64 bits, two bits per weight).

The multiply is written as an explicit conditional add or subtract rather than
an infix `*`, so synthesis infers an adder and never reaches for a DSP. Two-bit
two's complement encodes {−1, 0, +1} directly with no decoder logic.

The Walsh–Hadamard rotation is not optional decoration. Without it, ternary
weights multiplying unbounded activations force a wide accumulator, and the
area saving is spent on the accumulator instead.
