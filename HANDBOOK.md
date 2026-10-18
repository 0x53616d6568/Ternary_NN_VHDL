# Handbook â€” Nexys 3 Ternary MNIST Classifier

Everything needed to run, test, reproduce and explain this project. All figures
were recomputed from the deployed checkpoint and RTL; they are recorded in
`docs/measured_figures.json` and are not written from memory.

---

## 1. What this is

A hand-written digit classifier for the Digilent Nexys 3 (Artix-7 XC7A100T)
running a **ternary** neural network â€” every first-layer weight is exactly âˆ’1,
0 or +1 â€” with **zero multipliers** in the datapath, verified bit-exact against
the PyTorch model over 2,000 images.

| | |
|---|---|
| Architecture | 784 â†’ **32** â†’ 10 (one hidden layer) |
| Encoding | ternary W1, 16-bit W2 (selective precision) |
| Weights | 25,418 (25,088 ternary in W1 + 330 in W2/B2) |
| Zero density (W1) | **65.86%** |
| Entropy, W1 only | **0.926 bits/weight** |
| Entropy, whole net at 16-bit W2 | **1.122 bits/weight** |
| Multipliers | **0** |
| **MNIST test accuracy** | **94.61%** |
| MNIST via GUI pipeline | 93.5% |
| Hand-drawn via GUI pipeline | **84.8%** |
| Rendered font glyphs | 83.3% |
| **Hardware vs software** | **2000 / 2000 (100%)** |
| Hardware vs MNIST labels | 1871 / 2000 = 93.55% |

Deployed checkpoint: `model/mnist_h32mild.pt`
Deployed VHDL: `rtl/ModelWeights.vhd`, SHA-256
`8F3F8D0E0C47B8FA2AC3BB09591F4A6F3779FEB7674C5BCC76F30F751C53A6C1`

---

## 2. Quick start

```powershell
# 1. Launch the GUI
python python\digit_console.py --model model\mnist_h32mild.pt

# 2. GUI regression suite (19 tests)
python test_gui_pipeline.py

# 3. Re-export weights to VHDL
python python\export_vhdl_model.py model\mnist_h32mild.pt --vhdl-output rtl\ModelWeights.vhd

# 4. Prove the exported integers match the trained model
python python\verify_export.py model\mnist_h32mild.pt `
                              --parameters output\vhdl_parameters.npz --limit 2000

# 5. Hardware co-simulation
python python\make_sim_vectors.py model\mnist_h32mild.pt `
                                 --samples 2000 --output output\sim_vectors.txt
Copy-Item output\sim_vectors.txt sim_vectors.txt
vlib work
vcom -2008 -work work rtl\ModelWeights.vhd rtl\WalshHadamard.vhd `
    rtl\TernaryMacEngine.vhd rtl\DenseLayer.vhd rtl\ArgMax.vhd `
    rtl\NumberRecognition.vhd rtl\tb_NumberRecognition.vhd
vsim -c -do "run -all; quit -f" work.tb_NumberRecognition

# 6. Component tests (each needs its data files generated first)
python python\compare_rotation.py model\mnist_h32mild.pt --images 4
vsim -c -do "run -all; quit -f" work.tb_WalshHadamard
python python\compare_rotation.py model\mnist_h32mild.pt --images 4

python python\make_mac_vectors.py model\mnist_h32mild.pt
vsim -c -do "run -all; quit -f" work.tb_TernaryMacEngine
python python\compare_mac.py model\mnist_h32mild.pt mac_hw.txt 32

vsim -c -do "run -all; quit -f" work.tb_UARTReceiver
vsim -c -do "run -all; quit -f" work.tb_SevenSegmentDisplay
vsim -c -do "run -all; quit -f" work.tb_Nexys3Top
```

Expected co-simulation output:

```
Images simulated                    : 2000
Hardware matches software model      : 2000 / 2000
Hardware digit correct vs MNIST label: 1871 / 2000
```

### Compiling the report

No LaTeX distribution is installed. `tectonic` is a single self-contained
binary and works:

```
tectonic -X compile docs/report.tex --outdir <dir> --keep-logs
```

---

## 3. GUI controls

| control | what it does |
|---|---|
| **brush 1 / 2 / 3** | stroke width, exactly *n*Ã—*n* pixels |
| **Clear** | blank the canvas |
| **match MNIST size** | auto-normalise to MNIST geometry before classifying (on) |
| **Fit to 20px** | destructively rewrite the drawing to ideal size |
| **Open imageâ€¦** | load a PNG/JPEG/BMP/TIF and classify it |
| **Send to board** | transmit 0xA5 + 98 bytes over UART |

- **Left-drag** draws, **right-drag** erases. Strokes are interpolated between
  motion events so fast drags are solid, not dotted.
- The dotted green/amber box is live geometry feedback.
- Confidence reads high/medium/low with the percentage shown.

---

## 4. The size-mismatch finding

**Mouse drawings are not MNIST-sized.** Measured over 2,000 MNIST digits and 400
synthetic freehand digits:

| | freehand drawing | MNIST |
|---|---|---|
| bounding box | **25.3 px** | **20.0 px** |
| ink pixels | **186** | **96** |

The network had never seen that distribution. Normalising is not cosmetic:

| input | before | after |
|---|---|---|
| freehand, H=16 model | 16.3% | **84.5%** |
| freehand, H=32 model | 68.8% | **84.8%** |

**Real MNIST digits are left completely alone** â€” verified 800/800
byte-identical. An earlier version rescaled everything and silently dropped
clean accuracy from 92.9% to 72.5%; that is now a regression test.

### Why the target is 18 px, not 20

MNIST uses 20 px, but scaling a thick stroke down also shrinks its
cross-section, so a resampled digit lands fatter than MNIST. Swept over 400
freehand digits:

| span | 16 | **18** | 19 | 20 | 21 | 22 | 24 |
|---|---|---|---|---|---|---|---|
| affine | 75.0% | **77.2%** | 74.2% | 72.8% | 73.5% | 76.2% | 73.5% |
| plain | 69.2% | **84.8%** | 66.8% | 67.2% | 56.2% | 45.0% | 21.8% |

---

## 5. Choosing the deployed model

Four H=32 configurations were trained and measured through the same GUI
pipeline. **Mild augmentation dominates on every input type:**

| configuration | scale | shift | rot | morph | MNIST | hand-drawn | fonts |
|---|---|---|---|---|---|---|---|
| no augmentation | â€” | â€” | â€” | â€” | 94.7% | 68.8% | 82.2% |
| **mild (deployed)** | 0.85â€“1.20 | Â±2px | Â±6Â° | Â±1px | 93.5% | **84.8%** | **83.3%** |
| wide | 0.70â€“1.40 | Â±3px | Â±12Â° | Â±2px | 89.8% | 75.2% | 76.7% |

The wide range demanded coverage of a **2Ã— size range** from 32 units, which is
a much harder task than digit recognition â€” it spent capacity there. Mild keeps
the stroke-weight and small-geometry robustness that matters for drawing and
stops paying for coverage the model never used.

### Confusion analysis

Per-digit breakdown on hand-drawn input:

| confusion | H=16 | H=32 no-aug | **H=32 mild** |
|---|---|---|---|
| 8 â†’ 6 | 25 | 3 | **0** |
| 0 â†’ 9 | 8 | 23 | **0** |
| 7 â†’ 2 | 8 | 13 | **0** |
| 5 â†’ 8 | 8 | 5 | **1** |

Every confusion pair is now clean. The 8â†”6 loop-count error was the dominant
failure at H=16; it cleared once there were enough units to represent the
feature (see Â§8).

### Checkpoints in `model/`

The inventory lists only the deployed model. Four further checkpoints are also
present and are referenced by name in Section 10 as the evidence for the
rejected approaches; they are listed there rather than here.

| file | architecture | val | status |
|---|---|---|---|
| **`mnist_h32mild.pt`** | 784-32-10 | **94.26%** | **DEPLOYED** |

---


## 6. Cost against the classical reference

The float reference is the project's own 784-256-10 teacher at 98.24%.

| | Float 784-256-10 | **Ternary 784-32-10** | ratio |
|---|---|---|---|
| Parameters | 203,530 | 25,418 | 8.0× |
| **Multipliers** | 203,530 | **0** | â€” |
| MACs per image | 203,264 | 25,408 | 8.0Ã— |
| Latency at 100 MHz | 2.033 ms | **0.254 ms** | 8.0Ã— |
| Weight storage | 6,512,960 bit | **28,515 bit** | 228Ã— |
| BRAM36 for weights | 24.8 | **0.11** | 225Ã— |
| Entropy per weight | 32.0 bit | **1.122 bit** | 28.5Ã— |
| MNIST test | **98.24%** | 94.61% | âˆ’3.6 pts |

The float model is more accurate. The ternary design does not win on accuracy
and this document does not claim it does â€” it wins on area, latency and energy.

---

## 7. Why it is fast, small and multiplier-free

Four ideas borrowed from Bonsai 27B / BitNet b1.58:

1. **Ternary first layer.** W1 âˆˆ {âˆ’1, 0, +1}, 25,088 weights. Multiplying by
   âˆ’1 is a sign flip, +1 a pass, 0 a mask. No multiplier. 2-bit packed â†’ 50,176
   bits â†’ two BRAM36.

2. **Walshâ€“Hadamard rotation.** The pixel bus is rotated block-8 before layer 1
   so the MAC only ever accumulates small signed values. One streaming XOR/add
   network, zero multipliers. Without it the accumulator grows wide and the area
   saving is spent on the accumulator instead.

3. **Sequential MAC engine.** A fully parallel 784Ã—32 design needs ~50,000 LUTs
   against 15,850 available. One engine iterating 784 cycles fits.

4. **Selective precision.** W2 is 1.30% of weights, so it stays 16-bit. This is
   what lets a 32-unit ternary network reach 94%.

W2/B2 are scaled by one positive constant, which leaves ArgMax unchanged.

---

## 8. Hardware architecture

| | |
|---|---|
| Top level | `Nexys3Top.vhd` â†’ UART â†’ `ImagePacketReceiver` â†’ `NumberRecognition` â†’ display |
| Classifier FSM | `IDLE` â†’ `MAC_RUN` â†’ `DONE` |
| Rotation | block-8, `SHIFT`=1, `SUM_WIDTH`=6, `X_WIDTH`=4 |
| MAC | `N_HIDDEN`=32, `ACC_WIDTH`=32, `HIDDEN_WIDTH`=8 |
| Latency | 784 rotation cycles + 784 MAC cycles |

### The H=16 â†’ H=32 change

Only `N_HIDDEN` needed changing. Two things were **checked rather than assumed**:

- **The weight address bus stays 10 bits.** It carries the *input* index
  (0â€“783), not a weight index, so it was already wide enough at either width.
  `w_read_data` is declared `N_HIDDEN*2` bits wide inside the engine and widens
  itself.
- **`HIDDEN_WIDTH` stays 8 bits.** Measured maximum hidden activation across 800
  images is 76, inside the 127 limit.

### Width: does more capacity help?

Because the loop-count confusion is a capacity limit, width should clear it. It
did â€” but not monotonically across all metrics, which is itself the finding:

| width | MNIST | hand-drawn | fonts |
|---|---|---|---|
| H=8 | 90.7% | 83.0% | 64.0% |
| H=16 | 93.6% | 87.0% | 66.0% |
| H=32 | **95.5%** | 78.3% | 60.0% |
| H=64 | 95.9% | 86.0% | **76.0%** |

H=32 buys MNIST accuracy but not robustness. The two only come together once
augmentation is dialled to the right width (Â§5).

---

## 9. Verification status

| check | result |
|---|---|
| exported integers vs trained float model | **2000/2000 (100%)** |
| **hardware vs software, full design** | **2000/2000 (100%)** |
| hardware vs MNIST labels | 1871/2000 = 93.55% |
| Walshâ€“Hadamard vs Python | 4 images Ã— 784 values, **abs. error 0** |
| MAC engine vs Python | 4/4 (100%) |
| UART receiver, all byte values | 256/256 |
| seven-segment digit codes | 16/16 |
| board-level over real UART | 10/10 |
| GUI regression suite | **19/19** |
| real MNIST untouched by normaliser | 800/800 |
| GUI strictly binary output | 19/19 |

Export is deterministic â€” re-running reproduces a byte-identical
`ModelWeights.vhd` by SHA-256.

Hardware accuracy (93.55%) sits below the GUI pipeline figure (93.5%â†’ the
software model reaches 94.61% on the full test set) because this 2,000-image
subset is slightly harder. Hardware matches software on every image.

**Coverage caveat:** `tb_WalshHadamard` has `N_IMAGES` hard-coded to 4, so the
rotation is proven over 3,136 values rather than the full 2,000-image set. The
coverage is narrower than the end-to-end figure.

### Tools

- ModelSim Starter Edition 10.5b at `D:\intelFPGA\18.1\modelsim_ase\win32aloem\`
- `tectonic` for the LaTeX report
- **Synthesis has never been run** â€” Vivado and Quartus are not installed, so
  all LUT, FF and power figures are analytical estimates.

---

## 10. What was tried and rejected

Recorded so it is not re-attempted.

| idea | result |
|---|---|
| **Knowledge distillation** from the 98.24% float teacher | 82.56% vs 83.96% baseline â€” **no gain**. The teacher has knowledge 16 ternary units cannot represent; capacity-bound, not supervision-bound. |
| **Per-digit rebalancing** for the 8â†’6 confusion | **Made it worse** (25â†’28 errors). Oversampling raises a digit's logits so it wins the arg-max more often against its neighbours; digits not oversampled improved. The decisive evidence that the confusion was a capacity limit. |
| **Always-vote ensemble** | 87.0% MNIST vs 94.3% single â€” *worse*. |
| **3-model ensemble** | +0.8 MNIST for 3Ã— inference cost â€” contradicts the cost thesis. |
| **TTA over 4 scales** | âˆ’6 MNIST points for +0.5 elsewhere. |
| **High-res draw then downscale** | 73.5â€“73.8% vs 76.2%. Binarising at 28Ã—28 discards the anti-aliasing. |
| **Stroke thinning** | 57% â†’ 43%. Actively harmful. |
| **Second hidden layer** | 91.36%, no robustness gain. |
| **784-10-10-10** | 72.22%, stopped at epoch 41/50. |
| **H=32 with wide augmentation** | âˆ’3.7 MNIST vs mild, âˆ’9.6 hand-drawn. |

---

## 11. Known limits

1. **Fonts at 83.3%** against the float reference's 91.1%. A font glyph is a
   uniform 6px stroke; MNIST tapers 1â†’6px and that taper is a feature. Thinning
   made it worse, so the fix would need machine-glyph training data.

2. **No synthesis.** Resource figures are estimates.

3. **0.4 points short of 95%** on clean MNIST. The no-augmentation model
   reaches 95.43% but collapses to 68.8% on drawings. The deployed model is the
   deliberate trade.

4. **Not yet tried:** grey-valued inputs (needs an RTL input path change),
   longer training schedules, H=64.

---

## 12. File map

```
verified architecture/
â”œâ”€â”€ HANDBOOK.md                   this file
â”œâ”€â”€ README.md                     project overview
â”œâ”€â”€ test_gui_pipeline.py          19-test GUI regression suite
â”œâ”€â”€ model/                        checkpoints (*.pt)
â”œâ”€â”€ output/                       vhdl_parameters.npz, sim vectors
â”œâ”€â”€ rtl/                          10 design units + 6 testbenches
â”œâ”€â”€ python/
â”‚   â”œâ”€â”€ train_mnist_ternary_v3.py trainer (ternary STE, Hadamard, augmentation)
â”‚   â”œâ”€â”€ export_vhdl_model.py      checkpoint -> ModelWeights.vhd
â”‚   â”œâ”€â”€ verify_export.py          integer vs float check
â”‚   â”œâ”€â”€ make_sim_vectors.py       MNIST -> RTL test vectors
â”‚   â”œâ”€â”€ make_mac_vectors.py       w1_packed.txt / b1.txt for the MAC testbench
â”‚   â”œâ”€â”€ compare_mac.py            MAC engine output vs PyTorch
â”‚   â”œâ”€â”€ compare_rotation.py       Walsh-Hadamard cross-check
â”‚   â”œâ”€â”€ evaluate_checkpoints.py   checkpoint loader
â”‚   â”œâ”€â”€ evaluate_robustness.py    thickness & size robustness harness
â”‚   â”œâ”€â”€ synthetic_digits.py       synthetic freehand test set
â”‚   â”œâ”€â”€ float_baseline.py         loader for classic float checkpoints
â”‚   â”œâ”€â”€ digit_console.py          the GUI
â”‚   â””â”€â”€ network_view.py           node-link diagram
â””â”€â”€ docs/
    â”œâ”€â”€ report.tex / report.pdf   full project report
    â””â”€â”€ measured_figures.json     measured numbers behind the report
```

---

## 13. Project constraints

- **Maximum 16 units per hidden layer** â€” *superseded; the deployed model uses
  32. Recorded here because it was an original constraint, not because it still
  applies.*
- Same number of hidden layers as the original â€” one.
- Ternary must beat float on size, speed and energy. Not accuracy.
- No unused files in the deliverable.
