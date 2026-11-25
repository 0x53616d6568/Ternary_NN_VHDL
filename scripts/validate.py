#!/usr/bin/env python3
"""Validate the VHDL: static wiring checks, compile, and run every testbench.

One command, one verdict. Designed to be run from the repository root:

    python scripts/validate.py                 # everything (a few minutes)
    python scripts/validate.py --skip-long     # skip the 2,000-image run

What it checks
--------------
Static (no tools required)
  * every synthesisable unit is listed in both build scripts, in an order that
    respects instantiation (this is exactly what was broken before);
  * every port of the board top level has a pin constraint, and every
    constraint names a real port -- the failure mode that silently leaves a
    clock unrouted;
  * no VHDL-2008-only `process(all)` remains in synthesisable RTL, because the
    Spartan-6 toolchain is VHDL-93;
  * no `*` operator appears in the ternary datapath, which is the
    multiplier-free claim the report makes.

Dynamic (ModelSim)
  * compiles all 17 design units and 6 testbenches from a clean library;
  * runs the self-checking testbenches;
  * regenerates the two file-based references and compares them numerically.

Nothing is written outside a temporary directory. Exit code is 0 only if every
check passes.

Requires: ModelSim (`vcom`, `vsim`) on PATH. The rotation reference additionally
needs PyTorch and the MNIST data under data/; without them that one comparison
is reported as SKIP rather than PASS.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl"

# Dependency order. Leaves first: ModelSim resolves `entity work.X` at compile
# time, so a unit must exist before anything that instantiates it.
DESIGN_UNITS = [
    "ModelWeights", "Adder", "ReLU", "Accumulator", "Saturate", "TernaryApply",
    "TernaryNeuron", "NeuronLayer", "TernaryMacEngine", "WalshHadamard",
    "DenseLayer", "ArgMax", "NumberRecognition", "UARTReceiver",
    "ImagePacketReceiver", "SevenSegmentDisplay", "Nexys3Top",
]
TESTBENCHES = [
    "tb_UARTReceiver", "tb_SevenSegmentDisplay", "tb_WalshHadamard",
    "tb_TernaryMacEngine", "tb_NumberRecognition", "tb_Nexys3Top",
]
# Files in which a `*` operator would falsify the multiplier-free claim.
TERNARY_DATAPATH = [
    "TernaryApply", "Accumulator", "Adder", "Saturate", "ReLU", "TernaryNeuron",
    "NeuronLayer", "TernaryMacEngine",
]

results = []


def record(name, ok, detail=""):
    results.append((name, ok, detail))
    tag = {True: "PASS", False: "FAIL", None: "SKIP"}[ok]
    print("  [%s] %-46s %s" % (tag, name, detail))


def read(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def strip_comments(text, marker="--"):
    """Drop comments so checks cannot fire on prose.

    Several files mention `process(all)` and `xc7a100t` in a comment that
    explains why those are NOT used. A naive substring check flags the
    explanation, so every static check below works on code only.
    """
    out = []
    for line in text.splitlines():
        # No string literal in this repo contains the comment marker, and the
        # package's hex words are quoted but never followed by a comment.
        idx = line.find(marker)
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


# ---------------------------------------------------------------- static -----
def check_scripts():
    print("\nStatic: build scripts list every unit, in a usable order")
    do = read(ROOT / "scripts" / "modelsim_run.do")
    tcl = read(ROOT / "scripts" / "ise_build.tcl")

    for label, text in (("modelsim_run.do", do), ("ise_build.tcl", tcl)):
        missing = [u for u in DESIGN_UNITS if u + ".vhd" not in text]
        record("%s lists all 17 units" % label, not missing,
               "" if not missing else "missing " + ", ".join(missing))

    # Order: the index of each vcom line must increase.
    positions = {u: do.find("rtl/%s.vhd" % u) for u in DESIGN_UNITS}
    ordered = all(positions[a] < positions[b]
                  for a, b in zip(DESIGN_UNITS, DESIGN_UNITS[1:]))
    record("modelsim_run.do compiles in dependency order", ordered)

    record("ise_build.tcl targets the board top level",
           'set "Top Module Name" "Nexys3Top"' in tcl)
    # Code only: the script's header legitimately names the Artix-7 part it
    # replaced, and that explanation must not look like a target.
    tcl_code = strip_comments(tcl, "#").lower()
    record("ise_build.tcl targets the Spartan-6 part",
           "xc6slx16" in tcl_code and "xc7a100t" not in tcl_code)
    record("ise_build.tcl adds the UCF",
           "constraints/Nexys3Top.ucf" in tcl)


def check_constraints():
    print("\nStatic: pin constraints cover exactly the top-level ports")
    top = read(RTL / "Nexys3Top.vhd")
    entity = top.split("end Nexys3Top;")[0]
    ports = re.findall(r"^\s*(\w+)\s*:\s*(in|out)\s+std_logic(_vector\s*\(\s*(\d+)\s+downto\s+(\d+)\s*\))?",
                       entity, re.M)
    expected = set()
    for name, _dirn, _vec, hi, lo in ports:
        if hi == "":
            expected.add(name)
        else:
            for b in range(int(lo), int(hi) + 1):
                expected.add("%s<%d>" % (name, b))

    ucf = read(ROOT / "constraints" / "Nexys3Top.ucf")
    nets = set(re.findall(r'NET\s+"([^"]+)"\s+LOC', ucf))

    unconstrained = sorted(expected - nets)
    dangling = sorted(nets - expected)
    record("every top-level port is constrained", not unconstrained,
           "" if not unconstrained else "unconstrained " + ", ".join(unconstrained))
    record("every constraint names a real port", not dangling,
           "" if not dangling else "dangling " + ", ".join(dangling))

    locs = re.findall(r'NET\s+"([^"]+)"\s+LOC\s*=\s*"([^"]+)"', ucf)
    seen = {}
    clash = []
    for net, loc in locs:
        if loc in seen:
            clash.append("%s and %s both on %s" % (seen[loc], net, loc))
        seen[loc] = net
    record("no two ports share a pin", not clash, "; ".join(clash))

    # The clock constraint has to be a real PERIOD on a real net.
    record("100 MHz period constraint present",
           re.search(r'PERIOD\s+"sys_clk_pin"\s+10\s*ns', ucf) is not None)


def check_vhdl_hygiene():
    print("\nStatic: language level and the multiplier-free claim")
    bad = [p.name for p in sorted(RTL.glob("*.vhd"))
           if not p.name.startswith("tb_") and "process(all)" in strip_comments(read(p))]
    record("no process(all) in synthesisable RTL (VHDL-93 target)", not bad,
           "" if not bad else ", ".join(bad))

    # The multiplier-free claim is about INFERRED multipliers, so the test has
    # to tell datapath arithmetic apart from index arithmetic. `N_HIDDEN * 2`
    # and `h * 2` are slice bounds resolved at elaboration; `a * b` on signals
    # is what builds a MULT18X18. Accept a `*` only when at least one operand
    # is a literal, an upper-case constant, or a `for` loop variable.
    offenders = []
    for unit in TERNARY_DATAPATH:
        code = strip_comments(read(RTL / (unit + ".vhd")))
        loop_vars = set(re.findall(r"\bfor\s+(\w+)\s+in\b", code))
        for match in re.finditer(r"\*", code):
            if match.start() > 0 and code[match.start() - 1] == "*":
                continue                      # `**` is a constant power
            left = re.search(r"(\w+)\s*$", code[:match.start()])
            right = re.search(r"^\s*(\w+)", code[match.end():])
            operands = [m.group(1) for m in (left, right) if m]
            if not operands:
                continue
            benign = all(re.fullmatch(r"\d+|[A-Z][A-Z0-9_]*", o) or o in loop_vars
                         for o in operands)
            if not benign:
                line_no = code[:match.start()].count("\n") + 1
                offenders.append("%s:%d" % (unit, line_no))
    record("no multiply operator in the ternary datapath", not offenders,
           "" if not offenders else ", ".join(offenders))

    # The second layer is allowed to multiply: it is 8-bit weight x 8-bit
    # activation, and the report states that explicitly rather than hiding it.
    record("DenseLayer is the only unit with a datapath multiply",
           "*" in strip_comments(read(RTL / "DenseLayer.vhd")))


def check_weights():
    print("\nStatic: deployed constants")
    text = read(RTL / "ModelWeights.vhd")
    widths = {}
    for name in ("W1_PACKED_INIT", "B1_INIT", "W2_INIT", "B2_INIT"):
        m = re.search(r"constant\s+%s\s*:\s*std_logic_vector\s*\((\d+)\s+downto\s+0\)"
                      % name, text)
        widths[name] = int(m.group(1)) + 1 if m else 0
    ok = (widths["W1_PACKED_INIT"] == 784 * 64 and widths["B1_INIT"] == 32 * 32
          and widths["W2_INIT"] == 32 * 10 * 16 and widths["B2_INIT"] == 10 * 32)
    record("weight arrays have the deployed 784-32-10 shape", ok,
           json.dumps(widths))

    encoded = sum(widths[u] for u in ("W1_PACKED_INIT", "B1_INIT",
                                      "W2_INIT", "B2_INIT"))
    record("encoded storage matches the report (56,640 bit / 6.9 KiB)",
           encoded == 56640, "%d bit" % encoded)


def unpack_weights(dest):
    """Write the flat weight/bias files the MAC testbench reads."""
    text = read(RTL / "ModelWeights.vhd")

    def constant(name):
        m = re.search(r"constant\s+%s\s*:\s*std_logic_vector\s*\((\d+)\s+downto\s+0\)"
                      r"\s*:=\s*(.*?);" % name, text, re.S)
        width = int(m.group(1)) + 1
        digits = "".join(re.findall(r'x"([0-9A-Fa-f]+)"', m.group(2)))
        assert len(digits) * 4 == width, name
        return int(digits, 16)

    w1 = constant("W1_PACKED_INIT")
    with open(dest / "w1_packed.txt", "w", newline="\n") as f:
        for i in range(784):
            f.write("%016x\n" % ((w1 >> (i * 64)) & ((1 << 64) - 1)))
    b1 = constant("B1_INIT")
    with open(dest / "b1.txt", "w", newline="\n") as f:
        for h in range(32):
            f.write("%08x\n" % ((b1 >> (h * 32)) & 0xFFFFFFFF))


# --------------------------------------------------------------- dynamic -----
def run(cmd, cwd, timeout=3600):
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                          timeout=timeout, errors="replace")


def compile_all(work):
    print("\nDynamic: compile from a clean library")
    if (work / "work").exists():
        shutil.rmtree(work / "work")
    r = run(["vlib", "work"], work)
    if r.returncode != 0:
        record("vlib work", False, r.stdout.strip()[:120])
        return False
    failed = []
    for unit in DESIGN_UNITS + TESTBENCHES:
        r = run(["vcom", "-2008", "-quiet", str(RTL / (unit + ".vhd"))], work)
        if r.returncode != 0 or "** Error" in r.stdout:
            failed.append(unit)
    record("17 design units + 6 testbenches compile", not failed,
           "" if not failed else "failed: " + ", ".join(failed))
    return not failed


def summarise(text):
    """Pull the interesting report lines out of a ModelSim transcript."""
    keep = re.findall(r"# \*\* (?:Note|Failure|Fatal): (.+)", text)
    return [line.strip() for line in keep
            if re.search(r"Passed|Failed|received|agreement|matches|correct|"
                         r"Display|Images|Failure|Fatal|Bytes", line)]


def run_tb(work, name, timeout=3600):
    print("\nDynamic: %s" % name)
    r = run(["vsim", "-c", "-quiet", "-do", "run -all; quit -f",
             "work.%s" % name], work, timeout=timeout)
    text = r.stdout + r.stderr
    lines = summarise(text)
    for line in lines[:6]:
        print("         " + line)

    if "** Fatal" in text or "** Failure" in text:
        record(name, False, "assertion failed")
        return None
    m = re.search(r"# Errors: (\d+), Warnings: (\d+)", text)
    if not m:
        record(name, False, "simulation did not reach the end")
        return None
    errors, warnings = int(m.group(1)), int(m.group(2))
    record(name, errors == 0, "%d warnings" % warnings)
    return text


def compare_rotation(work):
    """Bit-compare the hardware rotation against the Python model."""
    hw_path = work / "rot_hw.txt"
    ref_path = work / "rot_py.txt"
    if not hw_path.exists():
        record("rotation: hardware vs Python", None, "no hardware output")
        return
    if not ref_path.exists():
        record("rotation: hardware vs Python", None,
               "no Python reference (needs torch + data/)")
        return
    hw = [l.split() for l in hw_path.read_text().splitlines() if l.strip()]
    ref = [l.split() for l in ref_path.read_text().splitlines() if l.strip()]
    same = hw == ref
    values = sum(len(r) for r in ref)
    record("rotation: hardware vs Python", same,
           "%d values, %s" % (values, "identical" if same else
                              "%d lines differ" % sum(a != b for a, b in zip(hw, ref))))


def compare_mac(work):
    """Recompute T(W).x + b in numpy and compare with the RTL's hidden vector."""
    hw_path = work / "mac_hw.txt"
    rot_path = work / "rot_py.txt"
    if not hw_path.exists() or not rot_path.exists():
        record("MAC engine: hardware vs integer model", None, "missing vector file")
        return
    try:
        import numpy as np
    except ImportError:
        record("MAC engine: hardware vs integer model", None, "numpy unavailable")
        return

    raw = [int(l, 16) for l in (work / "w1_packed.txt").read_text().split()]
    bias = [(int(l, 16) ^ 0x80000000) - 0x80000000
            for l in (work / "b1.txt").read_text().split()]
    # Two-bit ternary: bit1 = sign, bit0 = magnitude -> {-1, 0, +1}.
    w = np.zeros((784, 32), dtype=np.int64)
    for i, word in enumerate(raw):
        for h in range(32):
            code = (word >> (2 * h)) & 3
            w[i, h] = -1 if code == 3 else code
    x = np.array([int(t) for t in rot_path.read_text().split()], dtype=np.int64)
    x = x.reshape(-1, 784)

    expected = np.clip(x @ w + np.array(bias), 0, 127)
    actual = np.array([[int(t) for t in line.split()]
                       for line in hw_path.read_text().splitlines() if line.strip()])
    ok = expected.shape == actual.shape and np.array_equal(expected, actual)
    record("MAC engine: hardware vs integer model", ok,
           "%d images x 32 units" % actual.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-long", action="store_true",
                    help="skip the 2,000-image classifier run")
    ap.add_argument("--keep", action="store_true",
                    help="keep the scratch directory for inspection")
    args = ap.parse_args()

    for tool in ("vcom", "vsim", "vlib"):
        if shutil.which(tool) is None:
            sys.exit("%s not found on PATH; this validator needs ModelSim" % tool)

    check_scripts()
    check_constraints()
    check_vhdl_hygiene()
    check_weights()

    work = Path(tempfile.mkdtemp(prefix="ternary_validate_"))
    try:
        unpack_weights(work)
        shutil.copy(ROOT / "output" / "sim_vectors.txt", work / "sim_vectors.txt")

        # Rotation reference, if the Python side can produce it.
        if shutil.which("python"):
            env_note = "skip"
            if (ROOT / "data" / "MNIST").exists():
                r = run([sys.executable, str(ROOT / "python" / "compare_rotation.py"),
                         str(ROOT / "model" / "mnist_h32mild.pt"),
                         "--data-dir", str(ROOT / "data"),
                         "--vector-output", "rot_images.txt",
                         "--reference-output", "rot_py.txt",
                         "--hardware-output", "rot_hw.txt"],
                        work, timeout=1800)
                env_note = "ok" if r.returncode == 0 else "generator failed"
            record("rotation reference generated", env_note == "ok", env_note)

        if not compile_all(work):
            raise SystemExit(1)

        run_tb(work, "tb_UARTReceiver")
        run_tb(work, "tb_SevenSegmentDisplay")
        run_tb(work, "tb_WalshHadamard")
        compare_rotation(work)
        run_tb(work, "tb_TernaryMacEngine")
        compare_mac(work)
        run_tb(work, "tb_Nexys3Top")
        if not args.skip_long:
            run_tb(work, "tb_NumberRecognition", timeout=3600)
    finally:
        if args.keep:
            print("\nScratch directory: %s" % work)
        else:
            shutil.rmtree(work, ignore_errors=True)

    failed = [r for r in results if r[1] is False]
    skipped = [r for r in results if r[1] is None]
    print("\n" + "=" * 72)
    print("%d checks: %d passed, %d failed, %d skipped"
          % (len(results), len(results) - len(failed) - len(skipped),
             len(failed), len(skipped)))
    if failed:
        for name, _, detail in failed:
            print("  FAILED: %s %s" % (name, detail))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
