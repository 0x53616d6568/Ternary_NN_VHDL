# Synthesis + implementation for the Nexys 3 build.
#
# WHY ISE AND NOT VIVADO: the Nexys 3 carries a Spartan-6 XC6SLX16, and the
# Spartan-6 family is not supported by any Vivado release. The previous script
# in this repository asked Vivado for `xc7a100tcsg324-1` -- an Artix-7 part on a
# different board -- and synthesised `NumberRecognition` rather than the board
# top level, so it could not have produced a working bitstream even on the
# wrong hardware. This script replaces it.
#
# Usage (from the folder root, with ISE 14.7 on PATH):
#   xtclsh scripts/ise_build.tcl
#
# NOT RUN IN THIS REPOSITORY: ISE is not installed in the environment where the
# design was verified, so the flow below is unexecuted. The file list, top
# level, part and constraint file ARE checked against the RTL by
# scripts/validate.py. Treat the first real run as a debugging session, not a
# formality -- see roadmap tasks 1-4 in the report.

set project_name "ternary_nexys3"

project new ${project_name}.xise
project set family  Spartan6
project set device  xc6slx16
project set package csg324
project set speed   -2
project set "Top-Level Source Type" "HDL"

# Every synthesisable file. Missing one of these is what broke the previous
# flow: XST reports "unknown module" and stops at elaboration.
foreach f {
    rtl/ModelWeights.vhd
    rtl/Adder.vhd
    rtl/ReLU.vhd
    rtl/Accumulator.vhd
    rtl/Saturate.vhd
    rtl/TernaryApply.vhd
    rtl/TernaryNeuron.vhd
    rtl/NeuronLayer.vhd
    rtl/TernaryMacEngine.vhd
    rtl/WalshHadamard.vhd
    rtl/DenseLayer.vhd
    rtl/ArgMax.vhd
    rtl/NumberRecognition.vhd
    rtl/UARTReceiver.vhd
    rtl/ImagePacketReceiver.vhd
    rtl/SevenSegmentDisplay.vhd
    rtl/Nexys3Top.vhd
} {
    xfile add $f
}

xfile add constraints/Nexys3Top.ucf

# The board top level. NumberRecognition has no pins and cannot be a top.
project set "Top Module Name" "Nexys3Top"

# The RTL is VHDL-93 compatible on purpose: `process(all)` was removed so XST
# accepts it. Only the testbenches need VHDL-2008 (std.env.stop).
project set "VHDL Source Analysis Standard" "VHDL-93"

# Synthesis and implementation. Watch for these in the log:
#   * "Xst: ... inferred RAM" for W1_PACKED_INIT -- 50,176 bits should become
#     block RAM, not 50,176 LUTs. If it reports LUT RAM or registers, task 3 of
#     the roadmap applies.
#   * DSP48 inference in DenseLayer: 320 multiplies live there. 32 DSP48A1
#     slices exist on the LX16; a LUT implementation is acceptable but must be
#     confirmed, not assumed.
#   * unconstrained I/O: the UCF covers clk100, RsRx, seg<7:0> and an<3:0>.
process run "Synthesize - XST"
process run "Translate"
process run "Map"
process run "Place & Route"
process run "Generate Post-Place & Route Static Timing"

puts "Done. Reports are under ${project_name}/."
puts "Check: map report (LUT/FF/BRAM/DSP), trce report (slack at 10 ns)."
