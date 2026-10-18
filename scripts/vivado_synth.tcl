# Synthesis script for the verified ternary classifier.
#
# No synthesis tool was available during development, so this has not been run.
# It is provided so the design can be taken straight to Vivado.
#
# Usage (from the folder root):
#   vivado -mode batch -source scripts/vivado_synth.tcl
#
# Then inspect:
#   synthesis_utilization.rpt   -- LUT / FF / BRAM / DSP counts
#   synthesis_timing.rpt        -- critical path and slack against 100 MHz

set part_name xc7a100tcsg324-1
set top_name  NumberRecognition

read_vhdl -vhdl2008 rtl/ModelWeights.vhd
read_vhdl -vhdl2008 rtl/WalshHadamard.vhd
read_vhdl -vhdl2008 rtl/TernaryMacEngine.vhd
read_vhdl -vhdl2008 rtl/DenseLayer.vhd
read_vhdl -vhdl2008 rtl/ArgMax.vhd
read_vhdl -vhdl2008 rtl/NumberRecognition.vhd

synth_design -top $top_name -part $part_name

# The two questions left open during development:
#  1. does the second layer infer DSP48 slices? Only 90 exist on this device
#     and the design contains 160 multiplies, so LUT implementation is expected
#     but unconfirmed.
#  2. does the combinational second layer and the indexed package read meet the
#     100 MHz target?
report_utilization -hierarchical -file synthesis_utilization.rpt
report_timing_summary -max_paths 10 -file synthesis_timing.rpt
report_dsp_utilization -file synthesis_dsp.rpt

puts "Synthesis complete. See synthesis_utilization.rpt and synthesis_timing.rpt."