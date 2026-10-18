-- Full-design ModelSim run: compile the ternary classifier and co-simulate it
-- against the PyTorch model on real MNIST images.
--
-- Run from the project root:
--   vsim -c -do "do scripts/modelsim_run.do"
--
-- The vector file is produced by:
--   python python/make_sim_vectors.py output/exact2_h16_b8.pt --samples 200
--
-- Compilation order matters: the weight package is needed by the engine and the
-- top level, and the datapath entities are needed by NumberRecognition.

transcript on

if {[file exists work]} {
    vdel -lib work -all
}
vlib work
vmap work work

vcom -2008 rtl/ModelWeights.vhd
vcom -2008 rtl/WalshHadamard.vhd
vcom -2008 rtl/TernaryMacEngine.vhd
vcom -2008 rtl/DenseLayer.vhd
vcom -2008 rtl/ArgMax.vhd
vcom -2008 rtl/NumberRecognition.vhd
vcom -2008 rtl/UARTReceiver.vhd
vcom -2008 rtl/ImagePacketReceiver.vhd
vcom -2008 rtl/SevenSegmentDisplay.vhd
vcom -2008 rtl/Nexys3Top.vhd
vcom -2008 rtl/tb_NumberRecognition.vhd
vcom -2008 rtl/tb_Nexys3Top.vhd

# --- Classifier only: digit output against the PyTorch model -------------------
vsim -voptargs=+acc work.tb_NumberRecognition
add wave -divider {Testbench control and image input}
add wave -radix unsigned sim:/tb_NumberRecognition/clk
add wave -radix unsigned sim:/tb_NumberRecognition/pixel_we
add wave -radix unsigned sim:/tb_NumberRecognition/pixel_addr
add wave -radix unsigned sim:/tb_NumberRecognition/pixel_data
add wave -divider {Predicted digit}
add wave -radix unsigned sim:/tb_NumberRecognition/digit
run -all

# --- Full board: serial in through to the seven-segment pins -----------------
# Drives real UART framing on RsRx and decodes the multiplexed segment outputs,
# which is the only path an observer of the board can actually confirm.
vsim -voptargs=+acc work.tb_Nexys3Top
add wave -divider {Serial input}
add wave -radix unsigned sim:/tb_Nexys3Top/rsrx
add wave -divider {Seven segment outputs}
add wave -radix binary sim:/tb_Nexys3Top/seg
add wave -radix binary sim:/tb_Nexys3Top/an
run -all