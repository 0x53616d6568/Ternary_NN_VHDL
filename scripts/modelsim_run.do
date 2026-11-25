-- Full-design ModelSim run: compile the ternary classifier and co-simulate it
-- against the PyTorch model on real MNIST images.
--
-- Run from the project root:
--   vsim -c -do "do scripts/modelsim_run.do"      (headless, all tests)
--   vsim -do "do scripts/modelsim_run.do"          (GUI, with waves)
--
-- The vector file is produced by:
--   python python/make_sim_vectors.py model/mnist_h32mild.pt --samples 2000 \
--          --output output/sim_vectors.txt
--
-- COMPILATION ORDER MATTERS and every design unit must be listed. ModelSim
-- resolves direct entity instantiation (`entity work.X`) at compile time, so a
-- leaf instantiated before it is compiled is a hard error. An earlier version
-- of this script listed only 10 of the 17 units and could not compile from a
-- clean library at all.

transcript on

if {[file exists work]} {
    vdel -lib work -all
}
vlib work
vmap work work

# --- Leaves first, then the units that instantiate them ----------------------
vcom -2008 rtl/ModelWeights.vhd
vcom -2008 rtl/Adder.vhd
vcom -2008 rtl/ReLU.vhd
vcom -2008 rtl/Accumulator.vhd
vcom -2008 rtl/Saturate.vhd
vcom -2008 rtl/TernaryApply.vhd
vcom -2008 rtl/TernaryNeuron.vhd
vcom -2008 rtl/NeuronLayer.vhd
vcom -2008 rtl/TernaryMacEngine.vhd
vcom -2008 rtl/WalshHadamard.vhd
vcom -2008 rtl/DenseLayer.vhd
vcom -2008 rtl/ArgMax.vhd
vcom -2008 rtl/NumberRecognition.vhd
vcom -2008 rtl/UARTReceiver.vhd
vcom -2008 rtl/ImagePacketReceiver.vhd
vcom -2008 rtl/SevenSegmentDisplay.vhd
vcom -2008 rtl/Nexys3Top.vhd

# --- Testbenches -------------------------------------------------------------
vcom -2008 rtl/tb_UARTReceiver.vhd
vcom -2008 rtl/tb_SevenSegmentDisplay.vhd
vcom -2008 rtl/tb_WalshHadamard.vhd
vcom -2008 rtl/tb_TernaryMacEngine.vhd
vcom -2008 rtl/tb_NumberRecognition.vhd
vcom -2008 rtl/tb_Nexys3Top.vhd

# --- Classifier only: digit output against the PyTorch model ------------------
# Reads sim_vectors.txt from the current directory, so run from output/ or copy
# the file next to the work library.
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
