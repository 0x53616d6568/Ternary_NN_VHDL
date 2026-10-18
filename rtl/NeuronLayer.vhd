-- NeuronLayer: N_HIDDEN TernaryNeuron instances plus bias extraction.
--
-- This is the "layer" block. It holds no logic of its own beyond slicing the
-- packed weight word and the bias vector; the arithmetic lives entirely in the
-- neurons, so the same neuron can be reused at any depth.
--
-- Weight packing, one word per input, two bits per hidden unit:
--
--     w_data(k) = [ w31 w30 | w29 w28 | ... | w1 w0 ]
--                   unit 31        unit 0
--
-- Biases arrive as one flat vector, unit h at
-- Bias((h+1)*ACC_WIDTH-1 downto h*ACC_WIDTH), matching the slice order the
-- exporter writes.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity NeuronLayer is
  generic (
    N_HIDDEN     : integer := 32;
    ACC_WIDTH    : integer := 32;
    HIDDEN_WIDTH : integer := 8;
    HIDDEN_MAX   : integer := 127;
    X_WIDTH      : integer := 4
  );
  port (
    clk         : in  std_logic;
    Start       : in  std_logic;
    FinalCycle  : in  std_logic;
    WData       : in  std_logic_vector(N_HIDDEN * 2 - 1 downto 0);
    X           : in  std_logic_vector(X_WIDTH - 1 downto 0);
    Biases      : in  std_logic_vector(N_HIDDEN * ACC_WIDTH - 1 downto 0);
    Hidden      : out std_logic_vector(N_HIDDEN * HIDDEN_WIDTH - 1 downto 0)
  );
end NeuronLayer;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of NeuronLayer is
begin
  neurons_g : for h in 0 to N_HIDDEN - 1 generate
    neuron : entity work.TernaryNeuron
      generic map (
        ACC_WIDTH    => ACC_WIDTH,
        HIDDEN_WIDTH => HIDDEN_WIDTH,
        HIDDEN_MAX   => HIDDEN_MAX,
        X_WIDTH      => X_WIDTH
      )
      port map (
        clk        => clk,
        Start      => Start,
        FinalCycle => FinalCycle,
        W          => WData(h * 2 + 1 downto h * 2),
        X          => X,
        Bias       => Biases((h + 1) * ACC_WIDTH - 1 downto h * ACC_WIDTH),
        Hidden     => Hidden((h + 1) * HIDDEN_WIDTH - 1 downto h * HIDDEN_WIDTH)
      );
  end generate;
end rtl;
