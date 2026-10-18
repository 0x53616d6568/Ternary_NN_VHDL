-- TernaryMacEngine: the sequential controller for one ternary layer.
--
-- There is NO ARITHMETIC IN THIS FILE. It is a sequencer: a counter, an address
-- generator and a start/done handshake. All of the arithmetic lives in
-- NeuronLayer -> TernaryNeuron -> {TernaryApply, Accumulator, Saturate}.
--
-- Why a fully parallel layer is not an option: 784 x 32 products would need
-- 26,138 adders, roughly 418,000 LUTs, against the Artix-7's 15,850. Streaming
-- one input per cycle needs N_HIDDEN accumulators and one adder each. The
-- trade is 784 cycles of latency for a design that fits at all.
--
-- Dataflow, across the whole engine:
--
--     address(k) = k
--     x(k)       = X(address(k) * X_WIDTH +: X_WIDTH)
--     A[h][0]    = Bias[h]
--     A[h][k+1]  = A[h][k] + T(W(address(k))[h]) * x(k)
--     hidden[h]  = Clamp(A[h][N_INPUTS-1], 0, HIDDEN_MAX)
--
-- with the clamp applied only when k = N_INPUTS-1.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity TernaryMacEngine is
  generic (
    N_INPUTS     : integer := 784;
    N_HIDDEN     : integer := 32;
    ACC_WIDTH    : integer := 32;
    HIDDEN_WIDTH : integer := 8;
    HIDDEN_MAX   : integer := 127;
    X_WIDTH      : integer := 4;
    ADDR_WIDTH   : integer := 10;
    -- Bias vector, passed in rather than imported, so this engine can be
    -- instantiated at any depth without a package dependency.
    BIASES       : in  std_logic_vector(N_HIDDEN * ACC_WIDTH - 1 downto 0) := (others => '0')
  );
  port (
    clk          : in  std_logic;
    start        : in  std_logic;
    busy         : out std_logic;
    done         : out std_logic;
    w_read_addr  : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    w_read_data  : in  std_logic_vector(N_HIDDEN * 2 - 1 downto 0);
    x            : in  std_logic_vector(N_INPUTS * X_WIDTH - 1 downto 0);
    hidden       : out std_logic_vector(N_HIDDEN * HIDDEN_WIDTH - 1 downto 0)
  );
end TernaryMacEngine;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of TernaryMacEngine is
  signal index     : integer range 0 to N_INPUTS := 0;
  signal active    : std_logic := '0';
  signal finished  : std_logic := '0';
  signal final_cyc : std_logic;
  signal x_now     : std_logic_vector(X_WIDTH - 1 downto 0);
begin
  -- The address generator walks the INPUTS 0..N_INPUTS-1, not the weights.
  -- That is why ADDR_WIDTH stays at 10 bits regardless of N_HIDDEN: it was
  -- already wide enough for 784 entries at either width.
  w_read_addr <= std_logic_vector(to_unsigned(index, ADDR_WIDTH));

  -- Combinational read of the current input from the rotated bus.
  x_now <= x(index * X_WIDTH + X_WIDTH - 1 downto index * X_WIDTH);

  -- '1' on the cycle that consumes the last input, which is the only cycle on
  -- which the ReLU and saturation may be applied.
  final_cyc <= '1' when index = N_INPUTS - 1 else '0';

  layer : entity work.NeuronLayer
    generic map (
      N_HIDDEN     => N_HIDDEN,
      ACC_WIDTH    => ACC_WIDTH,
      HIDDEN_WIDTH => HIDDEN_WIDTH,
      HIDDEN_MAX   => HIDDEN_MAX,
      X_WIDTH      => X_WIDTH
    )
    port map (
      clk        => clk,
      Start      => start,
      FinalCycle => final_cyc,
      WData      => w_read_data,
      X          => x_now,
      Biases     => BIASES,
      Hidden     => hidden
    );

  -- The sequencer. Load the bias on start, then one input per cycle.
  process(clk)
  begin
    if rising_edge(clk) then
      finished <= '0';

      if start = '1' then
        index  <= 0;
        active <= '1';
      elsif active = '1' then
        if index = N_INPUTS - 1 then
          index    <= 0;
          active   <= '0';
          finished <= '1';
        else
          index <= index + 1;
        end if;
      end if;
    end if;
  end process;

  busy <= active;
  done <= finished;
end rtl;
