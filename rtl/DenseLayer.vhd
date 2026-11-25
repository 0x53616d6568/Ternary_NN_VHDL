library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Second (output) layer: N_INPUTS x N_OUTPUTS full-precision products.
--
-- SEQUENTIAL ON PURPOSE. The original version evaluated all 320 products
-- (32 inputs x 10 classes) in one combinational process. On a Spartan-6 that is
-- 320 multipliers in a single cycle, against 32 DSP48A1 slices on the whole
-- device, so most of them spill into LUTs at roughly 60-80 LUTs each: tens of
-- thousands of LUTs on a part with 9,112. The layer is therefore walked one
-- input per cycle with N_OUTPUTS multipliers, which is 10 DSP slices and one
-- adder per class.
--
-- The arithmetic is unchanged and still exact: for each output the accumulator
-- starts at its bias and adds resize(A[i] * W[o][i], ACC_WIDTH) in increasing
-- i. That is the same sequence of integer additions as the combinational
-- version, so no result can differ.
--
-- Latency is N_INPUTS + 1 cycles (32 + 1 for the deployed shape), which is
-- negligible against the 784-cycle MAC pass and invisible next to the UART
-- frame that carries the image.

entity DenseLayer is
  generic (
    N_INPUTS    : integer := 4;
    N_OUTPUTS   : integer := 2;
    DATA_WIDTH  : integer := 8;
    WEIGHT_WIDTH: integer := 8;
    ACC_WIDTH   : integer := 32;
    OUTPUT_WIDTH: integer := 8;
    USE_RELU    : boolean := true
  );
  port (
    clk     : in  std_logic;
    start   : in  std_logic;
    done    : out std_logic;
    A       : in  std_logic_vector(N_INPUTS * DATA_WIDTH - 1 downto 0);
    Weights : in  std_logic_vector(N_INPUTS * N_OUTPUTS * WEIGHT_WIDTH - 1 downto 0);
    Biases  : in  std_logic_vector(N_OUTPUTS * ACC_WIDTH - 1 downto 0);
    Y       : out std_logic_vector(N_OUTPUTS * OUTPUT_WIDTH - 1 downto 0)
  );
end DenseLayer;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture sequential of DenseLayer is
  type acc_array is array (0 to N_OUTPUTS - 1) of signed(ACC_WIDTH - 1 downto 0);
  signal acc   : acc_array := (others => (others => '0'));
  signal index : integer range 0 to N_INPUTS - 1 := 0;
  signal busy  : std_logic := '0';
begin
  process(clk)
    variable input_value  : signed(DATA_WIDTH - 1 downto 0);
    variable weight_value : signed(WEIGHT_WIDTH - 1 downto 0);
    variable product      : signed(DATA_WIDTH + WEIGHT_WIDTH - 1 downto 0);
    variable updated      : signed(ACC_WIDTH - 1 downto 0);
    variable output_value : signed(OUTPUT_WIDTH - 1 downto 0);
    variable max_output   : signed(OUTPUT_WIDTH - 1 downto 0);
    variable w_index      : integer range 0 to N_INPUTS * N_OUTPUTS - 1;
  begin
    if rising_edge(clk) then
      done <= '0';

      if start = '1' then
        for output_index in 0 to N_OUTPUTS - 1 loop
          acc(output_index) <= signed(
            Biases((output_index + 1) * ACC_WIDTH - 1 downto output_index * ACC_WIDTH)
          );
        end loop;
        index <= 0;
        busy  <= '1';

      elsif busy = '1' then
        input_value := signed(A((index + 1) * DATA_WIDTH - 1 downto index * DATA_WIDTH));

        max_output := (others => '1');
        max_output(OUTPUT_WIDTH - 1) := '0';

        for output_index in 0 to N_OUTPUTS - 1 loop
          w_index := output_index * N_INPUTS + index;
          weight_value := signed(
            Weights((w_index + 1) * WEIGHT_WIDTH - 1 downto w_index * WEIGHT_WIDTH)
          );

          product := input_value * weight_value;
          updated := acc(output_index) + resize(product, ACC_WIDTH);
          acc(output_index) <= updated;

          -- The last input completes every accumulator, so that is the only
          -- cycle on which the result is scaled and committed.
          if index = N_INPUTS - 1 then
            if USE_RELU then
              if updated(ACC_WIDTH - 1) = '1' then
                output_value := (others => '0');
              elsif updated > resize(max_output, ACC_WIDTH) then
                output_value := max_output;
              else
                output_value := resize(updated, OUTPUT_WIDTH);
              end if;
            else
              output_value := resize(updated, OUTPUT_WIDTH);
            end if;
            Y((output_index + 1) * OUTPUT_WIDTH - 1 downto output_index * OUTPUT_WIDTH)
              <= std_logic_vector(output_value);
          end if;
        end loop;

        if index = N_INPUTS - 1 then
          busy <= '0';
          done <= '1';
        else
          index <= index + 1;
        end if;
      end if;
    end if;
  end process;
end sequential;
