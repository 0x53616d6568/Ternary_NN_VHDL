library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

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
    A       : in  std_logic_vector(N_INPUTS * DATA_WIDTH - 1 downto 0);
    Weights : in  std_logic_vector(N_INPUTS * N_OUTPUTS * WEIGHT_WIDTH - 1 downto 0);
    Biases  : in  std_logic_vector(N_OUTPUTS * ACC_WIDTH - 1 downto 0);
    Y       : out std_logic_vector(N_OUTPUTS * OUTPUT_WIDTH - 1 downto 0)
  );
end DenseLayer;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture combinational of DenseLayer is
begin
  process(A, Weights, Biases)
    variable accumulator : signed(ACC_WIDTH - 1 downto 0);
    variable input_value : signed(DATA_WIDTH - 1 downto 0);
    variable weight_value: signed(WEIGHT_WIDTH - 1 downto 0);
    variable product     : signed(DATA_WIDTH + WEIGHT_WIDTH - 1 downto 0);
    variable output_value: signed(OUTPUT_WIDTH - 1 downto 0);
    variable max_output  : signed(OUTPUT_WIDTH - 1 downto 0);
  begin
    max_output := (others => '1');
    max_output(OUTPUT_WIDTH - 1) := '0';

    for output_index in 0 to N_OUTPUTS - 1 loop
      accumulator := signed(
        Biases((output_index + 1) * ACC_WIDTH - 1 downto output_index * ACC_WIDTH)
      );

      for input_index in 0 to N_INPUTS - 1 loop
        input_value := signed(
          A((input_index + 1) * DATA_WIDTH - 1 downto input_index * DATA_WIDTH)
        );
        weight_value := signed(
          Weights((output_index * N_INPUTS + input_index + 1) * WEIGHT_WIDTH - 1
                  downto (output_index * N_INPUTS + input_index) * WEIGHT_WIDTH)
        );

        product := input_value * weight_value;
        accumulator := accumulator + resize(product, ACC_WIDTH);
      end loop;

      if USE_RELU then
        if accumulator(ACC_WIDTH - 1) = '1' then
          output_value := (others => '0');
        elsif accumulator > resize(max_output, ACC_WIDTH) then
          output_value := max_output;
        else
          output_value := resize(accumulator, OUTPUT_WIDTH);
        end if;
      else
        output_value := resize(accumulator, OUTPUT_WIDTH);
      end if;

      Y((output_index + 1) * OUTPUT_WIDTH - 1 downto output_index * OUTPUT_WIDTH)
        <= std_logic_vector(output_value);
    end loop;
  end process;
end combinational;
