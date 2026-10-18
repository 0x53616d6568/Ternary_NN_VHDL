-- Testbench for the streaming Walsh-Hadamard rotation.
--
-- Feeds the same binary images that the Python model consumes, captures the
-- rotated bus, and writes it to a file. compare_rotation.py then checks it
-- against train_mnist_ternary_v3.prepare_inputs element by element. Agreement
-- must be exact, because the hardware and the trained model are required to
-- compute the identical integer activations.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity tb_WalshHadamard is
end tb_WalshHadamard;

architecture simulation of tb_WalshHadamard is
  constant CLOCK_PERIOD : time := 10 ns;
  constant N_INPUTS     : integer := 784;
  constant BLOCK_SIZE   : integer := 8;
  constant SHIFT        : integer := 1;
  constant SUM_WIDTH    : integer := 6;
  constant OUT_WIDTH    : integer := 4;
  constant N_IMAGES     : integer := 4;

  signal clk       : std_logic := '0';
  signal start     : std_logic := '0';
  signal in_valid  : std_logic := '0';
  signal in_index  : std_logic_vector(9 downto 0) := (others => '0');
  signal in_data   : std_logic := '0';
  signal out_valid : std_logic;
  signal out_data  : std_logic_vector(N_INPUTS * OUT_WIDTH - 1 downto 0);
begin
  clk <= not clk after CLOCK_PERIOD / 2;

  dut : entity work.WalshHadamard
    generic map (
      N_INPUTS  => N_INPUTS,
      BLOCK_SIZE => BLOCK_SIZE,
      SHIFT     => SHIFT,
      SUM_WIDTH => SUM_WIDTH,
      OUT_WIDTH => OUT_WIDTH
    )
    port map (
      clk       => clk,
      start     => start,
      in_valid  => in_valid,
      in_index  => in_index,
      in_data   => in_data,
      out_valid => out_valid,
      out_data  => out_data
    );

  stimulus : process
    file image_file : text open read_mode is "rot_images.txt";
    file result_file : text open write_mode is "rot_hw.txt";
    variable input_line : line;
    variable output_line : line;
    variable pixel_byte : std_logic_vector(7 downto 0);
    variable pixel_index : integer;
    variable chunk : integer;
    variable status_text : string(1 to 4);
  begin
    for image_number in 1 to N_IMAGES loop
      readline(image_file, input_line);
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';
      pixel_index := 0;

      for byte_index in 0 to 97 loop
        hread(input_line, pixel_byte);
        for bit_index in 0 to 7 loop
          in_index <= std_logic_vector(to_unsigned(pixel_index, 10));
          in_data  <= pixel_byte(bit_index);
          in_valid <= '1';
          wait until rising_edge(clk);
          in_valid <= '0';
          pixel_index := pixel_index + 1;
        end loop;
      end loop;

      wait until out_valid = '1' for 100 ns;

      -- Write the rotated bus, 16 values per line, most significant first so
      -- the file reads in the same order as the flattened vector.
      chunk := 0;
      for index in 0 to N_INPUTS - 1 loop
        -- textio concatenates bare writes, so each value needs its own leading
        -- space or the numbers run together on the line.
        write(output_line, string'(" "));
        write(output_line,
              to_integer(signed(out_data((index + 1) * OUT_WIDTH - 1
                                         downto index * OUT_WIDTH))));
        chunk := chunk + 1;
        if chunk = 16 then
          writeline(result_file, output_line);
          chunk := 0;
        end if;
      end loop;
      report "image " & integer'image(image_number) & " captured" severity note;
    end loop;

    file_close(image_file);
    file_close(result_file);
    report "Wrote rot_hw.txt" severity note;
    std.env.stop;
    wait;
  end process;
end simulation;