-- End-to-end testbench for the ternary classifier.
--
-- Streams each MNIST image into NumberRecognition exactly as the UART path
-- does and reads back the predicted digit. The summary reports how often the
-- VHDL agrees with the PyTorch prediction, and how often the digit is correct.
-- Agreement with the software model is the number that matters: it is what
-- proves the FPGA is running the network that was trained and evaluated.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity tb_NumberRecognition is
end tb_NumberRecognition;

architecture simulation of tb_NumberRecognition is
  constant CLOCK_PERIOD : time := 10 ns;
  -- The rotation overlaps reception, so after the last pixel only the 784-cycle
  -- MAC pass remains. 10 us covers that with margin and keeps a 10,000-image run
  -- tractable; anything shorter would race the accumulator.
  constant SETTLE        : time := 10 us;

  signal clk         : std_logic := '0';
  signal pixel_we    : std_logic := '0';
  signal pixel_addr  : std_logic_vector(9 downto 0) := (others => '0');
  signal pixel_data  : std_logic := '0';
  signal digit       : std_logic_vector(3 downto 0);
begin
  clk <= not clk after CLOCK_PERIOD / 2;

  dut : entity work.NumberRecognition
    port map (
      clk               => clk,
      pixel_write_enable=> pixel_we,
      pixel_write_addr  => pixel_addr,
      pixel_write_data  => pixel_data,
      digit             => digit
    );

  stimulus : process
    file vector_file : text open read_mode is "sim_vectors.txt";
    file result_file : text open write_mode is "simulation_results.txt";

    variable input_line  : line;
    variable output_line : line;
    variable pixel_byte  : std_logic_vector(7 downto 0);
    variable label_value : integer;
    variable software    : integer;
    variable hardware    : integer;
    variable pixel_index : integer;
    variable image_number : integer := 0;
    variable model_match : integer := 0;
    variable correct     : integer := 0;
    variable verdict     : string(1 to 4);
  begin
    write(output_line, string'("IMG LABEL SW FPGA MATCH OK"));
    writeline(result_file, output_line);
    report "IMG LABEL SW FPGA MATCH OK" severity note;

    while not endfile(vector_file) loop
      readline(vector_file, input_line);
      read(input_line, label_value);
      read(input_line, software);
      pixel_index := 0;

      for byte_index in 0 to 97 loop
        hread(input_line, pixel_byte);
        for bit_index in 0 to 7 loop
          wait until falling_edge(clk);
          pixel_we   <= '1';
          pixel_addr <= std_logic_vector(to_unsigned(pixel_index, 10));
          pixel_data <= pixel_byte(bit_index);
          wait until rising_edge(clk);
          pixel_we   <= '0';
          pixel_index := pixel_index + 1;
        end loop;
      end loop;

      -- The rotation overlaps reception, so the only wait left is the MAC pass.
      wait for SETTLE;
      hardware := to_integer(unsigned(digit));
      image_number := image_number + 1;

      if hardware = software then
        model_match := model_match + 1;
        verdict := "OK  ";
      else
        verdict := "DIFF";
      end if;
      if hardware = label_value then
        correct := correct + 1;
      end if;

      write(output_line, image_number, right, 3);
      write(output_line, string'("  "));
      write(output_line, label_value, right, 5);
      write(output_line, string'("  "));
      write(output_line, software, right, 2);
      write(output_line, string'("  "));
      write(output_line, hardware, right, 4);
      write(output_line, string'("  "));
      write(output_line, verdict);
      writeline(result_file, output_line);
      -- Per-image detail goes to the file only. Reporting 10,000 lines would
      -- dominate the run time and bury the summary.
    end loop;

    assert image_number > 0
      report "No samples found in sim_vectors.txt" severity failure;

    report "--------------------------------------------------" severity note;
    report "Images simulated                    : " & integer'image(image_number) severity note;
    report "Hardware matches software model      : " & integer'image(model_match)
           & " / " & integer'image(image_number) severity note;
    report "Hardware digit correct vs MNIST label: " & integer'image(correct)
           & " / " & integer'image(image_number) severity note;

    file_close(vector_file);
    file_close(result_file);
    std.env.stop;
    wait;
  end process;
end simulation;