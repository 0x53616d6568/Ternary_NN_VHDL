-- Board-level testbench for Nexys3Top.
--
-- This exercises the complete path a user actually sees:
--
--   serial line -> UARTReceiver -> ImagePacketReceiver -> NumberRecognition
--               -> SevenSegmentDisplay -> seg / an pins
--
-- Bytes are driven as real UART framing on RsRx at 115200 baud against a
-- 100 MHz clock, and the result is read back by decoding the multiplexed
-- seven-segment outputs. Verifying the segments rather than an internal digit
-- signal is deliberate: it proves the digit reaches the display correctly,
-- which is the only thing an observer of the board can confirm.
--
-- The clock is driven at 1 ns rather than the real 10 ns. The UART and display
-- are clock-cycle accurate rather than time accurate, so the shorter period
-- changes nothing about behaviour and keeps a 99-byte transfer fast enough to
-- simulate comfortably.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity tb_Nexys3Top is
end tb_Nexys3Top;

architecture simulation of tb_Nexys3Top is
  constant CLOCK_PERIOD  : time := 1 ns;
  constant CLKS_PER_BIT  : integer := 868;    -- 100 MHz / 115200 baud
  constant N_IMAGES      : integer := 10;
  constant SETTLE        : time := 40 us;    -- rotation plus 784 MAC cycles

  signal clk  : std_logic := '0';
  signal rsrx : std_logic := '1';            -- idle high
  signal seg  : std_logic_vector(7 downto 0);
  signal an   : std_logic_vector(3 downto 0);
begin
  clk <= not clk after CLOCK_PERIOD / 2;

  dut : entity work.Nexys3Top
    port map (clk100 => clk, RsRx => rsrx, seg => seg, an => an);

  stimulus : process
    file vector_file : text open read_mode is "sim_vectors.txt";
    file result_file : text open write_mode is "board_results.txt";

    variable input_line  : line;
    variable output_line : line;
    variable pixel_byte  : std_logic_vector(7 downto 0);
    variable label_value : integer;
    variable software    : integer;
    variable display_ok  : integer := 0;
    variable image_count : integer := 0;
    variable pattern_ok  : boolean;

    -- Drive one byte with standard UART framing: start bit, eight data bits
    -- least significant first, then a stop bit. Each bit is held for exactly
    -- CLKS_PER_BIT cycles so the receiver, which samples mid-bit, stays aligned.
    procedure send_byte(value : std_logic_vector(7 downto 0)) is
    begin
      rsrx <= '0';
      for i in 0 to CLKS_PER_BIT - 1 loop
        wait until rising_edge(clk);
      end loop;
      for bit_index in 0 to 7 loop
        rsrx <= value(bit_index);
        for i in 0 to CLKS_PER_BIT - 1 loop
          wait until rising_edge(clk);
        end loop;
      end loop;
      rsrx <= '1';
      for i in 0 to CLKS_PER_BIT - 1 loop
        wait until rising_edge(clk);
      end loop;
    end procedure send_byte;

    -- Active-low cathode patterns, identical to the encoding inside
    -- SevenSegmentDisplay. Index is the digit, bit order is dp g f e d c b a.
    function expected_pattern(digit : integer) return std_logic_vector is
    begin
      -- 'array' is a VHDL reserved word, so the patterns are selected with a
      -- case statement rather than a lookup table.
      case digit is
        when 0      => return "1000000";
        when 1      => return "1111001";
        when 2      => return "0100100";
        when 3      => return "0110000";
        when 4      => return "0011001";
        when 5      => return "0010010";
        when 6      => return "0000010";
        when 7      => return "1111000";
        when 8      => return "0000000";
        when 9      => return "0010000";
        when others => return "1111111";
      end case;
    end function expected_pattern;
  begin
    write(output_line, string'("IMG LABEL SW SEGMENTS RESULT"));
    writeline(result_file, output_line);
    report "IMG LABEL SW SEGMENTS RESULT" severity note;

    while image_count < N_IMAGES and not endfile(vector_file) loop
      readline(vector_file, input_line);
      read(input_line, label_value);
      read(input_line, software);
      image_count := image_count + 1;

      send_byte(x"A5");
      for byte_index in 0 to 97 loop
        hread(input_line, pixel_byte);
        send_byte(pixel_byte);
      end loop;

      wait for SETTLE;

      -- The display is multiplexed across four digit positions at 4 kHz. Only
      -- position zero carries the pattern; the others are blanked. Wait for
      -- position zero and sample it.
      for attempt in 0 to 400 loop
        wait for 1 us;
        if an = "1110" then
          exit;
        end if;
      end loop;

      pattern_ok := (seg(6 downto 0) = expected_pattern(software));
      if pattern_ok then
        display_ok := display_ok + 1;
      end if;

      write(output_line, image_count, right, 3);
      write(output_line, string'("  "));
      write(output_line, label_value, right, 5);
      write(output_line, string'("  "));
      write(output_line, software, right, 2);
      write(output_line, string'("  "));
      write(output_line, to_integer(unsigned(seg)));
      if pattern_ok then
        write(output_line, string'("  OK"));
      else
        write(output_line, string'("  DIFF"));
      end if;
      writeline(result_file, output_line);
      report output_line.all severity note;
    end loop;

    assert image_count > 0
      report "No samples found in sim_vectors.txt" severity failure;

    report "----------------------------------------------------" severity note;
    report "Images sent over UART              : " & integer'image(image_count) severity note;
    report "Display showed the expected digit : " & integer'image(display_ok)
           & " / " & integer'image(image_count) severity note;

    file_close(vector_file);
    file_close(result_file);
    std.env.stop;
    wait;
  end process;
end simulation;
