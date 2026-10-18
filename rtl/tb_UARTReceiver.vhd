-- Unit test for UARTReceiver.
--
-- Sends all 256 byte values across the serial line and checks that each one is
-- recovered. The board-level test only ever exercises the 0xA5 sync byte and
-- image data, so this confirms the receive path is correct for every possible
-- value, including bit patterns that stress the start-bit and stop-bit
-- detection.
--
-- Framing is standard: idle high, start bit low, eight data bits least
-- significant first, stop bit high.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_UARTReceiver is
end tb_UARTReceiver;

architecture simulation of tb_UARTReceiver is
  constant CLOCK_PERIOD : time := 1 ns;
  constant CLOCK_HZ     : integer := 100000000;
  constant BAUD         : integer := 115200;
  constant CLKS_PER_BIT : integer := CLOCK_HZ / BAUD;   -- 868

  signal clk        : std_logic := '0';
  signal serial_in  : std_logic := '1';
  signal data_out   : std_logic_vector(7 downto 0);
  signal data_valid : std_logic;

  type rx_array is array (0 to 255) of std_logic_vector(7 downto 0);
  signal rx_buf   : rx_array := (others => (others => '0'));
  signal rx_count : integer range 0 to 256 := 0;
begin
  clk <= not clk after CLOCK_PERIOD / 2;

  dut : entity work.UARTReceiver
    generic map (CLOCK_HZ => CLOCK_HZ, BAUD => BAUD)
    port map (clk => clk, serial_in => serial_in,
              data_out => data_out, data_valid => data_valid);

  -- The receiver raises data_valid for a single cycle partway through the stop
  -- bit, while the next byte is still being transmitted, so checking for it
  -- after a send has finished always misses it. A monitor process watches the
  -- strobe continuously and records each byte into a shared buffer. A file was
  -- tried first, but the monitor is killed by std.env.stop before it closes the
  -- file, so the buffered contents were lost.
  monitor : process
  begin
    loop
      wait until rising_edge(clk);
      if data_valid = '1' and rx_count < 256 then
        rx_buf(rx_count) <= data_out;
        rx_count <= rx_count + 1;
      end if;
    end loop;
  end process;

  stimulus : process
    file result_file : text open write_mode is "uart_results.txt";
    variable output_line : line;
    variable passes      : integer := 0;
    variable failures    : integer := 0;
    variable sent_byte   : std_logic_vector(7 downto 0);
    variable total       : integer;
  begin
    write(output_line, string'("SENT RECEIVED RESULT"));
    writeline(result_file, output_line);

    for value in 0 to 255 loop
      sent_byte := std_logic_vector(to_unsigned(value, 8));

      serial_in <= '0';
      for i in 0 to CLKS_PER_BIT - 1 loop
        wait until rising_edge(clk);
      end loop;

      for bit_index in 0 to 7 loop
        serial_in <= sent_byte(bit_index);
        for i in 0 to CLKS_PER_BIT - 1 loop
          wait until rising_edge(clk);
        end loop;
      end loop;

      serial_in <= '1';
      for i in 0 to CLKS_PER_BIT - 1 loop
        wait until rising_edge(clk);
      end loop;

      -- Allow this byte's strobe to fire before starting the next frame.
      wait for CLKS_PER_BIT * 2 ns;
    end loop;

    serial_in <= '1';
    wait for 200 us;

    total := rx_count;
    for value in 0 to 255 loop
      if value < total then
        write(output_line, string'("0x"));
        write(output_line, to_hstring(std_logic_vector(to_unsigned(value, 8))));
        write(output_line, string'("  0x"));
        write(output_line, to_hstring(rx_buf(value)));
        if rx_buf(value) = std_logic_vector(to_unsigned(value, 8)) then
          write(output_line, string'("  OK"));
          passes := passes + 1;
        else
          write(output_line, string'("  DIFF"));
          failures := failures + 1;
        end if;
      else
        write(output_line, string'("byte never received"));
        failures := failures + 1;
      end if;
      writeline(result_file, output_line);
    end loop;

    report "----------------------------------------------" severity note;
    report "Bytes sent      : 256" severity note;
    report "Bytes received  : " & integer'image(total) severity note;
    report "Passed          : " & integer'image(passes) severity note;
    report "Failed          : " & integer'image(failures) severity note;

    file_close(result_file);
    std.env.stop;
    wait;
  end process;
end simulation;
