-- Unit test for SevenSegmentDisplay.
--
-- The board-level test only exercises whichever digits appear in the sampled
-- images. This drives all ten digit codes directly and checks the multiplexed
-- segment output against the standard active-low cathode encoding, so every
-- entry in the pattern table is verified rather than only the common ones.
--
-- The Nexys 3 uses active-low anodes and cathodes: a '0' in the table means the
-- segment is lit, and digit position zero is selected when the anodes read
-- "1110".

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity tb_SevenSegmentDisplay is
end tb_SevenSegmentDisplay;

architecture simulation of tb_SevenSegmentDisplay is
  constant CLOCK_PERIOD : time := 1 ns;
  -- A short clock keeps the scan divider small so a full multiplex cycle is
  -- quick to simulate; the divider ratio is what matters, not real time.
  constant CLOCK_HZ     : integer := 4000 * 64;

  signal clk      : std_logic := '0';
  signal digit    : std_logic_vector(3 downto 0) := (others => '0');
  signal segments : std_logic_vector(7 downto 0);
  signal anodes   : std_logic_vector(3 downto 0);
begin
  clk <= not clk after CLOCK_PERIOD / 2;

  dut : entity work.SevenSegmentDisplay
    generic map (CLOCK_HZ => CLOCK_HZ)
    port map (clk => clk, digit => digit,
              segments => segments, anodes => anodes);

  stimulus : process
    file result_file : text open write_mode is "display_results.txt";
    variable output_line : line;
    variable passes      : integer := 0;
    variable failures    : integer := 0;
    variable want        : std_logic_vector(6 downto 0);
    variable got         : std_logic_vector(6 downto 0);

    -- Standard active-low cathode encoding, bit order dp g f e d c b a.
    -- Codes above 9 are not valid digits and must blank the display.
    function expected_pattern(d : integer) return std_logic_vector is
    begin
      case d is
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
    write(output_line, string'("CODE SEGMENTS ANODES RESULT"));
    writeline(result_file, output_line);
    report "CODE SEGMENTS ANODES RESULT" severity note;

    -- All sixteen 4-bit codes, so the valid digits and the blanking behaviour
    -- for undefined codes are both covered.
    for code in 0 to 15 loop
      digit <= std_logic_vector(to_unsigned(code, 4));

      -- Wait for a scan position in which digit zero is selected, then sample
      -- on the following edge so the pattern has settled.
      for attempt in 0 to 4000 loop
        wait until rising_edge(clk);
        if anodes = "1110" then
          exit;
        end if;
      end loop;
      wait until rising_edge(clk);

      want := expected_pattern(code);
      got  := segments(6 downto 0);

      write(output_line, code, right, 4);
      write(output_line, string'(" "));
      write(output_line, got);
      write(output_line, string'("   "));
      write(output_line, anodes);
      if got = want then
        write(output_line, string'("  OK"));
        passes := passes + 1;
      else
        write(output_line, string'("  DIFF (want "));
        write(output_line, want);
        write(output_line, string'(")"));
        failures := failures + 1;
      end if;
      writeline(result_file, output_line);
      report output_line.all severity note;
    end loop;

    report "------------------------------------------------" severity note;
    report "Digit codes verified : " & integer'image(passes + failures) severity note;
    report "Passed               : " & integer'image(passes) severity note;
    report "Failed               : " & integer'image(failures) severity note;

    file_close(result_file);
    std.env.stop;
    wait;
  end process;
end simulation;
