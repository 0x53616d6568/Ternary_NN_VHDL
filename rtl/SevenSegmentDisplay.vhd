library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity SevenSegmentDisplay is
  generic (
    CLOCK_HZ : integer := 100000000
  );
  port (
    clk       : in  std_logic;
    digit     : in  std_logic_vector(3 downto 0);
    segments  : out std_logic_vector(7 downto 0);
    anodes    : out std_logic_vector(3 downto 0)
  );
end SevenSegmentDisplay;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of SevenSegmentDisplay is
  constant SCAN_DIV : integer := CLOCK_HZ / 4000;
  signal divider : integer range 0 to SCAN_DIV - 1 := 0;
  signal scan_digit : unsigned(1 downto 0) := (others => '0');
  signal pattern : std_logic_vector(6 downto 0);
begin
  process(clk)
  begin
    if rising_edge(clk) then
      if divider = SCAN_DIV - 1 then
        divider <= 0;
        scan_digit <= scan_digit + 1;
      else
        divider <= divider + 1;
      end if;
    end if;
  end process;

  -- Nexys 3 display anodes and cathode segments are active-low.
  anodes <= "1110" when scan_digit = "00" else
            "1111";

  with digit select
    pattern <= "1000000" when "0000",
               "1111001" when "0001",
               "0100100" when "0010",
               "0110000" when "0011",
               "0011001" when "0100",
               "0010010" when "0101",
               "0000010" when "0110",
               "1111000" when "0111",
               "0000000" when "1000",
               "0010000" when "1001",
               "1111111" when others;

  segments <= '1' & pattern when scan_digit = "00" else "11111111";
end rtl;
