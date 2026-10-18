library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ImagePacketReceiver is
  port (
    clk          : in  std_logic;
    byte_in      : in  std_logic_vector(7 downto 0);
    byte_valid   : in  std_logic;
    pixel_we     : out std_logic;
    pixel_addr   : out std_logic_vector(9 downto 0);
    pixel_data   : out std_logic;
    image_done   : out std_logic
  );
end ImagePacketReceiver;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of ImagePacketReceiver is
  signal receiving    : std_logic := '0';
  signal unpacking    : std_logic := '0';
  signal bytes_seen   : integer range 0 to 98 := 0;
  signal bits_seen    : integer range 0 to 7 := 0;
  signal pixel_index  : integer range 0 to 783 := 0;
  signal addr_reg     : std_logic_vector(9 downto 0) := (others => '0');
  signal byte_shift   : std_logic_vector(7 downto 0) := (others => '0');
  signal write_reg    : std_logic := '0';
  signal data_reg     : std_logic := '0';
  signal done_reg     : std_logic := '0';
begin
  pixel_we   <= write_reg;
  pixel_addr <= addr_reg;
  pixel_data <= data_reg;
  image_done <= done_reg;

  process(clk)
  begin
    if rising_edge(clk) then
      write_reg <= '0';
      done_reg <= '0';

      if unpacking = '1' then
        write_reg <= '1';
        addr_reg <= std_logic_vector(to_unsigned(pixel_index, 10));
        data_reg <= byte_shift(0);
        byte_shift <= '0' & byte_shift(7 downto 1);

        if pixel_index = 783 then
          pixel_index <= 0;
          unpacking <= '0';
          receiving <= '0';
          done_reg <= '1';
        else
          pixel_index <= pixel_index + 1;
        end if;

        if bits_seen = 7 then
          bits_seen <= 0;
          unpacking <= '0';
        else
          bits_seen <= bits_seen + 1;
        end if;
      elsif byte_valid = '1' then
        if receiving = '0' then
          if byte_in = x"A5" then
            receiving <= '1';
            bytes_seen <= 0;
            pixel_index <= 0;
            bits_seen <= 0;
          end if;
        else
          byte_shift <= byte_in;
          unpacking <= '1';
          bits_seen <= 0;
          if bytes_seen = 97 then
            bytes_seen <= 98;
          else
            bytes_seen <= bytes_seen + 1;
          end if;
        end if;
      end if;
    end if;
  end process;
end rtl;
