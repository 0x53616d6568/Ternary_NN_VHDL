library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity UARTReceiver is
  generic (
    CLOCK_HZ : integer := 100000000;
    BAUD     : integer := 115200
  );
  port (
    clk        : in  std_logic;
    serial_in  : in  std_logic;
    data_out   : out std_logic_vector(7 downto 0);
    data_valid : out std_logic
  );
end UARTReceiver;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of UARTReceiver is
  constant CLKS_PER_BIT : integer := CLOCK_HZ / BAUD;
  type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
  signal state      : state_type := IDLE;
  signal timer      : integer range 0 to CLKS_PER_BIT := 0;
  signal bit_index  : integer range 0 to 7 := 0;
  signal shift_reg  : std_logic_vector(7 downto 0) := (others => '0');
  signal data_reg   : std_logic_vector(7 downto 0) := (others => '0');
  signal valid_reg  : std_logic := '0';
begin
  data_out   <= data_reg;
  data_valid <= valid_reg;

  process(clk)
  begin
    if rising_edge(clk) then
      valid_reg <= '0';

      case state is
        when IDLE =>
          if serial_in = '0' then
            timer <= CLKS_PER_BIT / 2;
            state <= START_BIT;
          end if;

        when START_BIT =>
          if timer = 0 then
            if serial_in = '0' then
              timer <= CLKS_PER_BIT - 1;
              bit_index <= 0;
              state <= DATA_BITS;
            else
              state <= IDLE;
            end if;
          else
            timer <= timer - 1;
          end if;

        when DATA_BITS =>
          if timer = 0 then
            shift_reg(bit_index) <= serial_in;
            timer <= CLKS_PER_BIT - 1;
            if bit_index = 7 then
              state <= STOP_BIT;
            else
              bit_index <= bit_index + 1;
            end if;
          else
            timer <= timer - 1;
          end if;

        when STOP_BIT =>
          if timer = 0 then
            if serial_in = '1' then
              data_reg <= shift_reg;
              valid_reg <= '1';
            end if;
            state <= IDLE;
          else
            timer <= timer - 1;
          end if;
      end case;
    end if;
  end process;
end rtl;
