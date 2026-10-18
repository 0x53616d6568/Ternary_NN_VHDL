library ieee;
use ieee.std_logic_1164.all;

entity Nexys3Top is
  port (
    clk100 : in  std_logic;
    RsRx   : in  std_logic;
    seg    : out std_logic_vector(7 downto 0);
    an     : out std_logic_vector(3 downto 0)
  );
end Nexys3Top;

library ieee;
use ieee.std_logic_1164.all;

architecture structural of Nexys3Top is
  component UARTReceiver is
    generic (
      CLOCK_HZ : integer;
      BAUD     : integer
    );
    port (
      clk        : in  std_logic;
      serial_in  : in  std_logic;
      data_out   : out std_logic_vector(7 downto 0);
      data_valid : out std_logic
    );
  end component;

  component ImagePacketReceiver is
    port (
      clk        : in  std_logic;
      byte_in    : in  std_logic_vector(7 downto 0);
      byte_valid : in  std_logic;
      pixel_we   : out std_logic;
      pixel_addr : out std_logic_vector(9 downto 0);
      pixel_data : out std_logic;
      image_done : out std_logic
    );
  end component;

  component NumberRecognition is
    port (
      clk : in std_logic;
      pixel_write_enable : in std_logic;
      pixel_write_addr : in std_logic_vector(9 downto 0);
      pixel_write_data : in std_logic;
      digit : out std_logic_vector(3 downto 0)
    );
  end component;

  component SevenSegmentDisplay is
    generic (
      CLOCK_HZ : integer
    );
    port (
      clk      : in  std_logic;
      digit    : in  std_logic_vector(3 downto 0);
      segments : out std_logic_vector(7 downto 0);
      anodes   : out std_logic_vector(3 downto 0)
    );
  end component;

  signal rx_byte : std_logic_vector(7 downto 0);
  signal rx_valid, pixel_we, pixel_data, image_done : std_logic;
  signal pixel_addr : std_logic_vector(9 downto 0);
  signal predicted_digit : std_logic_vector(3 downto 0);
begin
  uart_rx : UARTReceiver
    generic map (CLOCK_HZ => 100000000, BAUD => 115200)
    port map (clk => clk100, serial_in => RsRx,
              data_out => rx_byte, data_valid => rx_valid);

  image_rx : ImagePacketReceiver
    port map (clk => clk100, byte_in => rx_byte, byte_valid => rx_valid,
              pixel_we => pixel_we, pixel_addr => pixel_addr,
              pixel_data => pixel_data, image_done => image_done);

  classifier : NumberRecognition
    port map (
      clk => clk100,
      pixel_write_enable => pixel_we,
      pixel_write_addr => pixel_addr,
      pixel_write_data => pixel_data,
      digit => predicted_digit
    );

  display : SevenSegmentDisplay
    generic map (CLOCK_HZ => 100000000)
    port map (clk => clk100, digit => predicted_digit,
              segments => seg, anodes => an);
end structural;
