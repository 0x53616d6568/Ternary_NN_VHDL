-- Testbench for the sequential ternary MAC engine.
--
-- Drives the rotated activations and the packed ternary weights for real MNIST
-- images and writes the resulting hidden vector. compare_mac.py checks it
-- against the trained model, and the agreement has to be exact: these integers
-- are what the ArgMax stage consumes, so any drift becomes a wrong digit.
--
-- A single process drives start and the weight bus. The engine consumes the
-- weight word for input i on the same cycle it presents address i, so the word
-- is placed before the edge that processes it.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity tb_TernaryMacEngine is
end tb_TernaryMacEngine;

architecture simulation of tb_TernaryMacEngine is
  constant CLOCK_PERIOD : time := 10 ns;
  constant N_INPUTS      : integer := 784;
  constant N_HIDDEN      : integer := 32;
  constant ACC_WIDTH     : integer := 32;
  constant HIDDEN_WIDTH  : integer := 8;
  constant X_WIDTH       : integer := 4;
  constant ADDR_WIDTH    : integer := 10;
  constant N_IMAGES      : integer := 4;
  constant PER_LINE      : integer := 16;

  type weight_array is array (0 to N_INPUTS - 1) of
    std_logic_vector(N_HIDDEN * 2 - 1 downto 0);
  type activation_array is array (0 to N_INPUTS - 1) of integer;

  -- Declared before the constant below, which calls it. Packed words are read as
  -- hex because a full 32-bit packed word overflows a VHDL integer.
  function read_weights return weight_array is
    variable table : weight_array := (others => (others => '0'));
    variable row    : line;
    variable word   : std_logic_vector(N_HIDDEN * 2 - 1 downto 0);
    file handle : text;
  begin
    file_open(handle, "w1_packed.txt", read_mode);
    for i in 0 to N_INPUTS - 1 loop
      readline(handle, row);
      hread(row, word);
      table(i) := word;
    end loop;
    file_close(handle);
    return table;
  end function read_weights;

  function read_bias return std_logic_vector is
    variable table : std_logic_vector(N_HIDDEN * ACC_WIDTH - 1 downto 0) := (others => '0');
    variable row    : line;
    variable value  : std_logic_vector(ACC_WIDTH - 1 downto 0);
    file handle : text;
  begin
    file_open(handle, "b1.txt", read_mode);
    for h in 0 to N_HIDDEN - 1 loop
      readline(handle, row);
      hread(row, value);
      table((h + 1) * ACC_WIDTH - 1 downto h * ACC_WIDTH) := value;
    end loop;
    file_close(handle);
    return table;
  end function read_bias;

  constant WEIGHTS : weight_array := read_weights;
  constant BIAS    : std_logic_vector(N_HIDDEN * ACC_WIDTH - 1 downto 0) := read_bias;

  signal clk    : std_logic := '0';
  signal start  : std_logic := '0';
  signal busy   : std_logic;
  signal done   : std_logic;
  signal waddr  : std_logic_vector(ADDR_WIDTH - 1 downto 0);
  signal x      : std_logic_vector(N_INPUTS * X_WIDTH - 1 downto 0);
  signal hidden : std_logic_vector(N_HIDDEN * HIDDEN_WIDTH - 1 downto 0);
  signal bias_s : std_logic_vector(N_HIDDEN * ACC_WIDTH - 1 downto 0) := BIAS;
  signal w_data : std_logic_vector(N_HIDDEN * 2 - 1 downto 0) := (others => '0');
begin
  clk <= not clk after CLOCK_PERIOD / 2;

  dut : entity work.TernaryMacEngine
    generic map (
      N_INPUTS     => N_INPUTS,
      N_HIDDEN     => N_HIDDEN,
      ACC_WIDTH    => ACC_WIDTH,
      HIDDEN_WIDTH => HIDDEN_WIDTH,
      X_WIDTH      => X_WIDTH,
      ADDR_WIDTH   => ADDR_WIDTH
    )
    port map (
      clk         => clk,
      start       => start,
      busy        => busy,
      done        => done,
      w_read_addr => waddr,
      w_read_data => w_data,
      x           => x,
      bias         => bias_s,
      hidden      => hidden
    );

  stimulus : process
    file rot_file    : text open read_mode is "rot_py.txt";
    file result_file : text open write_mode is "mac_hw.txt";

    variable row        : line;
    variable out_row    : line;
    variable values     : activation_array := (others => 0);
    variable index      : integer;
    variable column     : integer;
    variable first      : integer;
    variable x_value    : integer;
  begin
    for image_number in 1 to N_IMAGES loop
      -- rot_py.txt holds one line per 16 activations.
      index  := 0;
      column := 0;
      while index < N_INPUTS loop
        if column = 0 then
          readline(rot_file, row);
        end if;
        -- textio needs a plain variable as the destination, not an array element.
        read(row, x_value);
        values(index) := x_value;
        index  := index + 1;
        column := column + 1;
        if column = PER_LINE then
          column := 0;
        end if;
      end loop;

      for i in 0 to N_INPUTS - 1 loop
        x((i + 1) * X_WIDTH - 1 downto i * X_WIDTH)
          <= std_logic_vector(to_signed(values(i), X_WIDTH));
      end loop;

      wait for 1 ns;
      start <= '1';
      wait until rising_edge(clk);
      start <= '0';

      for i in 0 to N_INPUTS - 1 loop
        w_data <= WEIGHTS(i);
        wait until rising_edge(clk);
      end loop;

      wait until rising_edge(clk);
      wait until rising_edge(clk);

      for h in 0 to N_HIDDEN - 1 loop
        write(out_row, string'(" "));
        write(out_row, to_integer(unsigned(
          hidden((h + 1) * HIDDEN_WIDTH - 1 downto h * HIDDEN_WIDTH))));
      end loop;
      writeline(result_file, out_row);
      report "image " & integer'image(image_number) & " done" severity note;
    end loop;

    file_close(rot_file);
    file_close(result_file);
    report "Wrote mac_hw.txt" severity note;
    std.env.stop;
    wait;
  end process;
end simulation;
