library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Sequential ternary MAC engine for the first layer.
--
-- A fully parallel first layer at 784 x 16 needs 12,544 accumulators and about
-- 25,000 LUTs, which does not fit the Artix-7's 15,850. Streaming one input per
-- cycle instead needs 16 accumulators and roughly 160 LUTs, and the whole weight
-- matrix lives in one block RAM.
--
-- Weights arrive 16 at a time from memory, already transposed so that the word
-- for input i holds one weight per hidden unit, two bits each. Two-bit
-- two's-complement encodes the ternary set directly: 00 = 0, 01 = +1, 11 = -1.
--
-- The multiply is written as an explicit conditional add or subtract rather
-- than a "*", because a ternary weight times a small activation is either the
-- activation, its negation, or nothing. Writing it this way guarantees
-- synthesis infers an adder and never reaches for a DSP.

entity TernaryMacEngine is
  generic (
    N_INPUTS    : integer := 784;
    N_HIDDEN    : integer := 16;
    ACC_WIDTH   : integer := 32;
    HIDDEN_WIDTH: integer := 8;
    HIDDEN_MAX  : integer := 127;
    X_WIDTH     : integer := 4;
    ADDR_WIDTH  : integer := 10
  );
  port (
    clk          : in  std_logic;
    start        : in  std_logic;
    busy         : out std_logic;
    done         : out std_logic;
    w_read_addr  : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
    w_read_data  : in  std_logic_vector(N_HIDDEN * 2 - 1 downto 0);
    x            : in  std_logic_vector(N_INPUTS * X_WIDTH - 1 downto 0);
    bias         : in  std_logic_vector(N_HIDDEN * ACC_WIDTH - 1 downto 0);
    hidden       : out std_logic_vector(N_HIDDEN * HIDDEN_WIDTH - 1 downto 0)
  );
end TernaryMacEngine;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of TernaryMacEngine is
  type acc_array is array (0 to N_HIDDEN - 1) of signed(ACC_WIDTH - 1 downto 0);
  signal accum : acc_array := (others => (others => '0'));
  signal index : integer range 0 to N_INPUTS := 0;
  signal active : std_logic := '0';
  signal finished : std_logic := '0';

  signal x_now : signed(X_WIDTH - 1 downto 0);
  signal w_now : signed(1 downto 0);
begin
  x_now <= signed(x(index * X_WIDTH + X_WIDTH - 1 downto index * X_WIDTH));
  w_read_addr <= std_logic_vector(to_unsigned(index, ADDR_WIDTH));
  busy    <= active;
  done    <= finished;

  process(clk)
    variable delta : signed(ACC_WIDTH - 1 downto 0);
    variable value : signed(ACC_WIDTH - 1 downto 0);
  begin
    if rising_edge(clk) then
      finished <= '0';

      if start = '1' then
        -- Seeding with the bias makes the running sum bias + sum(w * x), which
        -- is exactly what the trained layer computes, and costs nothing: the
        -- accumulator has to be loaded with something anyway.
        for h in 0 to N_HIDDEN - 1 loop
          accum(h) <= signed(bias((h + 1) * ACC_WIDTH - 1 downto h * ACC_WIDTH));
        end loop;
        index  <= 0;
        active <= '1';

      elsif active = '1' then
        for h in 0 to N_HIDDEN - 1 loop
          w_now <= signed(w_read_data(h * 2 + 1 downto h * 2));
          -- 00 = 0, 01 = +1, 11 = -1. 2'b10 would mean -2 and is never emitted.
          if w_read_data(h * 2 + 1 downto h * 2) = "01" then
            delta := resize(x_now, ACC_WIDTH);
          elsif w_read_data(h * 2 + 1 downto h * 2) = "11" then
            delta := -resize(x_now, ACC_WIDTH);
          else
            delta := (others => '0');
          end if;
          value := accum(h) + delta;
          -- The accumulator must run freely while the partial sum builds. Only
          -- the completed total gets the integer ReLU and the saturation to
          -- HIDDEN_MAX; clamping every cycle would truncate intermediate sums.
          if index /= N_INPUTS - 1 then
            accum(h) <= value;
          elsif value(ACC_WIDTH - 1) = '1' then
            accum(h) <= (others => '0');
          elsif value > to_signed(HIDDEN_MAX, ACC_WIDTH) then
            accum(h) <= to_signed(HIDDEN_MAX, ACC_WIDTH);
          else
            accum(h) <= value;
          end if;
        end loop;

        if index = N_INPUTS - 1 then
          index  <= 0;
          active <= '0';
          finished <= '1';
        else
          index <= index + 1;
        end if;
      end if;
    end if;
  end process;

  process(all)
  begin
    for h in 0 to N_HIDDEN - 1 loop
      hidden((h + 1) * HIDDEN_WIDTH - 1 downto h * HIDDEN_WIDTH)
        <= std_logic_vector(resize(accum(h), HIDDEN_WIDTH));
    end loop;
  end process;
end rtl;