-- Saturating clamp to [0, HIDDEN_MAX].
--
-- The integer ReLU. Two's complement sign-bit test plus a magnitude compare:
--
--   if sign = 1            -> 0        (negative: ReLU)
--   elsif value > MAX      -> MAX      (saturate, keeps HIDDEN_WIDTH bits)
--   else                   -> value
--
-- ReLU.vhd performs the sign mask alone; this component adds the saturation
-- so HIDDEN_WIDTH can stay at 8 bits without overflow.
--
-- CRITICAL: this must be applied only on the FINAL cycle of the accumulation.
-- Clamping every cycle truncates intermediate partial sums and produces wrong
-- answers. That was a real defect class found during verification -- see the
-- FinalCycle input on Accumulator, which is what enforces it.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Saturate is
  generic (
    RESOLUTION : integer := 32;
    MAX_VALUE  : integer := 127
  );
  port (
    Value  : in  std_logic_vector(RESOLUTION - 1 downto 0);  -- signed
    Y      : out std_logic_vector(RESOLUTION - 1 downto 0)
  );
end Saturate;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of Saturate is
  signal value_signed : signed(RESOLUTION - 1 downto 0);
  signal max_signed   : signed(RESOLUTION - 1 downto 0);
  signal relu_out     : std_logic_vector(RESOLUTION - 1 downto 0);
begin
  value_signed <= signed(Value);
  max_signed   <= to_signed(MAX_VALUE, RESOLUTION);

  -- ReLU as its own block: Y(i) = X(i) and not sign. One LUT per bit.
  relu_block : entity work.ReLU
    generic map (
      Resolution => RESOLUTION
    )
    port map (
      X => Value,
      Y => relu_out
    );

  -- A process rather than a selected signal assignment: the two conditions
  -- are not a clean enumeration (a negative value must not also be tested
  -- against +MAX), and VHDL will not parse a multi-line selected assignment
  -- with an aggregate choice.
  process(Value, relu_out)
    variable is_negative : boolean;
    variable is_too_big  : boolean;
  begin
    is_negative := (Value(RESOLUTION - 1) = '1');
    -- If negative it is already below +MAX, so only test magnitude when the
    -- sign bit is clear. That lets one unsigned comparator do the whole job.
    is_too_big  := (not is_negative) and
                  (unsigned(Value) > unsigned(std_logic_vector(max_signed)));

    if is_negative then
      Y <= relu_out;                              -- 0, from the ReLU block
    elsif is_too_big then
      Y <= std_logic_vector(max_signed);          -- saturate
    else
      Y <= Value;
    end if;
  end process;
end rtl;
