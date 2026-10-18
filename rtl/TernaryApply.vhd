-- Ternary weight apply:  delta = T(w) * x
--
-- The step that removes the multiplier from the datapath.
--
--   T(00) =  0      mask to zero      -- a wire
--   T(01) = +1      pass through       -- a wire
--   T(11) = -1      two's complement   -- invert + increment
--   T(10) = undefined, never emitted by the exporter
--
-- Two-bit two's complement encodes {0,+1,-1} with no decoder logic: bit 1 is
-- the sign and bit 0 is the "is it one" flag.
--
--   T(w) = w[1] ? -1 : w[0]
--
-- The -1 case is a genuine two's complement negation, not a wire. X is signed
-- and measures -2..+4 on real MNIST data, so -X = ~X + 1 requires a carry. The
-- increment is folded into the adder's carry-in in Accumulator, so no separate
-- incrementer stage is built.
--
-- This is the whole multiplier-free argument in one component. Because T is
-- restricted to {0,+1,-1}, there is no partial-product array and no Wallace
-- tree -- only a conditional invert feeding an adder.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity TernaryApply is
  generic (
    RESOLUTION : integer := 32
  );
  port (
    W     : in  std_logic_vector(1 downto 0);         -- 2-bit ternary weight
    X     : in  std_logic_vector(RESOLUTION - 1 downto 0);  -- signed operand
    Delta : out std_logic_vector(RESOLUTION - 1 downto 0)
  );
end TernaryApply;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of TernaryApply is
  signal w_is_zero : std_logic;
  signal w_is_minus : std_logic;
  signal x_signed  : signed(RESOLUTION - 1 downto 0);
  signal x_negated : signed(RESOLUTION - 1 downto 0);
begin
  -- 2'b10 would mean -2 and is never produced, so it falls through to zero.
  w_is_zero  <= not (W(1) or W(0));
  w_is_minus <= W(1);

  x_signed  <= signed(X);
  x_negated <= -x_signed;

  -- Two multiplexers. In terms of the dataflow equation delta = T(W) * X this
  -- is the T(W) factor; the multiply itself has vanished because T is ternary.
  with W select
    Delta <= (others => '0')           when "00",
            std_logic_vector(x_signed)  when "01",
            std_logic_vector(x_negated) when "11",
            (others => '0')             when others;   -- "10": unreachable
end rtl;
