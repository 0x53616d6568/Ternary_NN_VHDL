-- Accumulator: a loadable, adding register.
--
--   Load = 1  ->  Reg <= LoadValue        (bias seed)
--   Load = 0  ->  Reg <= Reg + AddValue   (accumulate)
--
-- This is the sequential dataflow edge. The equation
--
--   A[h][0]   = B[h]
--   A[h][k+1] = A[h][k] + delta[h][k]
--
-- is implemented by driving LoadValue from the bias at k=0 and AddValue from
-- TernaryApply thereafter.
--
-- FinalCycle gates the clamp. It must be 0 for every cycle except the last of
-- an image: applying the ReLU and saturation on every pass truncates the
-- intermediate partial sums and changes the answer. Verified as a real defect
-- class during this project.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity Accumulator is
  generic (
    RESOLUTION : integer := 32
  );
  port (
    clk         : in  std_logic;
    Load        : in  std_logic;
    LoadValue   : in  std_logic_vector(RESOLUTION - 1 downto 0);
    AddValue    : in  std_logic_vector(RESOLUTION - 1 downto 0);
    FinalCycle  : in  std_logic;    -- 1 only on the last input of an image
    Clamped     : in  std_logic_vector(RESOLUTION - 1 downto 0);
    Reg         : out std_logic_vector(RESOLUTION - 1 downto 0)
  );
end Accumulator;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of Accumulator is
  -- NOTE: 'reg' is a VHDL reserved word (the 'reg attribute), so the internal
  -- signal must not be called that.
  signal acc_q    : std_logic_vector(RESOLUTION - 1 downto 0) := (others => '0');
  signal sum_bits : std_logic_vector(RESOLUTION - 1 downto 0);
begin
  -- Explicit ripple-carry adder rather than the `+` operator, so the structure
  -- matches the dataflow equation A[k+1] = A[k] + delta term by term and the
  -- design's arithmetic blocks are visible in the hierarchy.
  sum_block : entity work.Adder
    generic map (
      Resolution => RESOLUTION
    )
    port map (
      A    => acc_q,
      B    => AddValue,
      Cin  => '0',
      S    => sum_bits,
      Cout => open
    );

  process(clk)
    variable sum : signed(RESOLUTION - 1 downto 0);
  begin
    if rising_edge(clk) then
      if Load = '1' then
        -- Bias seed. Free: the register had to be loaded with something anyway.
        acc_q <= LoadValue;
      else
        if FinalCycle = '1' then
          -- Last input: apply ReLU and saturation exactly once.
          acc_q <= Clamped;
        else
          -- Running partial sum: pass through untouched.
          acc_q <= sum_bits;
        end if;
      end if;
    end if;
  end process;

  Reg <= acc_q;
end rtl;
