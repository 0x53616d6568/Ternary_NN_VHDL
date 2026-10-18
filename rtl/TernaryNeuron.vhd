-- TernaryNeuron: one hidden unit.
--
-- The complete per-unit datapath, assembled from four smaller blocks:
--
--     W (2 bits) ---> TernaryApply ---> Accumulator ---> Saturate ---> Hidden
--     X (signed) --^                                    ^
--                                     Bias ---> Accumulator (load)
--
-- Which is the dataflow equation, structurally:
--
--     delta[h]  = T(W) * X
--     A[h][k+1] = A[h][k] + delta[h][k]
--     hidden[h] = Clamp( A[h][N_INPUTS-1], 0, HIDDEN_MAX )
--
-- One instance per hidden unit. TernaryMacEngine builds an array of these.
--
-- There is no `*` operator anywhere in this file, or in any file it
-- instantiates. That is the entire multiplier-free claim, and it is enforced
-- structurally rather than by convention: the only way to get a product here
-- is to bypass TernaryApply.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity TernaryNeuron is
  generic (
    ACC_WIDTH    : integer := 32;
    HIDDEN_WIDTH : integer := 8;
    HIDDEN_MAX   : integer := 127;
    X_WIDTH      : integer := 4
  );
  port (
    clk          : in  std_logic;
    Start        : in  std_logic;   -- load the bias, clear for a new image
    FinalCycle   : in  std_logic;   -- 1 on the last input only
    W            : in  std_logic_vector(1 downto 0);            -- 2-bit ternary
    X            : in  std_logic_vector(X_WIDTH - 1 downto 0);  -- signed rotated pixel
    Bias         : in  std_logic_vector(ACC_WIDTH - 1 downto 0);
    Hidden       : out std_logic_vector(HIDDEN_WIDTH - 1 downto 0)
  );
end TernaryNeuron;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of TernaryNeuron is
  signal delta_wide  : std_logic_vector(ACC_WIDTH - 1 downto 0);
  signal x_wide      : std_logic_vector(ACC_WIDTH - 1 downto 0);
  signal acc_value   : std_logic_vector(ACC_WIDTH - 1 downto 0);
  signal acc_sum     : std_logic_vector(ACC_WIDTH - 1 downto 0);
  signal clamped     : std_logic_vector(ACC_WIDTH - 1 downto 0);
begin
  -- Sign-extend the 4-bit rotated input to the accumulator width. This is the
  -- only place the width change happens.
  x_wide <= std_logic_vector(resize(signed(X), ACC_WIDTH));

  -- delta = T(W) * X
  apply_unit : entity work.TernaryApply
    generic map (
      RESOLUTION => ACC_WIDTH
    )
    port map (
      W     => W,
      X     => x_wide,
      Delta => delta_wide
    );

  -- A[h][k+1] = A[h][k] + delta, with the clamp on the final cycle only
  accumulator : entity work.Accumulator
    generic map (
      RESOLUTION => ACC_WIDTH
    )
    port map (
      clk        => clk,
      Load       => Start,
      LoadValue  => Bias,
      AddValue   => delta_wide,
      FinalCycle => FinalCycle,
      Clamped    => clamped,
      Reg        => acc_value
    );

  -- ReLU + saturation to [0, HIDDEN_MAX]
  saturate : entity work.Saturate
    generic map (
      RESOLUTION => ACC_WIDTH,
      MAX_VALUE  => HIDDEN_MAX
    )
    port map (
      Value => acc_sum,
      Y     => clamped
    );

  -- acc_sum is the register plus the incoming delta, i.e. the running total
  -- BEFORE the final-cycle decision. It is what Saturate must inspect.
  acc_sum <= std_logic_vector(signed(acc_value) + signed(delta_wide));

  -- Output. The register already holds the clamped value on the final cycle,
  -- so this is just a narrow to HIDDEN_WIDTH.
  Hidden <= std_logic_vector(resize(signed(acc_value), HIDDEN_WIDTH));
end rtl;
