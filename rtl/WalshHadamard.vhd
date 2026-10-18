library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Streaming Walsh-Hadamard rotation of the pixel bus.
--
-- Bonsai 2 rotates the weights with a blockwise Walsh-Hadamard transform before
-- ternarising them, and applies the matching transform to the activations at
-- run time. Here the transform is folded into a fixed permutation with fixed
-- signs, so the only variable is the pixel itself and the whole thing collapses
-- to "add the pixel or its negation into 784 accumulators".
--
--   y_b[i] = sum over k of H[i][k] * x[b * BLOCK_SIZE + k]        (Sylvester H)
--
-- Every pixel therefore updates every accumulator with a single conditional
-- add or subtract. Accumulating all N_INPUTS outputs at once needs
-- N_INPUTS small adders rather than the N_INPUTS * BLOCK_SIZE that a combinational
-- transform would need, which is what keeps this inside the LUT budget.
--
-- The trailing arithmetic shift restores the scale that the unnormalised
-- transform inflates by sqrt(BLOCK_SIZE). SHIFT = log2(BLOCK_SIZE) / 2. Because it is a
-- shift of an integer it is exact, and in hardware it costs nothing: the low
-- bits are simply not routed.

entity WalshHadamard is
  generic (
    N_INPUTS  : integer := 784;
    BLOCK_SIZE     : integer := 16;
    SHIFT     : integer := 2;
    SUM_WIDTH : integer := 6;   -- signed, holds +/- BLOCK_SIZE
    OUT_WIDTH : integer := 4    -- signed, after the shift
  );
  port (
    clk       : in  std_logic;
    start     : in  std_logic;                     -- begin a new image
    in_valid  : in  std_logic;                     -- one pixel presented
    in_index  : in  std_logic_vector(9 downto 0);
    in_data   : in  std_logic;                     -- a single binary pixel
    out_valid : out std_logic;                     -- transform complete
    out_data  : out std_logic_vector(N_INPUTS * OUT_WIDTH - 1 downto 0)
  );
end WalshHadamard;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture rtl of WalshHadamard is
  constant N_BLOCKS : integer := N_INPUTS / BLOCK_SIZE;
  constant ADDR_BITS : integer := 15;

  -- Sylvester-constructed Hadamard signs: H[i][k] = (-1) ** popcount(i and k).
  type sign_table is array (0 to BLOCK_SIZE - 1, 0 to BLOCK_SIZE - 1) of bit;
  function build_signs return sign_table is
    variable table : sign_table;
    variable i      : integer;
    variable k      : integer;
    variable overlap : integer;
    variable parity  : integer;
  begin
    for i in 0 to BLOCK_SIZE - 1 loop
      for k in 0 to BLOCK_SIZE - 1 loop
        -- Parity of the bitwise overlap decides the sign; 'and' is a logical
        -- operator in VHDL, so the bits are tested one at a time.
        parity  := 0;
        for bit_index in 0 to ADDR_BITS loop
          if (i / (2 ** bit_index)) mod 2 = 1 and (k / (2 ** bit_index)) mod 2 = 1 then
            parity := parity + 1;
          end if;
        end loop;
        if parity mod 2 = 0 then
          table(i, k) := '0';   -- +1
        else
          table(i, k) := '1';   -- -1
        end if;
      end loop;
    end loop;
    return table;
  end function build_signs;

  constant H_SIGN : sign_table := build_signs;

  type sum_array is array (0 to N_INPUTS - 1) of signed(SUM_WIDTH - 1 downto 0);
  signal sums   : sum_array := (others => (others => '0'));
  signal active : std_logic := '0';
  signal done   : std_logic := '0';

  signal pixel : std_logic_vector(1 downto 0) := "00";
  signal k_in_block : integer range 0 to BLOCK_SIZE - 1;
  signal block_base : integer range 0 to N_INPUTS - 1;
begin
  -- Zero-extended, not sign-extended: the pixel lands in the least significant
-- bit so signed("01") = +1 and signed("00") = 0.
  pixel <= (0 => in_data, others => '0');

  -- k_in_block is the pixel index modulo the block size, and block_base is the
  -- first accumulator belonging to the block that owns this pixel.
  process(all)
  begin
    k_in_block <= to_integer(unsigned(in_index)) mod BLOCK_SIZE;
    block_base <= (to_integer(unsigned(in_index)) / BLOCK_SIZE) * BLOCK_SIZE;
  end process;

  process(clk)
    variable contribution : signed(SUM_WIDTH - 1 downto 0);
  begin
    if rising_edge(clk) then
      done <= '0';

      if start = '1' then
        for index in 0 to N_INPUTS - 1 loop
          sums(index) <= (others => '0');
        end loop;
        active <= '1';

        -- The first pixel arrives on the same cycle as start, because start is
        -- derived from the write to address 0. Clearing and consuming it
        -- together stops that pixel from being dropped; treating it as an
        -- ordinary pixel would lose one of the 784 inputs.
        if in_valid = '1' then
          for i in 0 to BLOCK_SIZE - 1 loop
            if H_SIGN(i, 0) = '0' then
              sums(i) <= resize(signed(pixel), SUM_WIDTH);
            else
              sums(i) <= -resize(signed(pixel), SUM_WIDTH);
            end if;
          end loop;
        end if;

      elsif active = '1' and in_valid = '1' then
        -- Only the block that owns this pixel is updated. Accumulating every
        -- block on every pixel would mix the whole image into each output.
        for i in 0 to BLOCK_SIZE - 1 loop
          if H_SIGN(i, k_in_block) = '0' then
            contribution := resize(signed(pixel), SUM_WIDTH);
          else
            contribution := -resize(signed(pixel), SUM_WIDTH);
          end if;
          sums(block_base + i) <= sums(block_base + i) + contribution;
        end loop;

        if to_integer(unsigned(in_index)) = N_INPUTS - 1 then
          active <= '0';
          done   <= '1';
        end if;
      end if;
    end if;
  end process;

  out_valid <= done;

  process(all)
  begin
    for index in 0 to N_INPUTS - 1 loop
      -- Arithmetic shift keeps the result exact; OUT_WIDTH must fit BLOCK_SIZE/2**SHIFT.
      out_data((index + 1) * OUT_WIDTH - 1 downto index * OUT_WIDTH)
        <= std_logic_vector(resize(shift_right(sums(index), SHIFT), OUT_WIDTH));
    end loop;
  end process;
end rtl;