library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Streaming Walsh-Hadamard rotation of the pixel bus.
--
-- Bonsai 2 rotates the weights with a blockwise Walsh-Hadamard transform before
-- ternarising them, and applies the matching transform to the activations at
-- run time. Here the transform is folded into a fixed permutation with fixed
-- signs, so the only variable is the pixel itself and the whole thing collapses
-- to "add the pixel or its negation into BLOCK_SIZE accumulators".
--
--   y_b[i] = sum over k of H[i][k] * x[b * BLOCK_SIZE + k]        (Sylvester H)
--
-- STORAGE. A pixel belongs to exactly one block, so only the BLOCK_SIZE
-- accumulators of the current block are ever touched. An earlier version kept
-- all 784 outputs as registers and indexed them with `block_base + i`, a
-- variable index into a 784-entry register array. That is not a memory to the
-- synthesiser: it becomes a 784-way write decoder plus a 784:1 read multiplexer
-- on every lane, several thousand LUTs on a part with 9,112. Here the eight
-- block accumulators are static indices, and a completed block is written into
-- eight small lane banks. Each bank holds one output per block (98 deep), has a
-- single write port and a single read port, and is therefore inferred as
-- distributed RAM. The arithmetic is unchanged and still bit-exact: the same
-- adds happen in the same order.
--
-- The trailing arithmetic shift restores the scale that the unnormalised
-- transform inflates by sqrt(BLOCK_SIZE). SHIFT = log2(BLOCK_SIZE) / 2. Because
-- it is a shift of an integer it is exact, and in hardware it costs nothing:
-- the low bits are simply not routed.

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

    -- One output per cycle, in the same order the MAC engine consumes them.
    -- The engine walks 0..N_INPUTS-1, so a simple address port is enough; the
    -- old 3,136-bit parallel dump is what forced a 784:1 multiplexer.
    rd_addr   : in  std_logic_vector(9 downto 0);
    rd_data   : out std_logic_vector(OUT_WIDTH - 1 downto 0)
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

  subtype sum_word is signed(SUM_WIDTH - 1 downto 0);
  type acc_array is array (0 to BLOCK_SIZE - 1) of sum_word;

  -- Lane bank: bank(i) holds outputs i, i + BLOCK_SIZE, i + 2*BLOCK_SIZE, ...
  constant BANK_DEPTH : integer := N_INPUTS / BLOCK_SIZE;
  type bank_t is array (0 to BANK_DEPTH - 1) of std_logic_vector(OUT_WIDTH - 1 downto 0);
  type bank_set_t is array (0 to BLOCK_SIZE - 1) of bank_t;

  signal acc    : acc_array := (others => (others => '0'));
  signal bank   : bank_set_t;
  signal active : std_logic := '0';
  signal done   : std_logic := '0';

  signal pixel : std_logic_vector(1 downto 0) := "00";
  signal k_in_block : integer range 0 to BLOCK_SIZE - 1;
  signal block_index : integer range 0 to BANK_DEPTH - 1;
begin
  -- Zero-extended, not sign-extended: the pixel lands in the least significant
  -- bit so signed("01") = +1 and signed("00") = 0.
  pixel <= (0 => in_data, others => '0');

  -- Position of this pixel inside its block, and which block that is. Both are
  -- combinational functions of the write address; BLOCK_SIZE is a power of two,
  -- so these are bit slices after constant folding, not dividers.
  -- Explicit sensitivity list rather than process(all): ISE 14.7's XST does not
  -- accept the VHDL-2008 form, and this design has to synthesise on the
  -- Spartan-6 toolchain. Behaviour is identical.
  process(in_index)
  begin
    k_in_block  <= to_integer(unsigned(in_index)) mod BLOCK_SIZE;
    block_index <= to_integer(unsigned(in_index)) / BLOCK_SIZE;
  end process;

  process(clk)
    variable contribution : sum_word;
    variable updated      : sum_word;
  begin
    if rising_edge(clk) then
      done <= '0';

      if start = '1' then
        for index in 0 to BLOCK_SIZE - 1 loop
          acc(index) <= (others => '0');
        end loop;
        active <= '1';
      end if;

      -- The first pixel arrives on the same cycle as start, because start is
      -- derived from the write to address 0. Consuming it in the same branch
      -- stops that pixel being dropped; treating it as an ordinary pixel of an
      -- already-active transform would lose one of the 784 inputs.
      if (start = '1' or active = '1') and in_valid = '1' then
        for i in 0 to BLOCK_SIZE - 1 loop
          if H_SIGN(i, k_in_block) = '0' then
            contribution := resize(signed(pixel), SUM_WIDTH);
          else
            contribution := -resize(signed(pixel), SUM_WIDTH);
          end if;

          -- At k = 0 the block starts from this pixel alone, which is why the
          -- accumulator is not cleared in a separate pass: a clear followed by
          -- an add in the same cycle would need the old value anyway.
          if k_in_block = 0 then
            updated := contribution;
          else
            updated := acc(i) + contribution;
          end if;
          acc(i) <= updated;

          -- Last pixel of the block: the sum is final, so scale it once and
          -- commit it to its lane bank. Shifting here rather than on every
          -- read is the same integer operation the old combinational output
          -- stage performed.
          if k_in_block = BLOCK_SIZE - 1 then
            bank(i)(block_index) <=
              std_logic_vector(resize(shift_right(updated, SHIFT), OUT_WIDTH));
          end if;
        end loop;

        if to_integer(unsigned(in_index)) = N_INPUTS - 1 then
          active <= '0';
          done   <= '1';
        end if;
      end if;
    end if;
  end process;

  out_valid <= done;

  -- Read port. Lanes are interleaved, so output address a lives in bank
  -- (a mod BLOCK_SIZE) at depth (a / BLOCK_SIZE). The bank select is a small
  -- multiplexer over BLOCK_SIZE memories, not a 784-way register mux.
  --
  -- `bank` must be in the sensitivity list as well as `rd_addr`. Without it the
  -- process only re-runs when the address changes, so a read of the address the
  -- port already holds returns whatever was in the bank the last time the
  -- address moved. The symptom is a single stale entry: address 0 is X on the
  -- first image, because rd_addr is 0 from time zero and the bank is written
  -- while the address never changes. Synthesis ignores sensitivity lists, so
  -- the hardware is the same combinational read either way; this is purely a
  -- simulation-correctness fix.
  process(rd_addr, bank)
    variable lane : integer range 0 to BLOCK_SIZE - 1;
    variable depth : integer range 0 to BANK_DEPTH - 1;
  begin
    lane  := to_integer(unsigned(rd_addr)) mod BLOCK_SIZE;
    depth := to_integer(unsigned(rd_addr)) / BLOCK_SIZE;
    rd_data <= bank(lane)(depth);
  end process;
end rtl;
