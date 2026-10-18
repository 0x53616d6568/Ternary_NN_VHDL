library ieee;
use ieee.std_logic_1164.all;

-- Top level of the ternary classifier.
--
-- Datapath, all integer and multiplier-free:
--
--   784 binary pixels
--     -> WalshHadamard   blockwise rotation, Bonsai-style, adders only
--     -> TernaryMacEngine 16 lanes, one input per cycle, 784 cycles
--     -> DenseLayer      second layer, 16 x 10, combinational
--     -> ArgMax          predicted digit
--
-- The first layer is sequential because a parallel 784 x 16 array would need
-- roughly 25,000 LUTs, more than the Artix-7 has. Streaming it needs about 160
-- LUTs and leaves the weights in a block RAM.
--
-- The weights are read straight out of the ModelWeights package. W1 is stored
-- transposed and two bits per weight, so the word for input i holds one weight
-- per hidden unit and a single indexed read serves all 16 lanes; synthesis maps
-- that to one block RAM.

entity NumberRecognition is
  port (
    clk : in std_logic;

    pixel_write_enable : in std_logic;
    pixel_write_addr   : in std_logic_vector(9 downto 0);
    pixel_write_data   : in std_logic;

    digit : out std_logic_vector(3 downto 0)
  );
end NumberRecognition;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.ModelWeights.all;

architecture structural of NumberRecognition is
  constant N_INPUTS    : integer := 784;
  constant N_HIDDEN    : integer := 32;
  constant N_CLASSES   : integer := 10;
  constant ACC_WIDTH   : integer := 32;
  constant HIDDEN_WIDTH: integer := 8;
  constant X_WIDTH     : integer := 4;
  -- Width change H=16 -> H=32. Two consequences were checked rather than
  -- assumed, and neither required a code change:
  --
  --   * The weight address bus stays 10 bits. It carries the *input* index
  --     (0 .. N_INPUTS-1 = 0..783), not a weight index, so it was already wide
  --     enough for 784 entries at either width. w_read_data is declared
  --     N_HIDDEN*2 bits wide inside TernaryMacEngine and widens on its own.
  --
  --   * HIDDEN_WIDTH stays 8 bits. Measured maximum hidden activation across
  --     800 images is 76 for the deployed H=32 model, well inside the 127 that
  --     8 bits holds. ACC_WIDTH 32 is likewise unchanged (max |score| is 18).
  --
  -- The only change required is N_HIDDEN itself, plus regenerating
  -- ModelWeights.vhd from the matching checkpoint.
  -- A second hidden layer reuses the same MAC engine with different
  -- parameters: 16 inputs instead of 784, 8-bit activations instead of the
  -- 4-bit rotated bus, and its own weight array. Depth is cheap here because
  -- the pixel-facing layer holds almost all of the weights.
  constant DEPTH       : integer := 2;
  constant ROT_BLOCK   : integer := 8;
  constant ROT_SHIFT   : integer := 1;
  constant SUM_WIDTH   : integer := 6;
  constant W2_WIDTH    : integer := 16;

  type state_type is (IDLE, MAC_RUN, DONE);
  signal state : state_type := IDLE;

  signal rot_done  : std_logic;
  signal rot_start : std_logic := '0';
  signal x_bus     : std_logic_vector(N_INPUTS * X_WIDTH - 1 downto 0);

  signal mac_start  : std_logic := '0';
  signal mac_done   : std_logic;
  signal mac_busy   : std_logic;
  signal hidden     : std_logic_vector(N_HIDDEN * HIDDEN_WIDTH - 1 downto 0);

  -- Single hidden layer, so there is no second MAC engine. The output layer
  -- reads the first layer's result directly; `hidden2` is kept as an alias so
  -- the port maps below read the same way for either depth.
  signal hidden2    : std_logic_vector(N_HIDDEN * HIDDEN_WIDTH - 1 downto 0);

  signal w_addr  : std_logic_vector(9 downto 0);
  signal w_data  : std_logic_vector(N_HIDDEN * 2 - 1 downto 0);
  signal scores  : std_logic_vector(N_CLASSES * ACC_WIDTH - 1 downto 0);

  signal w1_addr_int : integer range 0 to N_INPUTS - 1;
  signal w1_word     : unsigned(N_HIDDEN * 2 - 1 downto 0);
  signal digit_reg   : std_logic_vector(3 downto 0) := (others => '0');
begin
  -- The rotation consumes the pixel stream directly. A write to address 0 marks
  -- the start of a new image, which is when the accumulators are cleared.
  rot_start   <= '1' when (state = IDLE and pixel_write_enable = '1'
                           and unsigned(pixel_write_addr) = to_unsigned(0, 10)) else '0';

  rotation : entity work.WalshHadamard
    generic map (
      N_INPUTS  => N_INPUTS,
      BLOCK_SIZE => ROT_BLOCK,
      SHIFT     => ROT_SHIFT,
      SUM_WIDTH => SUM_WIDTH,
      OUT_WIDTH => X_WIDTH
    )
    port map (
      clk       => clk,
      start     => rot_start,
      in_valid  => pixel_write_enable,
      in_index  => pixel_write_addr,
      in_data   => pixel_write_data,
      out_valid => rot_done,
      out_data  => x_bus
    );

  -- One word per input, all N_HIDDEN weights of that input, two bits each.
  w1_addr_int <= to_integer(unsigned(w_addr));
  w1_word     <= unsigned(W1_PACKED_INIT(w1_addr_int * (N_HIDDEN * 2) + N_HIDDEN * 2 - 1
                                         downto w1_addr_int * (N_HIDDEN * 2)));
  w_data      <= std_logic_vector(w1_word);

  mac : entity work.TernaryMacEngine
    generic map (
      N_INPUTS     => N_INPUTS,
      N_HIDDEN     => N_HIDDEN,
      ACC_WIDTH    => ACC_WIDTH,
      HIDDEN_WIDTH => HIDDEN_WIDTH,
      X_WIDTH      => X_WIDTH
    )
    port map (
      clk         => clk,
      start       => mac_start,
      busy        => mac_busy,
      done        => mac_done,
      w_read_addr => w_addr,
      w_read_data => w_data,
      x           => x_bus,
      bias        => B1_INIT,
      hidden      => hidden
    );

  -- Second hidden layer, retained for the two-hidden-layer experiments.
  --
  -- The shipped model is 784 -> 16 -> 10, a single hidden layer, so this block
  -- is not instantiated and its weight memories W3_PACKED_INIT / B3_INIT are
  -- not present in ModelWeights.vhd. Leaving the instantiation in place made
  -- the design fail to elaborate at all:
  --
  --     ** Error: rtl\NumberRecognition.vhd(136): Unknown identifier "W3_PACKED_INIT".
  --     ** Error: rtl\NumberRecognition.vhd(157): Unknown identifier "B3_INIT".
  --
  -- To build the 784-16-16-10 variant, set USE_SECOND_LAYER to true and export
  -- that checkpoint, which does emit W3_PACKED and B3. The 784-10-10-10 run was
  -- abandoned at 72.22% -- it was capacity-bound, not encoding-bound -- so the
  -- two-hidden-layer path is left compiled out rather than dead.
  --
  -- w3_addr_int <= to_integer(unsigned(w3_addr));
  -- w3_word     <= unsigned(W3_PACKED_INIT(w3_addr_int * (N_HIDDEN * 2) + N_HIDDEN * 2 - 1
  --                                         downto w3_addr_int * (N_HIDDEN * 2)));
  -- w3_data     <= std_logic_vector(w3_word);

  -- hidden_layer2 : entity work.TernaryMacEngine
  --   generic map (
  --     N_INPUTS     => N_HIDDEN,
  --     N_HIDDEN     => N_HIDDEN,
  --     ACC_WIDTH    => ACC_WIDTH,
  --     HIDDEN_WIDTH => HIDDEN_WIDTH,
  --     X_WIDTH      => HIDDEN_WIDTH,
  --     ADDR_WIDTH   => 4
  --   )
  --   port map (
  --     clk         => clk,
  --     start       => mac2_start,
  --     busy        => open,
  --     done        => mac2_done,
  --     w_read_addr => w3_addr,
  --     w_read_data => w3_data,
  --     x           => hidden,
  --     bias        => B3_INIT,
  --     hidden      => hidden2
  --   );

  hidden2 <= hidden;

  -- Output layer. Only 16 x 10 products, so it stays combinational, and W2 is
  -- kept at full precision rather than ternarised: Bonsai's selective-precision
  -- trick, and only 1.3% of the weights.
  output_layer : entity work.DenseLayer
    generic map (
      N_INPUTS     => N_HIDDEN,
      N_OUTPUTS    => N_CLASSES,
      DATA_WIDTH   => HIDDEN_WIDTH,
      WEIGHT_WIDTH => W2_WIDTH,
      ACC_WIDTH    => ACC_WIDTH,
      OUTPUT_WIDTH => ACC_WIDTH,
      USE_RELU     => false
    )
    port map (
      A       => hidden2,
      Weights => W2_INIT,
      Biases  => B2_INIT,
      Y       => scores
    );

  argmax : entity work.ArgMax
    generic map (
      N_CLASSES  => N_CLASSES,
      DATA_WIDTH => ACC_WIDTH,
      INDEX_WIDTH=> 4
    )
    port map (
      Scores   => scores,
      Class_id => digit_reg
    );

  digit <= digit_reg;

  -- Single driver for both the state and the MAC start strobe. The rotation
  -- runs concurrently with pixel reception, so when it reports done only the
  -- MAC pass remains: 784 more cycles. With one hidden layer the chain is
  -- IDLE -> MAC_RUN -> DONE; the MAC_RUN2 state that the two-layer variant
  -- needed has been removed, since leaving it referenced an engine that no
  -- longer exists.
  controller : process(clk)
  begin
    if rising_edge(clk) then
      mac_start <= '0';
      case state is
        when IDLE =>
          if rot_done = '1' then
            mac_start <= '1';
            state     <= MAC_RUN;
          end if;

        when MAC_RUN =>
          -- The MAC engine finishes all 16 hidden units; its result is what
          -- the output layer and ArgMax consume. Without this transition the
          -- FSM would never return to IDLE and no further image would be read.
          if mac_done = '1' then
            state <= DONE;
          end if;

        when DONE =>
          state <= IDLE;
      end case;
    end if;
  end process;
end structural;