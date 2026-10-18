library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ArgMax is
  generic (
    N_CLASSES : integer := 10;
    DATA_WIDTH: integer := 32;
    INDEX_WIDTH: integer := 4
  );
  port (
    Scores   : in  std_logic_vector(N_CLASSES * DATA_WIDTH - 1 downto 0);
    Class_id : out std_logic_vector(INDEX_WIDTH - 1 downto 0)
  );
end ArgMax;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

architecture combinational of ArgMax is
begin
  process(Scores)
    variable best_score : signed(DATA_WIDTH - 1 downto 0);
    variable this_score : signed(DATA_WIDTH - 1 downto 0);
    variable best_index : unsigned(INDEX_WIDTH - 1 downto 0);
  begin
    best_score := signed(Scores(DATA_WIDTH - 1 downto 0));
    best_index := (others => '0');

    for i in 1 to N_CLASSES - 1 loop
      this_score := signed(Scores((i + 1) * DATA_WIDTH - 1 downto i * DATA_WIDTH));
      if this_score > best_score then
        best_score := this_score;
        best_index := to_unsigned(i, INDEX_WIDTH);
      end if;
    end loop;

    Class_id <= std_logic_vector(best_index);
  end process;
end combinational;
