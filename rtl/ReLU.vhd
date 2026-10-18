library ieee;
use ieee.std_logic_1164.all;

entity ReLU is
  generic (
    Resolution : integer := 8
  );
  port (
    X : in  std_logic_vector(Resolution - 1 downto 0);
    Y : out std_logic_vector(Resolution - 1 downto 0)
  );
end ReLU;

library ieee;
use ieee.std_logic_1164.all;

architecture logic of ReLU is
begin
  bits : for i in 0 to Resolution - 1 generate
    Y(i) <= X(i) and not X(Resolution - 1);
  end generate;
end logic;
