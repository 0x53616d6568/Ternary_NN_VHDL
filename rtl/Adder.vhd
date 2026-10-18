library ieee;
use ieee.std_logic_1164.all;

entity Adder is
  generic (
    Resolution : integer := 8
  );
  port (
    A    : in  std_logic_vector(Resolution - 1 downto 0);
    B    : in  std_logic_vector(Resolution - 1 downto 0);
    Cin  : in  std_logic;
    S    : out std_logic_vector(Resolution - 1 downto 0);
    Cout : out std_logic
  );
end Adder;

library ieee;
use ieee.std_logic_1164.all;

architecture logic of Adder is
  signal carry : std_logic_vector(Resolution downto 0);
  signal prop  : std_logic_vector(Resolution - 1 downto 0);
begin
  carry(0) <= Cin;
  bits : for i in 0 to Resolution - 1 generate
    prop(i) <= A(i) xor B(i);
    S(i) <= prop(i) xor carry(i);
    carry(i + 1) <= (A(i) and B(i)) or (prop(i) and carry(i));
  end generate;
  Cout <= carry(Resolution);
end logic;
