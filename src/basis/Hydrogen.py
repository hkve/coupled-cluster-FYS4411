from .Basis import SpinRestrictedBasis
import numpy as np
import re
from sympy import sympify

class Hydrogen(SpinRestrictedBasis):
    def __init__(self, L=3, Z=2, **kwargs):
        self.shell_numbers_ = [1,2,3]
        self.Z_ = Z
        assert L in self.shell_numbers_, f"{L = } not in {self.shell_numbers_}"
        super().__init__(L=L, **kwargs)

    def load_elements(self, filename):
        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                bra, = re.findall("(?<=\<)(\d\d)(?=\|)", line)
                ket, = re.findall("(?<=\|)(\d\d)(?=\>)", line)
                [i,j] = [int(c)-1 for c in bra]
                [k,l] = [int(c)-1 for c in ket]
                
                if any([idx >= self.L_ for idx in [i,j,k,l]]):
                    continue

                elm = line.split("=")[-1].strip()
                elm = re.sub("Sqrt\[", "sqrt(", elm)
                elm = re.sub("\]", ")", elm)
                elm = sympify(elm)
                
                
                self.v_[i,j,k,l] = float(elm.subs("Z", self.Z_))

    
if __name__ == "__main__":
    hy = Hydrogen(L=3).load_TB("hydrogen.txt")

    print(hy.v)