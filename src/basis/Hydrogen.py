from .Basis import Basis
import numpy as np
import re
from sympy import sympify

class Hydrogen(Basis):
    def __init__(self, L=6, N=2, spinrestricted=True, Z=2, **kwargs):
        super().__init__(L=L, N=N, spinrestricted=spinrestricted, **kwargs)
        self.shell_numbers_ = np.array([1,2,3])
        self.degeneracies_ = 2*self.shell_numbers_
        self.cummulative_Ns_ = np.cumsum(self.degeneracies_)
        self.Z_ = Z
        assert self.degeneracy_*self.L_ in self.cummulative_Ns_, \
                f"{self.L_ = } does not give a closed shell, must be in {self.cummulative_Ns_//self.degeneracies_}"

    def calculate_OB(self):
        if self.spinrestricted_:
            for i in range(self.L_):
                self.h_[i,i] = -self.Z_**2 / (2*(i+1)**2)
        else:
            for i in range(self.L_//2):
                self.h_[2*i,2*i] = -self.Z_**2 / (2*(i+1)**2)
                self.h_[2*i+1,2*i+1] = -self.Z_**2 / (2*(i+1)**2)
    
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
                
                v_elm = float(elm.subs("Z", self.Z_))
                if self.spinrestricted_:
                    self.v_[i,j,k,l] = v_elm
                else:
                    self.fill_with_spin(v_elm, i, j, k, l)

        if not self.spinrestricted_:
            self.v_ = np.nan_to_num(self.v)
            self.make_AS()