from .Basis import Basis
import numpy as np
import pathlib as pl

class ChiralPT(Basis):
    def __init__(self, L=10, N=4, **kwargs):
        super().__init__(L=L, N=N, spinrestricted=False, **kwargs)
        self.load_spins()
        self.is_AS_ = True

    def _spin_format(self, string):
        return [int(x) for x in string.strip().split()]

    def _matrix_elements_format(self, string):
        string = string.strip().split()
        return [int(x)-1 for x in string[:-1]] + [float(string[-1])]

    def load_spins(self):
        self.qnums = {}

        filename = "nucleispnumbers.dat"
        path = pl.Path(__file__)

        path = path.parent / pl.Path(f"sets/{filename}")

        with open(path, "r") as file:
            for line in map(self._spin_format, file):
                i, n, l, j, mj, tz = line
                nums = {}
                nums["n"] = n
                nums["l"] = l
                nums["j"] = j
                nums["mj"] = mj
                nums["tz"] = tz

                i = i-1

                if i >= self.L_:
                    break

                self.qnums[i] = nums

    def calculate_OB(self):
        for i, nums in enumerate(self.qnums):
            n, l = self.qnums[i]["n"], self.qnums[i]["l"]
            self.h_[i,i] = 16*(2*n+l+1.5) # MeV

    def load_elements(self, filename):
        with open(filename, "r") as file:
            x = 1
            for line in map(self._matrix_elements_format, file):
                i, j, k, l, elm = line

                if(all([x < self.L_ for x in [i,j,k,l]])):
                    self.v_[i,j,k,l] = elm

    def spin_projection_expval(self, C=None):
        N, L = self.N_, self.L_
        expval = 0
        
        if C is None:
            for i in range(N):
                expval += self.qnums[i]["mj"]
        else:
            for i in range(N):
                for p in range(L):
                    expval += self.C[p,i]**2 * self.qnums[p]["mj"]

        return expval
    
    def make_AS(self):
        raise UserWarning("Matrix elements are already antisymmetric!")

if __name__ == "__main__":
    pt = ChiralPT(L=10).load_TB("nucleitwobody.dat")
    pt.calculate_OB()
    print(pt.h)