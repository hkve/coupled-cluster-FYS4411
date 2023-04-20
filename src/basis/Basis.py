import numpy as np
import pathlib as pl
from abc import ABC, abstractmethod

class Basis(ABC):
    @abstractmethod
    def __init__(self, L, N, spinrestricted, **kwargs):
        self.spinrestricted_ = spinrestricted
        self.degeneracy_ = 1

        if spinrestricted:
            assert not N%2, f"{N = } must be even when using spinrestricted"
            assert not L%2, f"{L = } must be even when using spinrestricted"
            L //= 2
            N //= 2
            self.degeneracy_ = 2


        assert N <= L, f"Cannot have more particles {N =} than basis functions {L = }"
        self.L_ = L
        self.N_ = N

        self.h_ = np.zeros((L, L), dtype=float)
        self.v_ = np.zeros((L, L, L, L), dtype=float)*np.nan
        
        self.occ_ = slice(0,N)
        self.vir_ = slice(N,L)

    @property
    def h(self):
        return self.h_

    @property
    def v(self):
        return self.v_

    @property
    def occ(self):
        return self.occ_
    
    @property
    def vir(self):
        return self.vir_

    def find_folder(self):
        print(__file__)

    def make_AS(self):
        assert not self.spinrestricted_, f"To use antisymmetric matrix elements, the basis can not be spinrestricted."
        L = self.L_
        v_as = np.zeros_like(self.v)
        v = self.v
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for l in range(L):
                        v_as[i,j,k,l] = v[i,j,k,l] - v[i,j,l,k] 

        self.v_ = v_as

        return self

    def fill_with_spin(self, v_elm, i, j, k, l):
        i, j, k, l = 2*i, 2*j, 2*k, 2*l
        
        self.v[i, j, k, l] = v_elm
        self.v[i+1, j, k+1, l] = v_elm
        self.v[i, j+1, k, l+1] = v_elm
        self.v[i+1, j+1, k+1, l+1] = v_elm

    def load_elements(self, filename):
        pass

    def save_TB(self, filename):
        pass

    def load_TB(self, filename):
        path = pl.Path(__file__).parent / pl.Path("sets")
        path.mkdir(exist_ok=True)
        path /= pl.Path(filename)

        assert path.exists(), f"No file at {str(path)}"
        self.load_elements(path)

        return self