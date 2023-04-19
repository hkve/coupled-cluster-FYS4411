import numpy as np
import pathlib as pl
from abc import ABC, abstractmethod

class SpinRestrictedBasis(ABC):
    @abstractmethod
    def __init__(self, L, N, **kwargs):
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