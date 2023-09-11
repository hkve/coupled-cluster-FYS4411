import numpy as np
import pathlib as pl
from abc import ABC, abstractmethod

class Basis(ABC):
    @abstractmethod
    def __init__(self, L, N, spinrestricted, **kwargs):
        self.spinrestricted_ = spinrestricted
        self.degeneracy_ = 1
        self.is_AS_ = False

        if spinrestricted:
            assert not N%2, f"{N = } must be even when using spinrestricted"
            assert not L%2, f"{L = } must be even when using spinrestricted"
            L //= 2
            N //= 2
            self.degeneracy_ = 2


        assert N <= L, f"Cannot have more particles {N =} than basis functions {L = }"
        self.L_ = L
        self.N_ = N

        self.s_ = np.zeros((L, L), dtype=float)
        self.h_ = np.zeros((L, L), dtype=float)
        self.v_ = np.zeros((L, L, L, L), dtype=float)
        
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


    def restricted_to_unrestricted(self):
        s_new = np.kron(self.s_, np.eye(2))
        h_new = np.kron(self.h, np.eye(2))
        
        extend = np.einsum("pr, qs -> pqrs", np.eye(2), np.eye(2))
        v_new = np.kron(self.v, extend)
        
        self.s_ = s_new
        self.h_ = h_new
        self.v_ = v_new


        self.L_ *= 2
        self.N_ *= 2
        self.occ_ = slice(0,self.N_)
        self.vir_ = slice(self.N_,self.L_)
        self.spinrestricted_ = False
        self.degeneracy_ = 1

        self.make_AS()

    def make_AS(self):
        assert not self.spinrestricted_, f"To use antisymmetric matrix elements, the basis can not be spinrestricted."
        L = self.L_
        v_AS = np.zeros_like(self.v)
        v = self.v

        v_AS = v - v.transpose(0,1,3,2)
        
        self.v_ = v_AS
        self.is_AS_ = True
        
        return self

    def fill_with_spin(self, v_elm, i, j, k, l):
        i, j, k, l = 2*i, 2*j, 2*k, 2*l
        
        self.v[i, j, k, l] = v_elm
        self.v[i+1, j, k+1, l] = v_elm
        self.v[i, j+1, k, l+1] = v_elm
        self.v[i+1, j+1, k+1, l+1] = v_elm

    def load_elements(self, filename):
        pass

    def save_elements(self, filename):
        pass

    def get_setspath_filename(self, filename):
        path = pl.Path(__file__).parent / pl.Path("sets")
        path.mkdir(exist_ok=True)
        path /= pl.Path(filename)

        return path
    

    def save_TB(self, filename):
        path = self.get_setspath_filename(filename)

        assert path.parent.exists(), f"No file at {str(path)}"
        self.save_elements(path)

    
    def load_TB(self, filename):
        path = self.get_setspath_filename(filename)

        assert path.exists(), f"No file at {str(path)}"
        self.load_elements(path)

        return self
    
    def evaluate_energy(self):
        occ = self.occ_
        if self.spinrestricted_:
            return 2*self.h[occ,occ].trace() + \
                   2*np.einsum("ijij", self.v[occ,occ,occ,occ]) \
                   - np.einsum("ijji", self.v[occ,occ,occ,occ])
        else:
            return self.h[occ,occ].trace()\
                   + 0.5*np.einsum("ijij", self.v[occ,occ,occ,occ]) \
                    