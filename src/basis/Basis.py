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

        self.h_ = np.zeros((L, L), dtype=float)
        self.v_ = np.zeros((L, L, L, L), dtype=float)
        
        self.occ_ = slice(0,N)
        self.vir_ = slice(N,L)

        self.large_loops = {
            "make_AS": self.make_AS_python,
        }

        default_args = {
            "fast": False
        }

        default_args.update(kwargs)

        if default_args["fast"]:
            self.setup_fast_functions()

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

    def setup_fast_functions(self):
        try:
            import cpputils
        except:
            raise ImportError("cpputils is not installed!")

        print("Going fast...")
        self.large_loops = {
            "make_AS": cpputils.make_AS
        }
    def make_AS_python(self, v, v_AS):
        L = self.L_
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    for l in range(L):
                        elm = v[i,j,k,l] - v[i,j,l,k]
                        v_AS[i,j,k,l] = elm
                        v_AS[i,j,l,k] = -elm
                        v_AS[j,i,k,l] = -elm
                        v_AS[j,i,l,k] = elm

    def restricted_to_unrestricted(self):
        L_new = 2*self.L_
        h_new = np.zeros(shape=(L_new, L_new))

        for i in range(self.L_):
            for j in range(self.L_):
                h_new[2*i,2*j] = self.h[i,j]
                h_new[2*i+1,2*j+1] = self.h[i,j]

        h_new2 = np.kron(self.h, np.eye(2))
        np.testing.assert_allclose(h_new, h_new2)

        v_new = np.zeros(shape=(L_new, L_new, L_new, L_new))
        for i in range(self.L_):
            for j in range(self.L_):
                for k in range(self.L_):
                    for l in range(self.L_):
                        v_new[2*i, 2*j, 2*k, 2*l] = self.v[i,j,k,l]
                        v_new[2*i+1, 2*j, 2*k+1, 2*l] = self.v[i,j,k,l]
                        v_new[2*i, 2*j+1, 2*k, 2*l+1] = self.v[i,j,k,l]
                        v_new[2*i+1, 2*j+1, 2*k+1, 2*l+1] = self.v[i,j,k,l]
        
        extend = np.einsum("pr, qs -> pqrs", np.eye(2), np.eye(2))
        v_new2 = np.kron(self.v, extend)
        np.testing.assert_allclose(v_new, v_new2)
        
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

        self.large_loops["make_AS"](v, v_AS)

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

    def save_TB(self, filename):
        pass

    def load_TB(self, filename):
        path = pl.Path(__file__).parent / pl.Path("sets")
        path.mkdir(exist_ok=True)
        path /= pl.Path(filename)

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
                    