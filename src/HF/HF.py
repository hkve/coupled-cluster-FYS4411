from ..basis.Hydrogen import Hydrogen
from .HFbase import HFbase
import numpy as np

class HF(HFbase):
    def __init__(self, basis):
        assert not basis.spinrestricted_, "HF requires a general spin basis"
        super().__init__(basis)

    def evaluate_energy_scheme(self):
        E_OB = np.sum(np.diag(self.rho_)*np.diag(self.basis.h))
        E_TB = 0.5*np.einsum("ag,bd,abgd", self.rho_, self.rho_, self.basis.v)
        return E_OB+E_TB        

    def evaluate_HFmat_scheme(self, rho, h, v):
        return h + np.einsum("gd,agbd->ab", rho, v)

class RHF(HFbase):
    def __init__(self, basis):
        assert basis.spinrestricted_, "HF requires a spin restricted basis"
        super().__init__(basis)

    def evaluate_energy_scheme(self):
        E_OB = np.sum(np.diag(self.rho_)*np.diag(self.basis.h))
        E_TB = 0.5*np.einsum("ag,bd,abgd", self.rho_, self.rho_, self.basis.v) \
            - 0.25*np.einsum("ag,bd,abdg", self.rho_, self.rho_, self.basis.v) 
   
        return E_OB+E_TB

    def evaluate_HFmat_scheme(self, rho, h, v):
        return h + np.einsum("gd,agbd->ab", rho, v) - 0.5*np.einsum("gd,agdb->ab", rho, v)


if __name__ == '__main__':
    hy = Hydrogen(L=6, N=4, spinrestricted=False, Z=4).load_TB("hydrogen.txt")
    hy.calculate_OB()

    hf = HF(hy)
    hf.run(maxiters=20)
    E = hf.evalute_energy()
    print(E)