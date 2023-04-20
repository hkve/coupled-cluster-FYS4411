from ..basis.Hydrogen import Hydrogen
from ..HF.HF import HF, RHF

import unittest
import numpy as np

class TestHF(unittest.TestCase):
    He_spinrestricted = Hydrogen(L=6, N=2, spinrestricted=True).load_TB("hydrogen.txt")
    He = Hydrogen(L=6, N=2, spinrestricted=False).load_TB("hydrogen.txt")

    Be_spinrestricted = Hydrogen(L=6, N=4, spinrestricted=True).load_TB("hydrogen.txt")
    Be = Hydrogen(L=6, N=4, spinrestricted=False).load_TB("hydrogen.txt")

    def test_HF_energies(self):
        rhf_He, hf_He = RHF(self.He_spinrestricted), HF(self.He)
        rhf_Be, hf_Be = RHF(self.Be_spinrestricted), HF(self.Be) 
        
        rhf_He.run()
        hf_He.run()
        rhf_Be.run()
        hf_Be.run()

        E_spinrestricted_He = rhf_He.evalute_energy()
        E_He = hf_He.evalute_energy()
        E_spinrestricted_Be = rhf_Be.evalute_energy()
        E_Be = hf_Be.evalute_energy()
        
        self.assertTrue(
            np.isclose(E_spinrestricted_He, E_He),
            "Error in Helium Hartree-Fuck energy"
        )

        self.assertTrue(
            np.isclose(E_spinrestricted_Be, E_Be),
            "Error in Helium Hartree-Fock energy"
        )