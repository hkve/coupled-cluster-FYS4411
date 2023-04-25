from ..basis.Hydrogen import Hydrogen
from ..HF.HF import HF, RHF

import unittest
import numpy as np

class TestHF(unittest.TestCase):
    He_spinrestricted = Hydrogen(L=6, N=2, spinrestricted=True).load_TB("hydrogen.txt")
    He = Hydrogen(L=6, N=2, spinrestricted=False).load_TB("hydrogen.txt")

    Be_spinrestricted = Hydrogen(L=6, N=4, spinrestricted=True).load_TB("hydrogen.txt")
    Be = Hydrogen(L=6, N=4, spinrestricted=False).load_TB("hydrogen.txt")

    rhf_He = RHF(He_spinrestricted)
    hf_He = HF(He)
    rhf_Be = RHF(Be_spinrestricted) 
    hf_Be = HF(Be)

    def test_HF_energies(self):
        rhf_He, hf_He = self.rhf_He, self.hf_He
        rhf_Be, hf_Be = self.rhf_Be, self.hf_Be 
        
        rhf_He.run()
        hf_He.run()
        rhf_Be.run()
        hf_Be.run()

        E_spinrestricted_He = rhf_He.evaluate_energy()
        E_He = hf_He.evaluate_energy()
        E_spinrestricted_Be = rhf_Be.evaluate_energy()
        E_Be = hf_Be.evaluate_energy()
        
        self.assertTrue(
            np.isclose(E_spinrestricted_He, E_He),
            "Error in Helium Hartree-Fuck energy"
        )

        self.assertTrue(
            np.isclose(E_spinrestricted_Be, E_Be),
            "Error in Helium Hartree-Fock energy"
        )

    def test_density_matrix_trace(self):
        expected = [2,4]
        rhfs, hfs = [self.rhf_He, self.rhf_Be], [self.hf_He, self.hf_Be]
        for rhf, hf, expect in zip(rhfs, hfs, expected):
            rhf.run()
            hf.run()

            self.assertTrue(
                np.isclose(rhf.rho_.trace(), expect),
                "Restricted Hartree-Fock trace does not conserve particles"
            )

            self.assertTrue(
                np.isclose(hf.rho_.trace(), expect),
                "Hartree-Fock trace does not conserve particles"
            )
