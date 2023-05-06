from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..CC.CCD import RCCD, CCD

import unittest
import numpy as np

class TestCCD(unittest.TestCase):
    ho = HarmonicsOscillator(L=12, N=2, spinrestricted=False)
    ho_spinrestricted = HarmonicsOscillator(L=12, N=2, spinrestricted=True)

    ho.calculate_OB()
    ho.calculate_TB()

    ho_spinrestricted.calculate_OB()
    ho_spinrestricted.calculate_TB()

    def test_CCD_energies(self):
        ho, ho_spinrestricted = self.ho, self.ho_spinrestricted

        ccd = CCD(ho)
        ccd.run(p=0.3)
        Eccd = ccd.evaluate_energy()
        
        rccd = RCCD(ho_spinrestricted)
        rccd.run(p=0.3)
        Erccd = rccd.evaluate_energy()

        tol = 1e-8

        self.assertTrue(
            abs(Erccd - Erccd) < tol,
            "Differing energies for CCD and RCCD using the Harmonic Oscillator"
        )