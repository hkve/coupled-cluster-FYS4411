from ..basis.HarmonicsOscillator import HarmonicsOscillator

import unittest
import numpy as np

class TestHarmonicsOscillator(unittest.TestCase):
    def test_OB(self):
        omega = 0.5
        ho = HarmonicsOscillator(L=6, N=2,omega=omega)
        ho.calculate_OB()

        calculated = np.diag(ho.h)
        expected = np.array([1,2,2], dtype=float)*omega

        self.assertTrue(np.allclose(calculated, expected), f"Differing single particle energies, {expected = }, {calculated = }")

    def test_mapping(self):
        ho = HarmonicsOscillator(L = 20, N=2, spinrestricted=True)
        
        expected = [1,2,3,4]
        calculated = set()
        for n in ho.n_to_p_.keys():
            calculated.add(sum(n)+1) # Convert sum(n_x, n_y) to shell number
            
        calculated = list(calculated)
        calculated.sort()

        nshells = len(calculated)
        self.assertTrue(nshells == 4, f"Wrong number of shells, expected = {4}, calculated = {nshells}")
        if(nshells == 4):
            self.assertTrue(
                all([a==b for a, b in zip(calculated, expected)]),
                f"Wrong shell numbers in mapping dict, {expected = }, {calculated = }"
            )

    def test_A(self):
        ho = HarmonicsOscillator(L = 90, N=2, omega=2)
        
        expected = np.array([
            0.7978845608,
            0.3989422804,
            0.08143375198
        ])

        A00 = ho.A(0,0)
        A11 = ho.A(1,1)
        A21 = ho.A(3,1)
        calcualted = np.array([A00, A11, A21])

        self.assertTrue(
            np.allclose(calcualted, expected),
            f"Error in normalisation. {expected = }, {calcualted = }"
        )

    # Only run if pybind and cpputils are installed
    def test_makeAS(self):
        try:
            ho_slow = HarmonicsOscillator(L=12, N=2, spinrestricted=False, fast=False)
            ho_fast = HarmonicsOscillator(L=12, N=2, spinrestricted=False, fast=True)

            ho_slow.calculate_OB()
            ho_slow.calculate_TB()

            ho_fast.calculate_OB()
            ho_fast.calculate_TB()

            self.assertTrue(
                np.allclose(ho_slow.v, ho_fast.v),
                f"Error in AS maker"
            )
        except:
            pass