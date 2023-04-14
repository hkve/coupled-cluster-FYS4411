from ..basis.HarmonicsOscillator import HarmonicsOscillator

import unittest
import numpy as np

class TestHarmonicsOscillator(unittest.TestCase):
    def test_OB(self):
        ho = HarmonicsOscillator(L=3)
        ho.calculate_OB()

        calculated = np.diag(ho.h)
        expected = np.array([1,2,2], dtype=float)

        self.assertTrue(np.allclose(calculated, expected), f"Differing single particle energies, {expected = }, {calculated = }")

    def test_mapping(self):
        ho = HarmonicsOscillator(L = 10)
        
        expected = [0,1,2,3,4]
        calculated = set()
        for n in ho.n_to_p.keys():
            calculated.add(sum(n))
            
        calculated = list(calculated)
        calculated.sort()

        nshells = len(calculated)
        self.assertTrue(nshells == 5, f"Wrong number of shells, expected = {5}, calculated = {nshells}")
        if(nshells == 5):
            self.assertTrue(
                all([a==b for a, b in zip(calculated, expected)]),
                f"Wrong shell numbers in mapping dict, {expected = }, {calculated = }"
            )