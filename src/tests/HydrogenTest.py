from ..basis.Hydrogen import Hydrogen

import unittest
import numpy as np

class TestHydrogen(unittest.TestCase):
    hy_spinrestricted = Hydrogen(L=6, N=2, spinrestricted=True).load_TB("hydrogen.txt")
    hy = Hydrogen(L=6, N=2, spinrestricted=False).load_TB("hydrogen.txt")

    def test_OB_energy(self):
        self.hy_spinrestricted.calculate_OB()
        self.hy.calculate_OB()

        self.assertTrue(
            np.isclose(2*np.sum(self.hy_spinrestricted.h), np.sum(self.hy.h)),
            "Different single particle energies in spin restricted and general hydrogen"
        )

    def test_v_AS(self):
        n = 10
        idc = np.random.randint(low=0, high=self.hy.L_-1, size=(n,4))
        v = self.hy.v
            
        v1, v2, v3, v4 = [], [], [], []
        for i,j, k, l in idc:
            v1.append(v[i,j,k,l])
            v2.append(-v[i,j,l,k])
            v3.append(-v[j,i,k,l])
            v4.append(v[j,i,l,k])
        
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        v4 = np.array(v4)

        vs = [v1, v2, v3, v4]

        for i in range(4):
            for j in range(i+1,4):
                self.assertTrue(
                    np.allclose(vs[i], vs[j]),
                    "Asymmetric matrix elements are not asymmetric"
                )