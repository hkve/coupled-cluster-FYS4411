import numpy as np
from .CCSD import CCSD
from .rhs.t_inter_CCSD import amplitudes_CCSD_t1_t2

import numpy as np

class fastCCSD(CCSD):
    def __init__(self, basis, **kwargs):
        super().__init__(basis, **kwargs)

        self._orders = ["S", "D"]

        self._f = basis.h + np.einsum("piqi->pq", basis.v[:, basis.occ_, :, basis.occ_])
        
        occ, vir = basis.occ_, basis.vir_


    def next_iteration(self, t_amplitudes, epsinvs):
        f = self._f
        np.fill_diagonal(f, 0)

        u = self._basis.v_
        o, v = self._basis.occ_, self._basis.vir_

        M = self._basis.L_ - self._basis.N_
        N = self._basis.N_
        
        t1, epsinv1 = t_amplitudes["S"], epsinvs["S"]
        t2, epsinv2 = t_amplitudes["D"], epsinvs["D"]

        t1_next, t2_next = amplitudes_CCSD_t1_t2(t1, t2, u, f, v, o) 
        t1_next = t1_next*epsinv1
        t2_next = t2_next*epsinv2

        return {"S": t1_next, "D": t2_next}

    def evalute_energy_iteration(self, t_amplitudes, u, o, v):
        t1, t2 = t_amplitudes["S"], t_amplitudes["D"]

        e = 0

        e += np.einsum(
            "ia,ai->", self._f[o, v], t1, optimize=True
        )

        e += np.einsum(
            "abij,ijab->", t2, u[o, o, v, v], optimize=True
        ) / 4

        e += np.einsum(
            "ai,bj,ijab->", t1, t1, u[o, o, v, v], optimize=True
        ) / 2
    
        return e

    def _check_amplitude_symmetry(self, t_amplitudes):
        t2 = t_amplitudes["D"]
        np.testing.assert_almost_equal(t2, -t2.transpose(0,1,3,2), decimal=8)