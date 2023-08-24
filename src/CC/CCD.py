import numpy as np
from .CCbase import CCbase
from .rhs.t_CCD import amplitudes_ccd
from .rhs.t_RCCD import amplitudes_ccd_restricted

class CCD(CCbase):
    def __init__(self, basis, **kwargs):
        assert not basis.spinrestricted_, f"Unrestricted CCD requires general matrix elements"
        assert basis.is_AS_, f"Unrestricted CCD requires antisymmetric matrix elements" 
        super().__init__(basis, **kwargs)
        
        self._orders = ["D"]

        self._f = basis.h + np.einsum("piqi->pq", basis.v[:, basis.occ_, :, basis.occ_])
        
        occ, vir = basis.occ_, basis.vir_
        self.f_pp_o = self._f[vir,vir].copy()
        self.f_hh_o = self._f[occ,occ].copy()
        np.fill_diagonal(self.f_pp_o, 0)
        np.fill_diagonal(self.f_hh_o, 0)


    def next_iteration(self, t_amplitudes, epsinvs):
        v = self._basis.v_
        occ, vir = self._basis.occ_, self._basis.vir_

        f_pp_o = self.f_pp_o
        f_hh_o = self.f_hh_o

        t2, epsinv2 = t_amplitudes["D"], epsinvs["D"]
        t2_next = amplitudes_ccd(t2, v, f_pp_o, f_hh_o, vir, occ)

        return {"D": t2_next*epsinv2}

    def evalute_energy_iteration(self, t_amplitudes, v, occ, vir):
        t2 = t_amplitudes["D"]
        return 0.25*np.einsum("ijab,abij", v[occ, occ, vir, vir], t2)


    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinvs):
        if iters == 0 and p == 0:
            epsinv = epsinvs["D"]
            E_mp2 = 0.25 * np.einsum("ijab,abij", v[occ, occ, vir, vir]**2, epsinv)
            assert np.isclose(E_CCD0, E_mp2), f"First iteration of CCD did not reproduce MP2 energy, {E_CCD0 = }, {E_mp2 =}"

    def check_amplitude_symmetry(self, t):
        np.testing.assert_almost_equal(t, -t.transpose(0,1,3,2), decimal=8)

class RCCD(CCbase):
    def __init__(self, basis, **kwargs):
        assert basis.spinrestricted_, f"Restricted CCD requires restricted matrix elements"
        assert not basis.is_AS_, f"Restricted CCD can not use antisymmetric matrix elements" 
        super().__init__(basis, **kwargs)

        self._orders = ["D"]

        D = np.einsum("piqi->pq", basis.v[:, basis.occ_, :, basis.occ_], optimize=True)
        E = np.einsum("piiq->pq", basis.v[:, basis.occ_, basis.occ_, :], optimize=True)
        self._f = basis.h + 2*D - E

        occ, vir = basis.occ_, basis.vir_
        self.f_pp_o = self._f[vir,vir].copy()
        self.f_hh_o = self._f[occ,occ].copy()
        np.fill_diagonal(self.f_pp_o, 0)
        np.fill_diagonal(self.f_hh_o, 0)

        
    def evalute_energy_iteration(self, t_amplitudes, v, occ, vir):
        t2 = t_amplitudes["D"]

        D = np.einsum("ijab,abij", v[occ, occ, vir, vir], t2, optimize=True)
        E = np.einsum("ijba,abij", v[occ, occ, vir, vir], t2, optimize=True)
        return 2*D - E

    def next_iteration(self, t_amplitudes, epsinvs):
        v = self._basis.v_

        occ, vir = self._basis.occ_, self._basis.vir_
        f_pp_o = self.f_pp_o
        f_hh_o = self.f_hh_o

        t2, epsinv2 = t_amplitudes["D"], epsinvs["D"]

        t2_next = amplitudes_ccd_restricted(t2, v, f_pp_o, f_hh_o, vir, occ)

        return {"D": t2_next*epsinv2}


    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinvs):
        if iters == 0 and p == 0:
            epsinv = epsinvs["D"]
            D = np.einsum("ijab,abij,abij", v[occ,occ,vir,vir], v[vir,vir,occ,occ], epsinv)
            E = np.einsum("ijba,abij,abij", v[occ,occ,vir,vir], v[vir,vir,occ,occ], epsinv)
            E_mp2 = 2*D-E
            assert np.isclose(E_CCD0, E_mp2), f"First iteration of CCD did not reproduce MP2 energy, {E_CCD0 = }, {E_mp2 =}"

    def check_amplitude_symmetry(self, t):
        np.testing.assert_almost_equal(t, t.transpose(1,0,3,2), decimal=8, verbose=True)


if __name__ == '__main__':
    from ..basis.HarmonicsOscillator import HarmonicsOscillator
    from ..HF.HF import HF, RHF
    N = 2
    basis = HarmonicsOscillator(L=72, N=N, Z=N, spinrestricted=False)
    basis.calculate_OB()
    basis.calculate_TB()
    
    hf = HF(basis)
    hf.run()
    Eref = hf.evaluate_energy()
    basis = hf.perform_basis_change(basis)

    ccd = CCD(basis)
    ccd.run()

    dE_ccd = ccd.deltaE
    print(Eref)
    print(dE_ccd)
    print(Eref + dE_ccd)
    print(ccd)