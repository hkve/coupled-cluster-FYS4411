import numpy as np
import textwrap
from .CCbase import CCbase

class CCD(CCbase):
    def __init__(self, basis, **kwargs):
        assert not basis.spinrestricted_, f"Unrestricted CCD requires general matrix elements"
        assert basis.is_AS_, f"Unrestricted CCD requires antisymmetric matrix elements" 
        super().__init__(basis, **kwargs)


    def next_iteration(self, t):
        f = self.f
        v = self.basis.v_
        occ, vir = self.basis.occ_, self.basis.vir_

        res = np.zeros_like(t)

        res += v[vir, vir, occ, occ] # v_abij

        # tp = np.einsum("bc,acij->abij", f[vir, vir], t)
        # res += (tp - tp.transpose(1,0,2,3))

        # tp = np.einsum("kj,abik->abij", f[occ, occ], t)
        # res -= (tp - tp.transpose(0,1,3,2))

        # Two first sums, over cd and kl
        res += 0.5*np.einsum("abcd,cdij->abij", v[vir, vir, vir, vir], t, optimize=True)
        res += 0.5*np.einsum("klij,abkl->abij", v[occ, occ, occ, occ], t, optimize=True)

        # First permutation term, P(ij|ab), over kc
        tp = np.einsum("kbcj,acik->abij", v[occ, vir, vir, occ], t, optimize=True)
        res += (tp - tp.transpose(1,0,2,3) - tp.transpose(0,1,3,2) + tp.transpose(1,0,3,2))

        # First double t sum, over klcd
        res += 0.25*np.einsum("klcd,cdij,abkl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        
        # First P(ij) permutation, double t sum over klcd
        tp = np.einsum("klcd,acik,bdjl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res += (tp - tp.transpose(0,1,3,2))

        # second P(ij) permutation, double t sum over klcd
        tp = np.einsum("klcd,dcik,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 0.5*(tp - tp.transpose(0,1,3,2))

        # Only P(ab) term, double t sum over klcd
        tp = np.einsum("klcd,aclk,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 0.5*(tp - tp.transpose(1,0,2,3))
        
        return res


    def evalute_energy_iteration(self, t, v, occ, vir):
        return 0.25*np.einsum("ijab,abij", v[occ, occ, vir, vir], t)


    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinv):
        if iters == 0 and p == 0:
            E_mp2 = 0.25 * np.einsum("ijab,abij", v[occ, occ, vir, vir]**2, epsinv)
            assert np.isclose(E_CCD0, E_mp2), f"First iteration of CCD did not reproduce MP2 energy, {E_ccd0 = }, {E_mp2 =}"


if __name__ == '__main__':
    from ..basis.Hydrogen import Hydrogen
    from ..HF.HF import HF
    N = 2
    basis = Hydrogen(L=6, N=N, Z=N, spinrestricted=False).load_TB("hydrogen.txt")
    basis.calculate_OB()
    
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