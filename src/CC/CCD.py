import numpy as np
from .CCbase import CCbase

class CCD(CCbase):
    def __init__(self, basis, **kwargs):
        assert not basis.spinrestricted_, f"Unrestricted CCD requires general matrix elements"
        assert basis.is_AS_, f"Unrestricted CCD requires antisymmetric matrix elements" 
        super().__init__(basis, **kwargs)
        self.f = basis.h + np.einsum("piqi->pq", basis.v[:, basis.occ_, :, basis.occ_])

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
            assert np.isclose(E_CCD0, E_mp2), f"First iteration of CCD did not reproduce MP2 energy, {E_CCD0 = }, {E_mp2 =}"

    def check_amplitude_symmetry(self, t):
        np.testing.assert_almost_equal(t, -t.transpose(0,1,3,2), decimal=8)

class RCCD(CCbase):
    def __init__(self, basis, **kwargs):
        assert basis.spinrestricted_, f"Restricted CCD requires restricted matrix elements"
        assert not basis.is_AS_, f"Restricted CCD can not use antisymmetric matrix elements" 
        super().__init__(basis, **kwargs)
        D = np.einsum("piqi->pq", basis.v[:, basis.occ_, :, basis.occ_], optimize=True)
        E = np.einsum("piiq->pq", basis.v[:, basis.occ_, basis.occ_, :], optimize=True)
        self.f = basis.h + 2*D - E
        
    def evalute_energy_iteration(self, t, v, occ, vir):
        D = np.einsum("ijab,abij", v[occ, occ, vir, vir], t, optimize=True)
        E = np.einsum("ijba,abij", v[occ, occ, vir, vir], t, optimize=True)
        return 2*D - E

    def next_iteration(self, t):
        f = self.f
        v = self.basis.v_

        occ, vir = self.basis.occ_, self.basis.vir_

        res = np.zeros_like(t)
        # Here we have a long permutation term, so we collect all sums and then perform the permutation

        # Fock terms, single sum
        # res += np.einsum("bc,acij->abij", f[vir,vir], t)
        # res += np.einsum("kj,abik->abij", f[occ,occ], t)

        # virvir and occocc sums
        res += 0.5*np.einsum("abcd,cdij->abij", v[vir, vir, vir, vir], t, optimize=True)
        res += 0.5*np.einsum("klij,abkl->abij", v[occ, occ, occ, occ], t, optimize=True)

        # vir occ double sum
        res += 2*np.einsum("kbcj,acik->abij", v[occ, vir, vir, occ], t, optimize=True)
        res -= np.einsum("kbcj,acki->abij", v[occ, vir, vir, occ], t, optimize=True)
        res -= np.einsum("kbic,ackj->abij", v[occ, vir, occ, vir], t, optimize=True)
        res -= np.einsum("kbjc,acik->abij", v[occ, vir, occ, vir], t, optimize=True)
        
        # vvoo sums
        res += 0.5*np.einsum("klcd,cdij,abkl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res += 2*np.einsum("klcd,acik,dblj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 2*np.einsum("klcd,acik,dbjl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res += 0.5*np.einsum("klcd,caik,bdlj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= np.einsum("klcd,adik,cblj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res += np.einsum("klcd,adki,cblj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        
        res += 0.5*np.einsum("klcd,cbil,adkj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 2*np.einsum("klcd,cdki,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res += np.einsum("klcd,cdik,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 2*np.einsum("klcd,cakl,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res += np.einsum("klcd,ackl,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        
        res = res + res.transpose(1,0,3,2)
        # DONE WITH PERM TERM
        
        res += v[vir, vir, occ, occ] # v_abij
    
        return res
    
    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinv):
        if iters == 0 and p == 0:
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