import numpy as np
from .CCbase import CCbase

class fastCCD(CCbase):
    def __init__(self, basis, **kwargs):
        assert not basis.spinrestricted_, f"Unrestricted CCD requires general matrix elements"
        assert basis.is_AS_, f"Unrestricted CCD requires antisymmetric matrix elements" 
        super().__init__(basis, **kwargs)
        self.f = basis.h + np.einsum("piqi->pq", basis.v[:, basis.occ_, :, basis.occ_])

    def evalute_energy_iteration(self, t, v, occ, vir):
        return 0.25*np.einsum("ijab,abij", v[occ, occ, vir, vir], t)

    def next_iteration(self, t):
        f = self.f
        v = self.basis.v_

        occ, vir = self.basis.occ_, self.basis.vir_

        res = np.zeros_like(t)

    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinv):
        if iters == 0 and p == 0:
            E_mp2 = 0.25 * np.einsum("ijab,abij", v[occ, occ, vir, vir]**2, epsinv)
            assert np.isclose(E_CCD0, E_mp2), f"First iteration of CCD did not reproduce MP2 energy, {E_CCD0 = }, {E_mp2 =}"

class fastRCCD(CCbase):
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

        F_pp = self.F_intermediate_pp(f,v,t,occ,vir)
        F_hh = self.F_intermediate_hh(f,v,t,occ,vir)

        W_hhhh = self.W_intermediate_hhhh(v, t, occ, vir)
        W_hpph = self.W_intermediate_hpph(v, t, occ, vir)
        exit()

    def F_intermediate_pp(self, f, v, t, occ, vir):
        res = -2*np.einsum("afmn,mnef->ae", t, v[occ,occ,vir,vir], optimize=True)
        res += np.einsum("afmn,mnfe",t,v[occ,occ,vir,vir])

        return f[vir,vir] + res

    def F_intermediate_hh(self, f, v, t, occ, vir):
        res = 2*np.einsum("efin,mnef->mi", t, v[occ,occ,vir,vir], optimize=True)
        res -= np.einsum("efin,mnfe->mi", t, v[occ,occ,vir,vir], optimize=True)

        return f[occ, occ] + res
    
    def W_intermediate_hhhh(self, v, t, occ, vir):
        res = np.einsum("efij,mnef->mnij", t, v[occ,occ,vir,vir], optimize=True)
        return v[occ,occ,occ,occ] + res

    def W_intermediate_hpph(self, v, t, occ, vir):
        res = -0.5*np.einsum("fbjn,mnef->mbej", t, v[occ, occ, vir, vir], optimize=True)
        res += np.einsum("fbnj,mnef->mbej", t, v[occ, occ, vir, vir], optimize=True)
        

        return v[occ,vir,vir,occ] + res

    def W_intermediate_hphp(self, v, t, occ, vir):
        pass

    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinv):
        if iters == 0 and p == 0:
            D = np.einsum("ijab,abij,abij", v[occ,occ,vir,vir], v[vir,vir,occ,occ], epsinv)
            E = np.einsum("ijba,abij,abij", v[occ,occ,vir,vir], v[vir,vir,occ,occ], epsinv)
            E_mp2 = 2*D-E
            assert np.isclose(E_CCD0, E_mp2), f"First iteration of CCD did not reproduce MP2 energy, {E_CCD0 = }, {E_mp2 =}"