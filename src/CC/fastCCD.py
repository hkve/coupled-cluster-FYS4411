import numpy as np
from .CCD import CCD, RCCD
from .rhs.t_inter_CCD import amplitudes_intermediates_ccd

import numpy as np

class fastCCD(CCD):
    def __init__(self, basis, **kwargs):
        super().__init__(basis, **kwargs)

    def next_iteration(self, t_amplitudes, epsinvs):
        f = self._f

        f_pp_o = self.f_pp_o
        f_hh_o = self.f_hh_o

        u = self._basis.v_
        o, v = self._basis.occ_, self._basis.vir_

        M = self._basis.L_ - self._basis.N_
        N = self._basis.N_
        
        t2, epsinv2 = t_amplitudes["D"], epsinvs["D"]
        t2_next = amplitudes_intermediates_ccd(t2, u, f_pp_o, f_hh_o, v, o)

        return {"D": t2_next*epsinv2}

class fastRCCD(RCCD):
    def __init__(self, basis, **kwargs):
        super().__init__(basis, **kwargs)
        
    def next_iteration(self, t):
        f = self._f
        v = self._basis.v_

        occ, vir = self._basis.occ_, self._basis.vir_

        res = np.zeros_like(t)

        # Calculate all intermediates
        F_pp = self.F_intermediate_pp(f,v,t,occ,vir)
        F_hh = self.F_intermediate_hh(f,v,t,occ,vir)

        W_hhhh = self.W_intermediate_hhhh(v, t, occ, vir)
        W_hpph = self.W_intermediate_hpph(v, t, occ, vir)
        W_hphp = self.W_intermediate_hphp(v, t, occ, vir)
        
        res += v[vir, vir, occ, occ]

        perm = np.einsum("aeij,be->abij", t, F_pp, optimize=True) - np.einsum("abim,mj->abij", t, F_hh, optimize=True)
        perm = perm + perm.transpose(1,0,3,2)

        res += perm

        res += np.einsum("abmn,mnij->abij", t, W_hhhh, optimize=True)
        res += np.einsum("efij,abef->abij", t, v[vir,vir,occ,occ], optimize=True)

        X = np.einsum("aeim,mbej->abij", t, W_hpph, optimize=True)

        perm = X - np.einsum("eaim,mbej->abij", t, W_hpph, optimize=True)
        perm = perm + perm.transpose(1,0,3,2)
        res += perm

        perm = X + np.einsum("aeim,mbje->abij", t, W_hphp, optimize=True)
        perm = perm + perm.transpose(1,0,3,2)
        res += perm

        perm = np.einsum("aemj,mbie->abij", t, W_hphp, optimize=True)
        perm = perm + perm.transpose(1,0,3,2)
        res += perm
        
        return res

    def F_intermediate_pp(self, f, v, t, occ, vir):
        res = -2*np.einsum("afmn,mnef->ae", t, v[occ,occ,vir,vir], optimize=True)
        res += np.einsum("afmn,mnfe->ae",t,v[occ,occ,vir,vir], optimize=True)

        return res# + f[vir,vir]

    def F_intermediate_hh(self, f, v, t, occ, vir):
        res = 2*np.einsum("efin,mnef->mi", t, v[occ,occ,vir,vir], optimize=True)
        res -= np.einsum("efin,mnfe->mi", t, v[occ,occ,vir,vir], optimize=True)

        return res #+ f[occ, occ]
    
    def W_intermediate_hhhh(self, v, t, occ, vir):
        res = np.einsum("efij,mnef->mnij", t, v[occ,occ,vir,vir], optimize=True)
        
        return v[occ,occ,occ,occ] + res

    def W_intermediate_hpph(self, v, t, occ, vir):
        res = -0.5*np.einsum("fbjn,mnef->mbej", t, v[occ, occ, vir, vir], optimize=True)
        res += np.einsum("fbnj,mnef->mbej", t, v[occ, occ, vir, vir], optimize=True)
        res += -0.5*np.einsum("fbnj,mnfe->mbej", t, v[occ, occ, vir, vir], optimize=True)
        return v[occ,vir,vir,occ] + res

    def W_intermediate_hphp(self, v, t, occ, vir):
        res = 0.5*np.einsum("fbjn,mnfe->mbje", t, v[occ,occ,vir,vir], optimize=True)

        return -v[occ, vir, occ, vir] + res