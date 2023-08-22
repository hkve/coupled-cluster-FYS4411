import numpy as np
from .CCD import CCD, RCCD

import numpy as np

class fastCCD(CCD):
    def __init__(self, basis, **kwargs):
        super().__init__(basis, **kwargs)

    def next_iteration(self, t):
        f = self.f

        f_pp_o = self.f_pp_o
        f_hh_o = self.f_hh_o

        u = self.basis.v_
        o, v = self.basis.occ_, self.basis.vir_

        M = self.basis.L_ - self.basis.N_
        N = self.basis.N_
        
        r2 = np.zeros_like(t)

        tau0 = np.zeros((N, N, M, M))

        tau0 += np.einsum(
            "ki,abjk->ijab", f_hh_o, t, optimize=True
        )

        r2 = np.zeros((M, M, N, N))

        r2 -= np.einsum(
            "ijba->abij", tau0, optimize=True
        )

        r2 += np.einsum(
            "jiba->abij", tau0, optimize=True
        )

        tau0 = None

        tau1 = np.zeros((N, N, M, M))

        tau1 += np.einsum(
            "ac,bcij->ijab", f_pp_o, t, optimize=True
        )

        tau6 = np.zeros((N, N, M, M))

        tau6 -= 2 * np.einsum(
            "jiab->ijab", tau1, optimize=True
        )

        tau1 = None

        tau2 = np.zeros((M, M))

        tau2 -= np.einsum(
            "acji,jicb->ab", t, u[o, o, v, v], optimize=True
        )

        tau3 = np.zeros((N, N, M, M))

        tau3 += np.einsum(
            "bc,acij->ijab", tau2, t, optimize=True
        )

        tau2 = None

        tau6 += np.einsum(
            "ijab->ijab", tau3, optimize=True
        )

        tau3 = None

        tau4 = np.zeros((N, N, M, M))

        tau4 += np.einsum(
            "acik,jkbc->ijab", t, u[o, o, v, v], optimize=True
        )

        tau5 = np.zeros((N, N, M, M))

        tau5 += np.einsum(
            "acik,jkbc->ijab", t, tau4, optimize=True
        )

        tau4 = None

        tau6 += 2 * np.einsum(
            "ijba->ijab", tau5, optimize=True
        )

        tau5 = None

        r2 -= np.einsum(
            "ijab->abij", tau6, optimize=True
        ) / 2

        r2 += np.einsum(
            "ijba->abij", tau6, optimize=True
        ) / 2

        tau6 = None

        tau7 = np.zeros((N, N))

        tau7 -= np.einsum(
            "baik,kjba->ij", t, u[o, o, v, v], optimize=True
        )

        tau8 = np.zeros((N, N, M, M))

        tau8 += np.einsum(
            "jk,abik->ijab", tau7, t, optimize=True
        )

        tau7 = None

        r2 -= np.einsum(
            "ijab->abij", tau8, optimize=True
        ) / 2

        r2 += np.einsum(
            "jiab->abij", tau8, optimize=True
        ) / 2

        tau8 = None

        tau9 = np.zeros((N, N, M, M))

        tau9 += np.einsum(
            "acik,kbjc->ijab", t, u[o, v, o, v], optimize=True
        )

        r2 -= np.einsum(
            "ijab->abij", tau9, optimize=True
        )

        r2 += np.einsum(
            "ijba->abij", tau9, optimize=True
        )

        r2 += np.einsum(
            "jiab->abij", tau9, optimize=True
        )

        r2 -= np.einsum(
            "jiba->abij", tau9, optimize=True
        )

        tau9 = None

        tau10 = np.zeros((N, N, N, N))

        tau10 += 2 * np.einsum(
            "jilk->ijkl", u[o, o, o, o], optimize=True
        )

        tau10 += np.einsum(
            "balk,jiba->ijkl", t, u[o, o, v, v], optimize=True
        )

        r2 += np.einsum(
            "bakl,klji->abij", t, tau10, optimize=True
        ) / 4

        tau10 = None

        r2 += np.einsum(
            "baji->abij", u[v, v, o, o], optimize=True
        )

        r2 += np.einsum(
            "dcji,badc->abij", t, u[v, v, v, v], optimize=True
        ) / 2


        return r2

class fastRCCD(RCCD):
    def __init__(self, basis, **kwargs):
        super().__init__(basis, **kwargs)
        
    def next_iteration(self, t):
        f = self.f
        v = self.basis.v_

        occ, vir = self.basis.occ_, self.basis.vir_

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