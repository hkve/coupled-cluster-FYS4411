import numpy as np
from einops import repeat
import textwrap

class CCD:
    def __init__(self, basis, **kwargs):
        self.basis = basis
        self.has_run = False
        self.converged = False


    def run(self, tol=1e-5, maxiters=1000):
        basis = self.basis
        v = basis.v_
        occ, vir = self.basis.occ_, self.basis.vir_

        vir_range = basis.L_ - basis.N_
        occ_range = basis.N_

        # Change this
        t = np.zeros(shape=(vir_range, vir_range, occ_range, occ_range))

        eps_v = np.diag(self.basis.h)[occ_range:]
        eps_o = np.diag(self.basis.h)[:occ_range]

        eps = -eps_v[:,None,None,None] - eps_v[None,:,None,None] \
               +eps_o[None,None,:,None] + eps_o[None,None,None,:]

        tri_ij = np.tri(occ_range, k=-1)
        tri_ab = np.tri(vir_range, k=-1)

        tri_ij = repeat(tri_ij, "i j -> a b i j", a=vir_range, b=vir_range)
        tri_ab = repeat(tri_ab, "a b -> a b i j", i=occ_range, j=occ_range)
        tri = np.logical_and(tri_ij, tri_ab)

        iters = 0
        diff = 1
        deltaE = 1.0

        while (iters < maxiters) and (diff > tol):
            t_next = self.next_iteration(t)/eps * tri 

            deltaE_next = np.einsum("ijab,abij", v[occ, occ, vir, vir], t_next)
            diff = np.abs(deltaE_next - deltaE)
            
            if iters == 1:
                print(
                    deltaE_next
                )
                print(
                    0.25* np.einsum("ijab,abij", v[occ, occ, vir, vir]**2, 1/eps)  
                )
            self.check_convergence(diff, iters, deltaE_next)
            deltaE = deltaE_next
            t = t_next
            iters += 1

        self.has_run = True
        if(iters < maxiters):
            self.converged = True
            self.final_iters = iters
            self.final_diff = diff

        self.t = t
        self.deltaE = deltaE

        # E_ccd_contri = 0.25 * np.einsum("ijab,abij", v[occ, occ, vir, vir], t_next)
        # E_mp2 = 0.25* np.einsum("ijab,abij", v[occ, occ, vir, vir]**2, 1/eps)
        # print(E_ccd_contri)
        # print(E_mp2)

    def next_iteration(self, t):
        v = self.basis.v_
        occ, vir = self.basis.occ_, self.basis.vir_

        res = v[vir, vir, occ, occ] # v_abij
        
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
        res += tp - tp.transpose(0,1,3,2)

        # second P(ij) permutation, double t sum over klcd
        tp = np.einsum("klcd,dcik,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 0.5*(tp + tp.transpose(0,1,3,2))

        # Only P(ab) term, double t sum over klcd
        tp = np.einsum("klcd,aclk,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)
        res -= 0.5*(tp - tp.transpose(0,1,3,2))

        return res
    
    def check_convergence(self, diff, iters, deltaE):
        if diff > 1e10:
            raise ValueError(textwrap.dedent(f"""
            Non-convergence of CCD calculation.
            {iters = }, {diff =}, {deltaE =}
            """))

    
    def __str__(self):
        if not self.has_run:
            return textwrap.dedent(f"""
                -------------------------------------------
                No CCD calculation has been run.
                Currently using:
                    L = {self.basis.L_} basis functions
                    N = {self.basis.N_} occupied functions
                -------------------------------------------
            """)
        return textwrap.dedent(f"""
            -----------------------------------------
            Results from CCD calculation
                dE = {self.deltaE} correlation energy
                converged? {self.converged}
                iters = {self.final_iters} used
                diff = {self.final_diff} at convergence 
            
            Used:
                L = {self.basis.L_} basis functions
                N = {self.basis.N_} occupied functions
            -----------------------------------------
        """)

if __name__ == '__main__':
    from ..basis.Hydrogen import Hydrogen
    from ..HF.HF import HF
    N = 4
    basis = Hydrogen(L=6, N=N, Z=N, spinrestricted=False).load_TB("hydrogen.txt")
    basis.calculate_OB()
    
    hf = HF(basis)
    hf.run()
    Eref = hf.evalute_energy()
    basis = hf.perform_basis_change(basis)

    ccd = CCD(basis)
    ccd.run()

    dE_ccd = ccd.deltaE
    print(Eref)
    print(dE_ccd)
    print(Eref + dE_ccd)
    print(ccd)