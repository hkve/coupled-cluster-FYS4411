import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import eigh

class HFbase(ABC):
    def __init__(self, basis):
        self.basis = basis
        self.has_run = False
        self.converged = False


    def density_matrix(self, C):
        return self.basis.degeneracy_*np.einsum("ai,bi->ab", C[:,self.basis.occ], C[:,self.basis.occ])


    @abstractmethod
    def evaluate_energy_scheme(self):
        pass        


    @abstractmethod
    def evaluate_HFmat_scheme(self, rho, h, v):
        pass


    def run(self, tol=1e-8, maxiters=1000, vocal=False):
        if self.has_run:
            self.has_run = False
            self.converged = False
            
        basis = self.basis
        
        L = basis.L_
        N = basis.N_

        C = np.eye(L, L)
        rho = self.density_matrix(C)
        
        eps_hf_old = np.zeros_like(np.diag(basis.h))
        eps_hf_new = np.zeros_like(np.diag(basis.h))

        iters = 0
        diff = 1

        while (iters < maxiters) and (diff > tol):
            HFmat = np.zeros(shape=(L,L))

            HFmat = self.evaluate_HFmat_scheme(rho, basis.h, basis.v)

            eps_hf_new, C = eigh(HFmat, basis.s_)

            rho = self.density_matrix(C)
            diff = np.mean(np.abs(eps_hf_new-eps_hf_old))
            eps_hf_old = eps_hf_new
            iters += 1

            if vocal:
                print(f"i = {iters}, mo = {eps_hf_new}")

        self.has_run = True
        if(iters < maxiters):
            self.converged = True
            self.iters_ = iters
            self.diff_ = diff

        self.HFmat_ = HFmat
        self.rho_ = rho
        self.C_ = C

        return self


    def evaluate_energy(self):
        if not self.has_run:
            raise RuntimeError("No Hartree-Fock calculation has been run. Perform .run() first.")
        if not self.converged:
            raise RuntimeWarning("Hartree-Fock calculation has not converged")
        return self.evaluate_energy_scheme() + self.basis.energy_shift_
    
    
    def perform_basis_change(self, basis, inverse=False):
        if not self.has_run:
            raise RuntimeError("No Hartree-Fock calculation has been run. Perform .run() first.")
        if not self.converged:
            raise RuntimeWarning("Hartree-Fock calculation has not converged")

        C = self.C_.copy()

        if inverse:
            C = C.T

        h_prime = np.einsum("ai,bj,ab->ij", C, C, basis.h, optimize=True)
        v_prime = np.einsum("ai,bj,gk,dl,abgd->ijkl", C, C, C, C, basis.v, optimize=True)
        
        occ = basis.occ_
        basis.v_ = v_prime

        basis.h_ = h_prime #+ np.einsum("piqi", basis.v[:,occ,:,occ])

        return basis