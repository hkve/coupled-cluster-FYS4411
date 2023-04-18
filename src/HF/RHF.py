from ..basis.Hydrogen import Hydrogen
import numpy as np

class RHF:
    def __init__(self, basis, Lf):
        self.basis = basis
        self.Lf_ = Lf
        self.has_run = False
        self.converged = False

    def density_matrix(self, C, L, Lf):
        rho = np.zeros_like(C)

        for alpha in range(L):
            for beta in range(L):
                elm = 0
                for i in range(Lf):
                    elm += C[alpha, i]*C[beta, i]
                rho[alpha,beta] = elm

        return rho

    def run(self, tol=1e-8, maxiters=1000):
        basis = self.basis
        
        L = basis.L_
        Lf = self.Lf_

        C = np.eye(L, L)
        rho = self.density_matrix(C, L, Lf)
        
        eps_hf_old = np.zeros_like(basis.h)
        eps_hf_new = np.zeros_like(basis.h)

        iters = 0
        diff = 1

        while (iters < maxiters) and (diff > tol):
            HFmat = np.zeros(shape=(L,L))

            HFmat = 2*basis.h + 4*np.einsum("gd,agbd->ab", rho, basis.v) - 2*np.einsum("gd,agdb->ab", rho, basis.v)
            
            eps_hf_new, C = np.linalg.eigh(HFmat)

            rho = self.density_matrix(C, L, Lf)
            diff = np.mean(np.abs(eps_hf_new-eps_hf_old))
            eps_hf_old = eps_hf_new
            iters += 1

        self.has_run = True
        if(iters < maxiters):
            self.converged = True

        self.HFmat_ = HFmat
        self.rho_ = rho
        self.C_ = C

    def evalute_energy(self):
        if not self.has_run:
            raise RuntimeError("No Hartree-Fock calculation has been run. Perform .run() first.")
        if not self.converged:
            raise RuntimeWarning("Hartree-Fock calculation has not converged")
        
        basis = self.basis
        L = basis.L_

        E_OB = 0
        E_TB = 0

        for i in range(L):
            E_OB += 2*self.rho_[i,i]*basis.h[i,i]
        
        E_TB = 2*np.einsum("ag,bd,abgd", self.rho_, self.rho_, basis.v) - np.einsum("ag,bd,abdg", self.rho_, self.rho_, basis.v) 

        print(self.converged)
        E = E_OB+E_TB        
        print(E)

    def perform_basis_change(self):
        if not self.has_run:
            raise RuntimeError("No Hartree-Fock calculation has been run. Perform .run() first.")
        if not self.converged:
            raise RuntimeWarning("Hartree-Fock calculation has not converged")
        
        basis = self.basis
        h_prime = np.einsum("ia,jb,ab->ij", self.C_, self.C_, basis.h)
        v_prime = np.einsum("ia,jb,kg,ld,abgd->ijkl", self.C_, self.C_, self.C_, self.C_, basis.v)

        self.basis.h_ = h_prime
        self.basis.v_ = v_prime


if __name__ == '__main__':
    hy = Hydrogen(L=3, Z=2).load_TB("hydrogen.txt")
    hy.calculate_OB()

    rhf = RHF(hy, Lf=1)
    rhf.run(maxiters=20)
    rhf.evalute_energy()
    # print(hy.h)
    # rhf.perform_basis_change()
    # print(hy.h)