import numpy as np
import textwrap
import warnings

from abc import ABC, abstractmethod


class CCbase(ABC):
    def __init__(self, basis, **kwargs): 
        self.basis = basis
        self.has_run = False
        self.converged = False

        self.deltaE = None

    def run(self, tol=1e-5, maxiters=100, p=0, vocal=False):
        basis = self.basis
        v = basis.v_
        occ, vir = self.basis.occ_, self.basis.vir_

        vir_range = basis.L_ - basis.N_
        occ_range = basis.N_

        t = np.zeros(shape=(vir_range, vir_range, occ_range, occ_range))

        eps_v = np.diag(self.f)[occ_range:]
        eps_o = np.diag(self.f)[:occ_range]

        eps = -eps_v[:,None,None,None] - eps_v[None,:,None,None] \
               +eps_o[None,None,:,None] + eps_o[None,None,None,:]

        epsinv = 1/eps

        iters = 0
        diff = 321
        deltaE = 321
        crashing = False
        while (iters < maxiters) and (diff > tol) and not crashing:
            t_next = t*p + (1-p)*self.next_iteration(t)*epsinv 

            deltaE_next = self.evalute_energy_iteration(t_next, v, occ, vir)
            diff = np.abs(deltaE_next - deltaE)
            
            self.check_MP2_first_iter(iters, p, deltaE_next, v, occ, vir, epsinv)

            if vocal:
                self.beVocal(diff, deltaE_next, deltaE, iters)
                self.check_amplitude_symmetry(t_next)

            crashing = self.check_convergence(diff, iters, deltaE_next)
            deltaE = deltaE_next
            t = t_next
            iters += 1
        
        self.has_run = True
        if(iters < maxiters and not crashing):
            self.converged = True
            self.final_iters = iters
            self.final_diff = diff

        self.t = t
        self.deltaE = deltaE

        return self
    
    @abstractmethod
    def next_iteration(self, t, v, occ, vir):
        pass 

    @abstractmethod
    def evalute_energy_iteration(self, t, v, occ, vir):
        pass

    def evaluate_energy(self, correlation=False):
        if not self.has_run:
            raise UserWarning("Did not run?")
            return None
        
        E = self.deltaE
        if not correlation:
            E += self.basis.evaluate_energy()
        if not self.converged:
            warnings.warn("Did not converge :(")
            E = 0

        return E
    
    def check_convergence(self, diff, iters, deltaE):
        if diff > 1e10:
            return True
        else:
            return False
            '''
            raise ValueError(textwrap.dedent(f"""
            Non-convergence of CCD calculation.
            {iters = }, {diff =}, {deltaE =}
            """))
            '''
    def beVocal(self, diff, deltaE_next, deltaE, iters):
        print(f"{diff = :.4e}, {deltaE = :.4f}, {deltaE_next = :.4f}, {iters = }")
    
    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinv):
        if iters == 0:
            warnings.warn("This scheme does not implement MP2 energy check after first iteration")

    def check_amplitude_symmetry(self, t):
        warnings.warn("This scheme does not implement amplitude symmetry check")

    def __str__(self):
        if not self.has_run:
            return textwrap.dedent(f"""
                -------------------------------------------
                No CCD calculation has been run.
                Currently using:
                    L = {self.basis.L_} basis functions
                    N = {self.basis.N_} occupied functions
                    Spinrestricted? {self.basis.spinrestricted_}
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
                Spinrestricted? {self.basis.spinrestricted_}
            -----------------------------------------
        """)