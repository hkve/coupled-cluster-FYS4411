import numpy as np
import textwrap
import warnings

from abc import ABC, abstractmethod


class CCbase(ABC):
    def __init__(self, basis, **kwargs): 
        self._basis = basis
        self._has_run = False
        self._converged = False

        self._M = basis.L_ - basis.N_
        self._N = basis.N_
        self._deltaE = None
        self._orders = []

    def run(self, tol=1e-5, maxiters=100, p=0, vocal=False):
        basis = self._basis
        v = basis.v_
        occ, vir = self._basis.occ_, self._basis.vir_

        N, M = self._N, self._M

        # Store amplitdue based on which orders to include
        t_amplitudes = self._get_t_amplitudes()
        epsinvs = self._get_epsinvs()

        iters = 0
        diff = 321
        deltaE = 321
        crashing = False
        while (iters < maxiters) and (diff > tol) and not crashing:
            t_amplitudes_next = self.next_iteration(t_amplitudes, epsinvs)
            # t_next = t*p + (1-p)*self.next_iteration(t)*epsinv

            for order in self._orders:
                t_amplitudes_next[order] = t_amplitudes[order]*p + (1-p)*t_amplitudes_next[order]

            deltaE_next = self.evalute_energy_iteration(t_amplitudes_next, v, occ, vir)
            diff = np.abs(deltaE_next - deltaE)
            
            self.check_MP2_first_iter(iters, p, deltaE_next, v, occ, vir, epsinvs)

            if vocal:
                self._beVocal(diff, deltaE_next, deltaE, iters)
                self._check_amplitude_symmetry(t_amplitudes_next)

            crashing = self._check_convergence(diff, iters, deltaE_next)
            deltaE = deltaE_next
            t_amplitudes = t_amplitudes_next
            iters += 1
        
        self._has_run = True
        if(iters < maxiters and not crashing):
            self._converged = True
            self.final_iters = iters
            self.final_diff = diff

        self.t_amplitudes = t_amplitudes_next
        self._deltaE = deltaE

        return self
    
    @abstractmethod
    def next_iteration(self, t, v, occ, vir):
        pass 

    @abstractmethod
    def evalute_energy_iteration(self, t, v, occ, vir):
        pass

    @abstractmethod
    def _get_t_amplitudes(self):
        pass

    def _get_t_amplitudes(self):
        N, M = self._N, self._M
        mapping = {"S": 1, "D": 2}

        t_amplitudes = {}
        for order in self._orders:
            shape = tuple([M]*mapping[order] + [N]*mapping[order])
            t_amplitudes[order] = np.zeros(shape=shape)

        return t_amplitudes
    

    def _get_epsinvs(self):
        N, M = self._N, self._M

        eps_v = np.diag(self._f)[N:]
        eps_o = np.diag(self._f)[:N]
        orders = self._orders

        epsinvs = {}
        if "S" in orders:
            eps = -eps_v[:,None] +eps_o[:,None]
            assert eps.shape == (M, N), "Error in singles sp energies"
            epsinvs["S"] = 1/eps
        if "D" in orders:
            eps = -eps_v[:,None,None,None] - eps_v[None,:,None,None] \
               +eps_o[None,None,:,None] + eps_o[None,None,None,:]
            assert eps.shape == (M, M, N, N), "Error in doubles sp energies"
            epsinvs["D"] = 1/eps

        return epsinvs   

    def evaluate_energy(self, correlation=False):
        if not self._has_run:
            raise UserWarning("Did not run?")
            return None
        
        E = self._deltaE
        if not correlation:
            E += self._basis.evaluate_energy()
        if not self._converged:
            warnings.warn("Did not converge :(")
            E = 0

        return E
    
    def _check_convergence(self, diff, iters, deltaE):
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
    def _beVocal(self, diff, deltaE_next, deltaE, iters):
        print(f"{diff = :.4e}, {deltaE = :.4f}, {deltaE_next = :.4f}, {iters = }")
    
    def check_MP2_first_iter(self, iters, p, E_CCD0, v, occ, vir, epsinv):
        if iters == 0:
            warnings.warn("This scheme does not implement MP2 energy check after first iteration")

    def _check_amplitude_symmetry(self, ts):
        warnings.warn("This scheme does not implement amplitude symmetry check")

    def __str__(self):
        if not self._has_run:
            return textwrap.dedent(f"""
                -------------------------------------------
                No CCD calculation has been run.
                Currently using:
                    L = {self._basis.L_} basis functions
                    N = {self._basis.N_} occupied functions
                    Spinrestricted? {self._basis.spinrestricted_}
                -------------------------------------------
            """)
        return textwrap.dedent(f"""
            -----------------------------------------
            Results from CCD calculation
                dE = {self._deltaE} correlation energy
                converged? {self._converged}
                iters = {self.final_iters} used
                diff = {self.final_diff} at convergence 
            
            Used:
                L = {self._basis.L_} basis functions
                N = {self._basis.N_} occupied functions
                Spinrestricted? {self._basis.spinrestricted_}
            -----------------------------------------
        """)