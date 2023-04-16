from .Basis import SpinRestrictedBasis
import numpy as np
from scipy.special import hermite, factorial
from scipy.integrate import nquad
from math import exp

class HarmonicsOscillator(SpinRestrictedBasis):
    def __init__(self, L=10, omega=1, **kwargs):
        self.omega_ = omega
        self.shell_numbers_ = np.arange(1, 10)
        self.degeneracies_ = [n for n in self.shell_numbers_]
        self.cummulative_Ns_ = [sum(self.degeneracies_[:i]) for i in self.shell_numbers_]
        
        assert L in self.cummulative_Ns_, f"{L = } does not give a closed shell. Must be in {self.cummulative_Ns_}"
        super().__init__(L=L, **kwargs)
        self.make_mappings()

    def make_mappings(self):        
        """
        Makes a mapping between quantum numbers and matrix indicies, both ways.
        """
        self.n_to_p_ = {(0,0): 0}
        
        shell_numbers, cummulative_Ns = [], []
        p = 1
        for shell, N in zip(self.shell_numbers_, self.cummulative_Ns_):
            shell_numbers.append(shell)
            cummulative_Ns.append(N)
            
            if(N >= self.L_):
                break

            for i in range(shell):
                n1 = (shell-i, i)
                n2 = (i, shell-i)
                
                if(i > shell//2):
                    break
                if n1 == n2:
                    self.n_to_p_[n1] = p
                    p += 1
                else:
                    self.n_to_p_[n1] = p
                    self.n_to_p_[n2] = p+1
                    p += 2

        self.p_to_n_ = {v: k for k, v in self.n_to_p_.items()}
        self.shell_numbers_ = shell_numbers
        self.cummulative_Ns_ = cummulative_Ns

    def A(self, nx, ny):
        return 2**( -(nx+ny)/2 ) * np.sqrt(self.omega_ / (factorial(nx) * factorial(ny) * np.pi))

    def phi(self, x, y, nx, ny):
        return hermite(nx)(x)*hermite(ny)(y)*exp( -(x**2+y**2)/2 )

    def integrand(self, x1, y1, x2, y2, n_p, n_q, n_r, n_s):
        return self.phi(x1, y1, *n_p)*self.phi(x2, y2, *n_q)*self.phi(x1, y1, *n_r)*self.phi(x2, y2, *n_s)
    
    
    def matrix_element(self, p, q, r, s):
        n_p = self.p_to_n_[p]
        n_q = self.p_to_n_[q]
        n_r = self.p_to_n_[r]
        n_s = self.p_to_n_[s]
        
        lims = [[-1, 1] for _ in range(4)]

        opts = {"maxp1": 3}
        I = nquad(self.integrand, lims, args=(n_p, n_q, n_r, n_s), opts=opts)

        return A(*n_p)*A(*n_q)*A(*n_r)*A(*n_s) * I / self.omega**(3/2)

    def calculate_TB(self):
        L = self.L_
        v = self.v_

        for p in range(L):
            for q in range(L):
                for r in range(L):
                    for s in range(L):
                        if np.isnan(v[p,q,r,s]):
                            v[p,q,r,s] = self.matrix_element(p,q,r,s)
                            v[q,p,s,r] = v[p,q,r,s]
                        else:
                            continue
    
    def calculate_OB(self):
        """
        Calculate unperturbed matrix elements for the Harmonics Oscillator
        Uses the fact that the energy is frequency times shell number.
        """
        k = 0
        for i in range(self.L_):
            if i >= self.cummulative_Ns_[k]:
                k += 1
            self.h_[i,i] = self.omega_*(self.shell_numbers_[k])

if __name__ == '__main__':
    ho = HarmonicsOscillator(L = 3)
    ho.calculate_OB()
    ho.calculate_TB()

