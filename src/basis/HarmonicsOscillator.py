from .Basis import Basis
import numpy as np
from scipy.special import hermite, factorial
from scipy.integrate import nquad
from math import exp

from quantum_systems import TwoDimensionalHarmonicOscillator
from scipy import sparse
import re
import einops

class HarmonicsOscillator(Basis):
    def __init__(self, L=12, N=3, spinrestricted=True, omega=1.0, R=None, **kwargs):
        self.omega_ = omega
        self.shell_numbers_ = np.arange(1, 13)
        self.degeneracies_ = 2*self.shell_numbers_
        self.cummulative_Ns_ = np.cumsum(self.degeneracies_)
        
        if R:
            L = self.cummulative_Ns_[R-1]

        super().__init__(L=L, N=N, spinrestricted=spinrestricted, **kwargs)
        assert self.degeneracy_*self.L_ in self.cummulative_Ns_, f"{self.L_ = } does not give a closed shell. Must be in {self.cummulative_Ns_}"
        assert self.degeneracy_*self.N_ in self.cummulative_Ns_, f"{self.N_ = } does not give a closed shell. Must be in {self.cummulative_Ns_}"

        self.make_mappings()

    def make_mappings(self):
        """
        Makes a mapping between quantum numbers and matrix indicies, both ways.
        """
        n_to_p_restricted = {(0,0): 0}
        
        shell_numbers, cummulative_Ns = [], []
        p = 1
        for shell, N in zip(self.shell_numbers_, self.cummulative_Ns_):
            shell_numbers.append(shell)
            cummulative_Ns.append(N)
            
            if(N >= self.degeneracy_*self.L_):
                break

            for i in range(shell):
                n1 = (shell-i, i)
                n2 = (i, shell-i)
                
                if(i > shell//2):
                    break
                if n1 == n2:
                    n_to_p_restricted[n1] = p
                    p += 1
                else:
                    n_to_p_restricted[n1] = p
                    n_to_p_restricted[n2] = p+1
                    p += 2

        n_to_p = n_to_p_restricted
        if not self.spinrestricted_:
            n_to_p = {}
            p = 0
            for k, v in n_to_p_restricted.items():
                n_up = (*k, 0)
                n_down = (*k, 1)
                n_to_p[n_up] = p
                n_to_p[n_down] = p+1
                p += 2
            
        self.n_to_p_ = n_to_p
        self.p_to_n_ = {v: k for k, v in self.n_to_p_.items()}
        self.shell_numbers_ = np.array(shell_numbers)
        self.cummulative_Ns_ = np.array(cummulative_Ns)

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

        return self.A(*n_p)*self.A(*n_q)*self.A(*n_r)*self.A(*n_s) * I / self.omega**(3/2)

    def calculate_TB(self):
        rL = self.degeneracy_*self.L_//2 # restricted L
        tdho = TwoDimensionalHarmonicOscillator(rL, 5, 11, omega=self.omega_)
        if self.spinrestricted_:
            self.v_ = tdho.u
        else:
            for i in range(rL):
                for j in range(rL):
                    for k in range(rL):
                        for l in range(rL):
                            self.fill_with_spin(tdho.u[i,j,k,l], i,j,k,l)

            self.make_AS()

        return self

    def save_elements(self, filename):
        P, Q, R, S = np.nonzero(self.v_)
        v_nonzero = self.v_[P, Q, R, S]
        L_saved = np.array([self.L_])

        np.savez(filename, L_saved, P, Q, R, S, v_nonzero)

    def load_elements(self, filename):
        npzfile = np.load(filename)#, allow_pickle=True)
        L_saved = npzfile["arr_0"][0]
        
        assert self.L_ <= L_saved, f"File {filename} you are reading from has fewer basis functions (L_read) {L_saved} than you require (L_req) {self.L_}. Calculating new ones are required"

        indicies = np.c_[npzfile["arr_1"],npzfile["arr_2"],npzfile["arr_3"],npzfile["arr_4"]]
        v_nonzero = npzfile["arr_5"]

        drops = np.where(indicies >= self.L_)[0]
        indicies = np.delete(indicies, drops, axis=0)
        v_nonzero = np.delete(v_nonzero, drops)

        P, Q, R, S = indicies.T

        self.v_[P, Q, R, S] = v_nonzero

    def calculate_OB(self):
        """
        Calculate unperturbed matrix elements for the Harmonics Oscillator
        Uses the fact that the energy is frequency times shell number.
        """
        k = 0
        for i in range(self.L_):
            if i >= self.cummulative_Ns_[k]//self.degeneracy_:
                k += 1
            self.h_[i,i] = self.omega_*(self.shell_numbers_[k])

        return self
    
    def change_frequency(self, omega):
        old_omega = self.omega_
        new_omega = omega

        assert new_omega > 0, f"Lower threshold for omega at 0"
        self.h_ *= new_omega/old_omega
        self.v_ *= np.sqrt(new_omega/old_omega)
        self.omega_ = new_omega


if __name__ == '__main__':
    ho = HarmonicsOscillator(L = 156, N=12, spinrestricted=True)

    for shell in ho.shell_numbers_:
        print(shell, ho.cummulative_Ns_[shell-1])
        for i in range(ho.cummulative_Ns_[shell-2], ho.cummulative_Ns_[shell-1]):
            print(ho.p_to_n_[i])

    print(ho.p_to_n_)
    # ho.calculate_OB()
    # ho.calculate_TB()
    # ho.make_AS()