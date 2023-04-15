from .Basis import SpinRestrictedBasis
import numpy as np

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
        self.n_to_p = {(0,0): 0}
        
        p = 1
        for shell, N in zip(self.shell_numbers_, self.cummulative_Ns_):
            if(N > self.L_): # Check shell table here, replace ">" with ">="?
                break

            for i in range(shell):
                n1 = (shell-i, i)
                n2 = (i, shell-i)
                if n1 == n2:
                    self.n_to_p[n1] = p
                    p += 1
                else:
                    self.n_to_p[n1] = p
                    self.n_to_p[n2] = p+1
                    p += 2

        self.p_to_n = {v: k for k, v in self.n_to_p.items()}


    def calculate_TB(self):
        Hermite = []
        H0 = lambda x: 1
        H1 = lambda x: 2*x


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
    ho = HarmonicsOscillator(L = 10)
    ho.calculate_OB()

    print(ho.n_to_p)
    # print(ho.h)