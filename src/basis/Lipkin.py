from .Basis import Basis
import numpy as np

class Lipkin(Basis):
    def __init__(self, N, eps, V, **kwargs):
        super().__init__(L=2*N, N=N, spinrestricted=False, **kwargs)
        self.eps = eps
        self.V = V

        self.setup()

    def setup(self):
        H0_lower = np.ones(self.N_)*-self.eps/2
        H0_upper = np.ones(self.N_)*self.eps/2

        self.h_ = np.diag(np.r_[H0_lower, H0_upper])

        self.fill_H1()

    def fill_H1(self):
        self.v_ = self.v_ - self.v_.transpose(0,1,3,2)
        L, N = self.L_, self.N_

        spin_equal = lambda i, j: i // N == j // N
        pos_equal = lambda i, j: i - N == j or i == j - N

        v = self.v_

        # After some math
        for g in range(L):
            for d in range(L):
                for a in range(L):
                    for b in range(L):
                        dspin_bra = spin_equal(g, d)
                        dspin_ket = spin_equal(a, b)
                        dspin_exchange = not spin_equal(g, a)

                        dspin = dspin_bra*dspin_ket*dspin_exchange

                        pos_direct = pos_equal(g, a) * pos_equal(d, b)
                        pos_exchange = pos_equal(g, b) * pos_equal(d, a)

                        v[g, d, a, b] = self.V*dspin*(pos_direct - pos_exchange)

        self.is_AS_ = np.allclose(v, -v.transpose(0,1,3,2))

        # First test        
        # for i in range(N):
        #     for j in range(N):
        #         i_bar = i + N
        #         j_bar = j + N

        #         self.v_[i, j, i_bar, j_bar] = self.V/2
        #         self.v_[j, i, j_bar, i_bar] = self.V/2

        #         self.v_[i, j, j_bar, i_bar] = -self.V/2
        #         self.v_[j, i, i_bar, j_bar] = -self.V/2

        #         self.v_[i_bar, j_bar, i, j] = self.V/2
        #         self.v_[j_bar, i_bar, j, i] = self.V/2

        #         self.v_[j_bar, i_bar, i, j] = -self.V/2
        #         self.v_[i_bar, j_bar, j, i] = -self.V/2