from .Basis import Basis
import numpy as np

class Lipkin(Basis):
    def __init__(self, N, eps, V, **kwargs):
        super().__init__(L=2*N, N=N, spinrestricted=False, **kwargs)
        self.eps = eps
        self.V = V

    def setup(self):
        L, N = self.L_, self.N_
        self.h_ = self.fill_H0(L, N)

        self.v_ = self.fill_H1(L, N)
        self.is_AS_ = np.allclose(self.v_, -self.v_.transpose(0,1,3,2))


    def fill_H0(self, L, N):
        H0_lower = np.ones(self.N_)*-self.eps/2
        H0_upper = np.ones(self.N_)*self.eps/2

        return np.diag(np.r_[H0_lower, H0_upper])
    

    def fill_H1(self, L, N):
        spin_equal = lambda i, j: i // N == j // N
        pos_equal = lambda i, j: i - N == j or i == j - N

        v = np.zeros_like(self.v_)
        print(v.shape)
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

        return v
    

class TwoLipkins(Lipkin):
    def __init__(self, N_each, eps, V, **kwargs):
        N = 2*N_each
        super().__init__(N=N, eps=eps, V=V, **kwargs)
        self.eps = eps
        self.V = V

        self.setup()

    def setup(self):
        Lsub, Nsub = self.L_//2, self.N_//2
        L = self.L_
        
        h = self.fill_H0(Lsub, Nsub)
        v = self.fill_H1(Lsub, Nsub)

    
        I = np.eye(2)
        self.h_ = np.kron(I, h)

        self.v_ = np.zeros(shape=(L,L,L,L))
        # Fill this to be like
        self.v_ = np.kron(v, I)

        print(self.v_.shape)