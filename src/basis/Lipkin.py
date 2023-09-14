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

        o_sub1 = slice(0,Nsub)
        v_sub1 = slice(Nsub,Lsub)
        o_sub2 = slice(Lsub,Lsub+Nsub)
        v_sub2 = slice(Lsub+Nsub,Lsub+Lsub)

        # Need to rethink order here...
        # occ needs to be in order, that is index as [p_sys(+/-)] 
        # (1_1s-,1_2s-,...,N_2s-,1_1s+,1_2s+,...,2_Ns+)
        # or
        # (1_1s-,1_2s-,...,N_1s-,1_2s-,...,N_2s-,...)

        indicies = np.arange(0, L)
        
        self.occ_ = np.ix_(indicies[o_sub1], indicies[o_sub1])
        self.vir_ = np.ix_(indicies[v_sub1], indicies[v_sub2])
        
        print(self.occ_)
        I = np.eye(2)
        self.h_ = np.kron(I, h)

        print(
            self.h_[self.occ_[0], self.occ[0]]
        )
        exit()
        self.v_ = np.zeros(shape=(L,L,L,L))
        # Fill this to be like
        self.v_ = np.kron(v, I)