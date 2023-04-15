import numpy as np
from abc import ABC, abstractmethod

class SpinRestrictedBasis(ABC):
    @abstractmethod
    def __init__(self, L=10, **kwargs):
        self.L_ = L
        self.h_ = np.zeros((L, L), dtype=float)
        self.v_ = np.zeros((L, L, L, L), dtype=float)

    @property
    def h(self):
        return self.h_

    @property
    def v(self):
        return self.v_

    def save_TB(self, filename):
        return self

    def load_TB(self, filename):
        return self