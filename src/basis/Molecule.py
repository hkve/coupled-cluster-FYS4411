from .Basis import Basis
import numpy as np
import pyscf

class Molecule(Basis):
    def __init__(self, atom, basis, spinrestricted=True, **kwargs):
        mol = pyscf.gto.Mole()
        mol.build(atom=atom, basis=basis)

        super().__init__(L=2*mol.nao, N=mol.nelectron, spinrestricted=spinrestricted)

        L = self.L_
        self.s_ = mol.intor_symmetric("int1e_ovlp")
        self.h_ = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        self.u_ = mol.intor("int2e").reshape(L, L, L, L).transpose(0, 2, 1, 3)

        if not self.spinrestricted_:
            self.make_AS()