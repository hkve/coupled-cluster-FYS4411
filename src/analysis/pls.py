from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..CC.CCD import CCD, RCCD
from ..HF.HF import HF, RHF
import numpy as np

def spin_projection_expval(basis, C):
    L = basis.L_
    Lhalf = L//2
    s_posibilities = np.array([0.5,-0.5])
    s = np.tile(s_posibilities, Lhalf)
    
    expval = 0
    for i in range(basis.N_):
        for p in range(L):
            expval += C[p,i]**2 * s[p]

    return expval

def main():
    ho = HarmonicsOscillator(L=20, N=2, spinrestricted=True, fast=True)
    ho.calculate_OB()
    ho.calculate_TB()

    hf = RHF(ho)
    hf.run()
    ho = hf.perform_basis_change(ho, keep_coefs=True)
    ho.restricted_to_unrestricted()
    print(ho.evaluate_energy())

    ccd = CCD(ho)
    ccd.run(tol=1e-8, p=0.3)

    print(ccd)
    print(ccd.evaluate_energy()) 
    # L = 12, N = 2
    # CCD(HF) 3.039048
    # CCD(HO) 3.141827
if __name__ == '__main__':
    main()