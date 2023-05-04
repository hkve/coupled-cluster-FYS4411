from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..CC.CCD import CCD
from ..HF.HF import HF, RHF
import numpy as np

def main():
    ho = HarmonicsOscillator(L=72, N=2, spinrestricted=True, fast=True)
    ho.calculate_OB()
    ho.calculate_TB()

    hf = RHF(ho)
    hf.run()
    ho = hf.perform_basis_change(ho)
    ho.restricted_to_unrestricted()
    print(np.round(ho.h, 2))
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