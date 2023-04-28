from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..CC.CCD import CCD
from ..HF.HF import HF
import numpy as np

def main():
    ho = HarmonicsOscillator(L=72, N=2, spinrestricted=False)
    ho.calculate_OB()
    ho.calculate_TB()

    hf = HF(ho)
    hf.run()
    ho = hf.perform_basis_change(ho)
    print(np.round(ho.h, 2))
    print(ho.evaluate_energy())
    ccd = CCD(ho)
    ccd.run(tol=1e-8)

    print(ccd)
    print(ccd.evaluate_energy()) # 3.039048
if __name__ == '__main__':
    main()