from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD

def main():
    ho = HarmonicsOscillator(L = 12, N=2, spinrestricted=False)
    ho.calculate_OB()
    ho.calculate_TB()

    print(f"E0ref = {ho.evaluate_energy():.5f}")

    # ccd1 = CCD(ho)
    # ccd1.run(vocal=False, tol=1e-6)
    # print(f"ECCD(HO) = {ccd1.evaluate_energy():.5f}")
    
    hf = HF(ho)
    hf.run()

    print(f"E(HF) = {hf.evaluate_energy():.5f}")

    ho = hf.perform_basis_change(ho)
    print(ho.h)
    # ccd2 = CCD(ho)
    # ccd2.run(vocal=False, tol=1e-6)

    # print(f"ECCD(HF) = {ccd2.evaluate_energy():.5f}")
    # print(ccd2)

if __name__ == '__main__':
    main()