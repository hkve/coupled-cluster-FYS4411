from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD

def main():
    ho = HarmonicsOscillator(L = 6, N=2, spinrestricted=False)
    ho.calculate_OB()
    ho.calculate_TB()

    print(ho.evaluate_energy())

    hf = HF(ho)
    hf.run()

    print(hf.evalute_energy())
    ho = hf.perform_basis_change(ho)

    ccd = CCD(ho)
    ccd.run(vocal=True)

    ccd.evaluate_energy()
    print(ccd)

if __name__ == '__main__':
    main()