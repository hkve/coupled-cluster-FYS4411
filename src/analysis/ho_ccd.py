from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD

def main():
    L, N = 12, 2
    # ho = HarmonicsOscillator(L = L, N=N, spinrestricted=False)
    # ho.calculate_OB()
    # ho.calculate_TB()

    # print(f"Eref = {ho.evaluate_energy()}")

    # ccd1 = CCD(ho)
    # ccd1.run(vocal=False, tol=1e-4, p=0.4)
    # print(f"CCD,HO = {ccd1.evaluate_energy()}")
    # print(ccd1)

    # ho = HarmonicsOscillator(L = L, N=N, spinrestricted=False)
    # ho.calculate_OB()
    # ho.calculate_TB()

    # hf = HF(ho)
    # hf.run()
    # print(f"HF = {hf.evaluate_energy()}")
    # ho = hf.perform_basis_change(ho)


    # ccd2 = CCD(ho)
    # ccd2.run(vocal=False, tol=1e-4, p=0)
    # print(f"CCD,HF = {ccd2.evaluate_energy()}")
    # print(ccd2)

    ho = HarmonicsOscillator(L = L, N=N, spinrestricted=True)
    ho.calculate_OB()
    ho.calculate_TB()

    hf = RHF(ho)
    hf.run()
    print(f"HF = {hf.evaluate_energy()}")
    ho = hf.perform_basis_change(ho)

    ho.restricted_to_unrestricted()
    ccd3 = CCD(ho)
    ccd3.run(vocal=False, tol=1e-4, p=0)
    print(f"CCD,RHF = {ccd3.evaluate_energy()}")
    print(ccd3)
if __name__ == '__main__':
    main()



     # L = ho.L_
    # for i in range(L):
    #     for j in range(L):
    #         print(f"{ho.h[i,j]:6.3f} ", end="")
    #     print()
    # ccd2 = CCD(ho)
    # ccd2.run(vocal=False, tol=1e-6)

    # print(f"ECCD(HF) = {ccd2.evaluate_energy():.5f}")
    # print(ccd2)
