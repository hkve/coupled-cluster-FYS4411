from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
import numpy as np

def main():
    L, N = 12, 2
    ho = HarmonicsOscillator(L=L, N=N, spinrestricted=False, omega=1.0)
    ho.calculate_OB()
    ho.calculate_TB()
    Eref = ho.evaluate_energy()

    hf = HF(ho)
    hf.run()

    Ehf = hf.evaluate_energy()
    ho_hf_basis = hf.perform_basis_change(ho)
    print(np.round(ho_hf_basis.h, 2))
    Eref_hf_basis = ho_hf_basis.evaluate_energy()

    print(f"""General:
        {Eref = }
        {Ehf = }
        {Eref_hf_basis =}
        """)
    
    ho = HarmonicsOscillator(L=L, N=N, spinrestricted=True, omega=1.0)
    ho.calculate_OB()
    ho.calculate_TB()
    Eref = ho.evaluate_energy()

    hf = RHF(ho)
    hf.run()


    Ehf = hf.evaluate_energy()
    ho_hf_basis = hf.perform_basis_change(ho)
    Eref_hf_basis = ho_hf_basis.evaluate_energy()

    print(f"""Spinrestricted:
        {Eref = }
        {Ehf = }
        {Eref_hf_basis =}
        """)

    ho_hf_basis.restricted_to_unrestricted()

    Eref_hf_basis = ho_hf_basis.evaluate_energy()

    print(f"""RHF but expand:
        {Eref_hf_basis =}
        """)
    
    print(np.round(ho_hf_basis.h, 2))


if __name__ == '__main__':
    main()