from ..basis.Hydrogen import Hydrogen
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD

def main():
    L, N = 6, 2
    hy = Hydrogen(L=L, N=N, Z=N, spinrestricted=True).load_TB("hydrogen.txt")
    hy.calculate_OB()
    print(hy.evaluate_energy())
    hf = RHF(hy)
    hf.run()
    hy = hf.perform_basis_change(hy)
    print(hy.evaluate_energy())
    hy.restricted_to_unrestricted()
    ccd = CCD(hy)
    ccd.run(p=0.5)
    print(ccd.evaluate_energy())
    print(ccd)


def main1():
    L, N = 6, 2
    hy = Hydrogen(L=L, N=N, Z=N, spinrestricted=True).load_TB("hydrogen.txt")
    hy.calculate_OB()
    hf = RHF(hy)
    hf.run()
    hy = hf.perform_basis_change(hy)
    import numpy as np
    np.save("h_elms", hy.h)
    np.save("elms", hy.v)
    # h = np.load("h_elm.npy")

    # np.allclose(h, hy.h)
if __name__ == "__main__":
    # main()
    main1()