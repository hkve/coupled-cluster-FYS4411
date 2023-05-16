from ..basis.ChiralPT import ChiralPT
from ..HF.HF import HF
from ..CC.CCD import CCD

def main():
    pt = ChiralPT(L=80, N=16).load_TB("nucleitwobody.dat")
    pt.calculate_OB()

    # spin = pt.spin_projection_expval()
    # print(spin)
    Eref = pt.evaluate_energy()

    print(Eref)

    hf = HF(pt)
    hf.run()

    Ehf = hf.evaluate_energy()
    print(Ehf)

    pt = hf.perform_basis_change(pt, keep_coefs=False)

    # spin = pt.spin_projection_expval(C=pt.C)
    # print(spin)
    ccd = CCD(pt)
    ccd.run()
    Eccd = ccd.evaluate_energy()

    print(Eccd)
if __name__ == "__main__":
    main()