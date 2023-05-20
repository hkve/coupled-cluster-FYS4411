from ..basis.Hydrogen import Hydrogen
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD, RCCD

def show(relerror, E, scheme, E_FCI):
    if relerror:
        e = abs((E-E_FCI)/E_FCI)
        print(f"Percent_error_{scheme} = {100*e:.2f}")
    else:
        print(f"E_{scheme} = {E:.4f}")

def main(N, spinrestricted=False, relerror=False):
    HF_scheme = {False: HF, True: RHF}
    CCD_scheme = {False: CCD, True: RCCD}
    E_FCI = {2: -2.9037, 4: -14.6674}

    L = 6
    atom = Hydrogen(L=L, N=N, Z=N, spinrestricted=spinrestricted).load_TB("hydrogen.txt")
    atom.calculate_OB()

    Eref = atom.evaluate_energy()
    show(relerror, Eref, "Ref", E_FCI[N])

    hf = HF_scheme[spinrestricted](atom)
    hf.run()
    Ehf = hf.evaluate_energy()
    show(relerror,Ehf, "HF", E_FCI[N])

    ccd = CCD_scheme[spinrestricted](atom)
    ccd.run()
    Eccd = ccd.evaluate_energy()
    show(relerror,Eccd, "CCD", E_FCI[N])

    atom = hf.perform_basis_change(atom)
    ccd = CCD_scheme[spinrestricted](atom)
    ccd.run()
    EccdHF = ccd.evaluate_energy()
    show(relerror,EccdHF, "CCD(HF)", E_FCI[N])

if __name__ == '__main__':
    relerror = True

    # Helium
    print("Helium unrestricted")
    main(N=2, spinrestricted=False, relerror=relerror)
    print("\nHelium unrestricted")
    main(N=2, spinrestricted=True, relerror=relerror)

    print("\n")
    
    # Beryllium
    print("\nBeryllium unrestricted")
    main(N=4, spinrestricted=False, relerror=relerror)
    print("\nBeryllium restricted")
    main(N=4, spinrestricted=True, relerror=relerror)