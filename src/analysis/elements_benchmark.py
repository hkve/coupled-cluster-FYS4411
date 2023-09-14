from ..basis.Molecule import Molecule
from ..HF.HF import HF, RHF

import pyscf

def run_single(atom, basisset, tol=1e-8):
    basis = Molecule(
        atom=atom,
        basis=basisset,
        spinrestricted=True,
    )


    hf1 = RHF(basis).run(tol=tol)
    E_rhf = hf1.evaluate_energy()

    basis.restricted_to_unrestricted()
    hf2 = HF(basis).run(tol=tol)
    E_hf = hf2.evaluate_energy()

    hf3 = pyscf.scf.HF(basis.mol).run(verbose=0,tol=tol)
    E_pyscf = hf3.e_tot

    print(f"""
    {atom} using {basisset}
        {E_rhf = }
        {E_hf = }
        {E_pyscf = }
    """)

def main():
    # Atoms: Helium, Beryllium and Neon
    run_single(atom="He 0 0 0", basisset="cc-pVDZ")
    run_single(atom="Be 0 0 0", basisset="cc-pVDZ")
    run_single(atom="Ne 0 0 0", basisset="cc-pVDZ")

    # Di-Atoms:
    run_single(atom="H 0 0 0; H 0 0 1.2", basisset="sto-3g")
if __name__ == "__main__":
    main()