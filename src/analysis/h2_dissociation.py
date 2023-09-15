from ..HF.HF import HF, RHF
from ..basis.Molecule import Molecule

import numpy as np
import matplotlib.pyplot as plt
import pyscf

def run_hf(r, restricted=False):
    H2 = Molecule(
        atom=f"H 0 0 0; H 0 0 {r}",
        basis="sto-3g",
        spinrestricted=restricted
    )

    hf = None
    if restricted:
        hf = RHF(H2).run()
    else:
        hf = HF(H2).run()

    E = hf.evaluate_energy()
    print(f"{r = :.4f}, {E = :.4f}, {restricted = }")
    return E      

def run_pyscf(r, restricted=False):
    mol = pyscf.gto.Mole()
    mol.build(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g")

    if restricted:
        hf = pyscf.scf.RHF(mol=mol).run(verbose=0)
    else:
        hf = pyscf.scf.GHF(mol=mol).run(verbose=0)

    return hf.e_tot

def run_pyscf_uhf(r):
    mol = pyscf.gto.Mole()
    mol.build(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g")

    return pyscf.scf.RHF(mol=mol).run(verbose=0).e_tot

def main():
    E, r  = {}, {}
    E_2H = 2*-0.4997905473364306

    r["koloswolniewicz"] = np.array([
    0.529177249, 0.6350126987999999, 0.6879304237, 0.7355563761099999, 0.7414302435739, 0.7461399210899999, 0.7937658734999999, 0.8466835984, 0.9525190482, 1.058354498, 1.1641899478, 1.2700253975999998, 1.3758608474, 1.4816962971999998, 1.5875317469999999, 1.6933671968, ])
    E["koloswolniewicz"] = np.array([
    -1.12453881, -1.16493435, -1.17234623, -1.17445199, -1.17447498, -1.17446041, -1.17285408, -1.16858212, -1.15506752, -1.13813155, -1.12013035, -1.10242011, -1.0857874, -1.07067758, -1.05731738, -1.04578647, ])- E_2H

    E["hf"] = np.zeros_like(E["koloswolniewicz"])
    E["rhf"] = np.zeros_like(E["koloswolniewicz"])
    E["phf"] = np.zeros_like(E["koloswolniewicz"])
    E["prhf"] = np.zeros_like(E["koloswolniewicz"])
    E["puhf"] = np.zeros_like(E["koloswolniewicz"])
    for i, r_ in enumerate(r["koloswolniewicz"]):
        E["hf"][i] = run_hf(r_) - E_2H
        E["rhf"][i] = run_hf(r_, restricted=True) - E_2H
        E["phf"][i] = run_pyscf(r_) - E_2H
        E["prhf"][i] = run_pyscf(r_, restricted=True) - E_2H
        E["puhf"][i] = run_pyscf_uhf(r_) - E_2H

    fig, ax = plt.subplots()

    au_to_Angstrom = 0.529177249
    r["koloswolniewicz"] = r["koloswolniewicz"]/au_to_Angstrom
    ax.plot(r["koloswolniewicz"], E["koloswolniewicz"], label="ex")
    ax.plot(r["koloswolniewicz"], E["hf"], label="hf")
    ax.plot(r["koloswolniewicz"], E["rhf"], label="rhf")
    ax.plot(r["koloswolniewicz"], E["phf"], label="phf", ls="-.")
    ax.plot(r["koloswolniewicz"], E["prhf"], label="prhf", ls="-.")
    ax.plot(r["koloswolniewicz"], E["puhf"], label="puhf", ls="-.")
    ax.hlines(0, *ax.get_xlim(), ls="--", color="k")
    ax.set(xlabel="r [a.u.]", ylabel="E [a.u.]")
    ax.legend()
    plt.show()



if __name__ == "__main__":
    main()

    # run_hf(r=1.1, restricted=False)
    # run_single(atom="H 0 0 0; H 0 0 1.1", basisset="sto-3g")