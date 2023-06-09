from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import RHF
from ..CC.CCD import RCCD
import src.analysis.plot_utils as pl

import numpy as np
import matplotlib.pyplot as plt

def run_vary_omega(N=2, R=12, calculate=False):
    n_omega = 5
    omegas = np.linspace(0.5, 1, n_omega)
    ho = HarmonicsOscillator(R=R, N=N, spinrestricted=True)
    ho.calculate_OB()
    if calculate:
        ho.calculate_TB()
    else:
        ho.load_TB("ho1.0.npz")

    E_per_particle = np.zeros_like(omegas)
    E_per_particle_noninteract = np.zeros_like(omegas)

    for i, omega in enumerate(omegas):
        ho.change_frequency(omega)
        hf = RHF(ho).run()
        ho = hf.perform_basis_change(ho)

        ccd = RCCD(ho).run()
        E_per_particle[i] = ccd.evaluate_energy()/N
        E_per_particle_noninteract[i] = 2*np.trace(ho.h[:N//2, :N//2])/N

        ho = hf.perform_basis_change(ho, inverse=True)

        print(f"Done {omega:.2f}, {N = }, E = {E_per_particle[i]}")
    return omegas, E_per_particle, E_per_particle_noninteract


def plot_vary_omega():
    Ns = [2, 6]

    fig, ax = plt.subplots()
    for N in Ns:
        omega, E_per_particle, E_per_particle_noninteract = run_vary_omega(N=N)
        print(E_per_particle)
        ax.plot(omega, E_per_particle, marker="o", label=f"{N = }")
        ax.plot(omega, E_per_particle_noninteract, ls="--", alpha=0.5)

    ax.set(xlabel=r"$\omega$", ylabel="$E_0$ [a.u.]")
    ax.legend()
    plt.show()

def test_inv():
    ho = HarmonicsOscillator(R=4, N=2, spinrestricted=True)
    ho.calculate_OB()
    ho.calculate_TB()
    h_old = ho.h.copy()

    hf = RHF(ho).run()

    ho = hf.perform_basis_change(ho, inverse=True)

    np.testing.assert_almost_equal(h_old, ho.h)


if __name__ == '__main__':
    plot_vary_omega()