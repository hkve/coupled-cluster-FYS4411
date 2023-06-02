from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import RHF
from ..CC.CCD import RCCD
import src.analysis.plot_utils as pl

import numpy as np
import matplotlib.pyplot as plt

def run_vary_omega():
    n_omega = 5
    omegas = np.linspace(0.5, 1, n_omega)
    N  = 2
    ho = HarmonicsOscillator(R=4, N=N, spinrestricted=True)
    ho.calculate_OB()
    ho.calculate_TB()

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

    return omegas, E_per_particle, E_per_particle_noninteract


def plot_vary_omega(omega, E_per_particle, E_per_particle_noninteract):

    fig, ax = plt.subplots()
    ax.plot(omega, E_per_particle, marker="o")
    ax.plot(omega, E_per_particle_noninteract, color="gray", ls="--")

    ax.set(xlabel=r"$\omega$", ylabel="$E_0$ [a.u.]")
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
    omega, E_per_particle, E_per_particle_noninteract = run_vary_omega()
    plot_vary_omega(omega, E_per_particle, E_per_particle_noninteract)