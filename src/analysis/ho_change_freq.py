from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import RHF
from ..CC.CCD import RCCD
import src.analysis.plot_utils as pu

import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt

def p_picker(omega, N):
    p = 0

    if N in [2, 6] and omega <= 0.1:
        p = 0.5
    
    if N == 12 and omega <= 0.4:
        p = 0.5
    
    if N == 20:
        p = 0.5

    return p


def run_vary_omega(N=2, R=12, calculate=False):
    n_omega = 50
    omegas = np.linspace(1,0.9, n_omega)
    ho = HarmonicsOscillator(R=R, N=N, spinrestricted=True)
    ho.calculate_OB()
    if calculate:
        ho.calculate_TB()
    else:
        ho.load_TB("ho1.0.npz")

    E_per_particle = np.full(omegas.shape, np.nan)
    E_per_particle_noninteract = np.full(omegas.shape, np.nan)

    for i, omega in enumerate(omegas):
        ho.change_frequency(omega)

        E_per_particle_noninteract[i] = 2*np.trace(ho.h[:N//2, :N//2])
        
        hf = RHF(ho).run()
        ho = hf.perform_basis_change(ho)

        p = p_picker(omega, N)
        ccd = RCCD(ho).run(p=p)
        
        if ccd.converged:
            E_per_particle[i] = ccd.evaluate_energy()
        else:
            E_per_particle_noninteract[i] = np.nan
            print(f"Convergence broke at {omega = :.2f}")
            break

        ho = hf.perform_basis_change(ho, inverse=True)

        print(f"Done {omega = :.3f}, {N = }, E = {E_per_particle[i]:.4f}, E_ID = {E_per_particle_noninteract[i]:.4f}")
    return omegas[::-1], E_per_particle[::-1], E_per_particle_noninteract[::-1]


def plot_vary_omega(filename="vary_omega", run=True):
    Ns = [2, 6, 12, 20]

    path = pl.Path(__file__).parent / pl.Path(f"data")


    fig, ax = plt.subplots()
    data = {}
    if not run:
        for N in Ns:
            full_path = path / pl.Path(filename + f"{N}.npz")
            npzfile = np.load(full_path)
            data[N] = {
                "omega": npzfile[f"arr_0"], "ratio": npzfile[f"arr_1"]
            }


    for N in Ns:
        if run:
            omega, E_per_particle, E_per_particle_noninteract = run_vary_omega(N=N)
            ratio = E_per_particle / E_per_particle_noninteract
            full_path = path / pl.Path(filename + f"{N}")
            np.savez(full_path, omega, ratio)
        else:
            omega, ratio = data[N]["omega"], data[N]["ratio"]

        ax.plot(omega, ratio, label=f"{N = }")

    ax.set(xlabel=r"$\omega$", ylabel="$E_{CCD}/E_{NI}$")
    ax.legend()
    pu.save(filename)
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
    plot_vary_omega(run=False)