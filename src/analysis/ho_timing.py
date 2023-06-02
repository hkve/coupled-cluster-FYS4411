from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD, RCCD

import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import pandas as pd
import time

# import sys
# sys.path.append(__file__)
# import plot_utils
import src.analysis.plot_utils as pu

def end(start):
    elapsed = time.time_ns() - start
    return elapsed *1e-6 # ms

def timing(filename="timing", R_max=12, spinrestricted=False):
    n = 10
    N, omega = 2, 1.0
    Rs = np.arange(3,R_max+1)

    T_hf = np.zeros((n, R_max-2), dtype=float)
    T_ccd = np.zeros((n, R_max-2), dtype=float)

    HFS = {True: RHF, False: HF}
    CCDS = {True: RCCD, False: CCD}

    for i, R in enumerate(Rs):
        for j in range(n):
            ho = HarmonicsOscillator(N=N, R=R, omega=omega, spinrestricted=True)
            ho.calculate_OB()
            ho.load_TB("ho1.0.npz")

            if not spinrestricted:
                ho.restricted_to_unrestricted()

            start = time.time_ns()
            hf = HFS[spinrestricted](ho).run()
            T_hf[j,i] = end(start)

            ho = hf.perform_basis_change(ho)

            start = time.time_ns()
            ccf = CCDS[spinrestricted](ho).run()
            T_ccd[j, i] = end(start)
            print(f"Done {j+1+i*len(Rs)}/{len(Rs)*n}")

    if filename:
        filename = f"data/" + filename + spinrestricted*"_restricted"
        filename = pl.Path(__file__).parent / pl.Path(filename)

        np.savez(filename, Rs, T_hf, T_ccd)

def plot_timing(filename="timing"):
    hf, ccd = [0]*2, [0]*2
    shf, sccd = [0]*2, [0]*2
    rs = [0]*2
    filename_urestriced = pl.Path(__file__).parent / pl.Path(f"data/{filename}.npz")
    filename_restriced = pl.Path(__file__).parent / pl.Path(f"data/{filename}_restricted.npz")

    filenames = [filename_urestriced, filename_restriced]
    
    for i in range(2):
        filename = filenames[i]
        npzfile = np.load(filename)

        rs[i] = npzfile["arr_0"]
        hf_time, ccd_time = npzfile["arr_1"], npzfile["arr_2"]

        hf[i] = hf_time.mean(axis=0)
        shf[i] = hf_time.std(axis=0, ddof=1)
        ccd[i] = ccd_time.mean(axis=0)
        sccd[i] = ccd_time.std(axis=0, ddof=1)
    
    hf, rhf = hf
    shf, srhf = shf

    ccd, rccd = ccd 
    sccd, srccd = sccd

    R, Rr = rs

    fig, ax = plt.subplots()
    ax.plot(R, hf, label="HF", marker="o", ms=4)
    ax.plot(R, ccd, label="CCD", marker="s", ms=4)
    ax.plot(Rr, rhf, label="RHF", marker="o", ms=4)
    ax.plot(Rr, rccd, label="RCCD", marker="s", ms=4)

    ax.fill_between(R, hf+shf, hf-shf, alpha=0.3)
    ax.fill_between(R, ccd+sccd, ccd-sccd, alpha=0.3)
    ax.fill_between(Rr, rhf+srhf, rhf-srhf, alpha=0.3)
    ax.fill_between(Rr, rccd+srccd, rccd-srccd, alpha=0.3)

    ax.set_yscale("log")
    ax.set(xlabel="R", ylabel="Time [ms]")
    ax.legend()
    pu.save("timing")
    plt.show()
if __name__ == "__main__":
    # timing(spinrestricted=False, R_max = 9)
    # timing(spinrestricted=True, R_max = 12)

    plot_timing()