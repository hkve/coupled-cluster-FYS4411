from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import RHF
from ..CC.CCD import RCCD

# To make tables
import numpy as np
import pandas as pd
import pathlib as pl

import time

def end(start, string):
    elapsed = time.time() - start
    print(f"{string}, T = {elapsed:.2f} seconds")

def calculate_matrix_elements(Rs, omega, filename):
    R = np.max(Rs)
    ho = HarmonicsOscillator(N=2, R=R, omega=omega, spinrestricted=True)
    if ho.get_setspath_filename(f"{filename}.npz").exists():
        return None
    
    start = time.time()
    ho.calculate_TB()
    end(start, "Calculations of TB")
    
    start = time.time()
    ho.save_TB(filename)
    end(start, "Saving of TB")

def run_vary_R(N=2, omega=1.0, Rs = [1,2,3], run=False, filename=None):
    Rs = np.array(Rs)

    Ehfs = np.zeros_like(Rs).astype(float)
    Eccds = np.zeros_like(Rs).astype(float)
    Eccdshf = np.zeros_like(Rs).astype(float)
    path = pl.Path(__file__).parent / pl.Path(f"data/dfN{N}")

    if run:
        n_Rs = len(Rs)
        for i, R in enumerate(Rs):
            start = time.time()
            ho = HarmonicsOscillator(N=N, R=R, omega=omega, spinrestricted=True)
            ho.calculate_OB()
            if filename:
                ho.load_TB(filename)
            else:
                ho.calculate_TB()

            end(start, "Setup basis")

            start = time.time()
            rccd = RCCD(ho).run()
            Eccds[i] = rccd.evaluate_energy()
            end(start, "CCD(HO) calculation")

            start = time.time()
            rhf = RHF(ho).run()
            Ehfs[i] = rhf.evaluate_energy()
            end(start, "HF calculation")
 
            start = time.time()
            ho = rhf.perform_basis_change(ho)
            end(start, "Perform basis change")

            start = time.time()
            rccd = RCCD(ho).run()
            Eccdshf[i] = rccd.evaluate_energy()
            end(start, "CCD(HF) calculation")

            print(f"Done {i+1}/{n_Rs}\n")

        df = pd.DataFrame({
            "R": Rs,
            r"$\omega$": omega,
            r"HF": Ehfs,
            "CCD": Eccds,
            "CCD(HF)": Eccdshf
        })

        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    
    df[r"$\omega$"] = df[r"$\omega$"].apply(lambda x: f"{x:.1f}")
    df = df.style.hide(axis="index").format(precision=6)

    # position_float

    table_df = df.to_latex(hrules=True, position_float="centering", position="H", column_format="ccccc")
    print(table_df)
if __name__ == '__main__':

    omega = 1.0
    filename = f"ho{omega}"
    Rs = np.arange(1,12+1)
    calculate_matrix_elements(Rs, omega=1.0, filename=filename)
    run_vary_R(Rs=Rs, run=True, filename=f"{filename}.npz")