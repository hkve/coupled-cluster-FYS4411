from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import RHF
from ..CC.CCD import RCCD

# To make tables
import numpy as np
import pandas as pd
import pathlib as pl
import re

import time

def end(start, string):
    elapsed = time.time() - start
    print(f"{string}, T = {elapsed:.2f} seconds")

def calculate_matrix_elements(Rs, omega, filename):
    R = np.max(Rs)
    ho = HarmonicsOscillator(N=2, R=R, omega=omega, spinrestricted=True)
    if ho.get_setspath_filename(f"{filename}.npz").exists():
        return None
    
    print(f"Running {omega = } matrix element calculation for {R = }")
    start = time.time()
    ho.calculate_TB()
    end(start, "Calculations of TB")
    
    start = time.time()
    ho.save_TB(filename)
    end(start, "Saving of TB")

def run_vary_R(N=2, omega=1.0, Rs = [1,2,3], p_ho={}, p_hf={}, run=False, filename=None):
    Rs = np.array(Rs)

    Ehfs = np.zeros_like(Rs).astype(float)
    Eccds = np.zeros_like(Rs).astype(float)
    Eccdshf = np.zeros_like(Rs).astype(float)
    path = pl.Path(__file__).parent / pl.Path(f"data/df_N{N}_omega{omega:.1f}")

    ps_ho, ps_hf = {R: 0 for R in Rs}, {R: 0 for R in Rs}
    ps_ho.update(p_ho)
    ps_hf.update(p_hf)

    ho_converged = True
    hf_converged = True
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

            if not ho_converged:
                Eccds[i] = 0
            else:    
                start = time.time()
                rccd = RCCD(ho).run(p=ps_ho[R])
                Eccds[i] = rccd.evaluate_energy()
                end(start, f"CCD(HO) calculation, p  = {ps_ho[R]}")
                ho_converged = rccd.converged

            start = time.time()
            rhf = RHF(ho).run()
            Ehfs[i] = rhf.evaluate_energy()
            end(start, "HF calculation")
 
            start = time.time()
            ho = rhf.perform_basis_change(ho)
            end(start, "Perform basis change")

            if not hf_converged:
                Eccdshf[i] = 0
            else:
                start = time.time()
                rccd = RCCD(ho).run(p=ps_hf[R])
                Eccdshf[i] = rccd.evaluate_energy()
                end(start, f"CCD(HF) calculation, p  = {ps_hf[R]}")
                hf_converged = rccd.converged

            print(f"Done {Rs[i]}/{Rs[-1]}\n")

        mixings = {}
        if sum(list(ps_ho.values())) != 0:
            mixings.update({r"$p$": list(ps_ho.values())})
        if sum(list(ps_hf.values())) != 0:
            mixings.update({r"$p(HF)$": list(ps_hf.values())})

        df = pd.DataFrame({
            "$R$": Rs,
            **mixings,
            r"HF": Ehfs,
            "CCD": Eccds,
            "CCD(HF)": Eccdshf
        })

        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    
    def format_p(x):
        if x > 0.99:
            return r"$>$0.99"
        else:
            return f"{x:.2f}"
    
    extra_cols = 0
    if "$p$" in df.columns:
        df["$p$"] = df["$p$"].apply(format_p)
        extra_cols += 1
    if "$p(HF)$" in df.columns:
        df["$p(HF)$"] = df["$p(HF)$"].apply(format_p)
        extra_cols += 1

    df = df.style.hide(axis="index").format(precision=6)

    coulmn_format = "cccc" + "c"*extra_cols
    table_df = df.to_latex(hrules=True, position_float="centering", position="H", column_format=coulmn_format, caption=rf"${N = }, \omega = {omega}$")
    
    table_df = re.sub(r"(0.000000)", "-", table_df)
    table_df = re.sub(r"(0.00)", "N", table_df)
    

    print(table_df)

def run_machinery(omega, Ns, ps_ho, ps_hf, runs):
    for i, N in enumerate(Ns):
        filename = f"ho{omega}"
        Rs = np.arange(i+1,12+1)
        calculate_matrix_elements(Rs, omega=omega, filename=filename)
        run_vary_R(N=N, omega=omega, Rs=Rs, run=runs[N], p_ho=ps_ho[N], p_hf=ps_hf[N], filename=f"{filename}.npz")

if __name__ == '__main__':
    # Normal frequency
    omega = 1.0
    Ns = [2,6,12,20]
    ps_ho = {
        2: {},
        6: {9: 0.3, 10: 0.3, 11: 0.3, 12: 0.3},
        12:{R: 0.999 for R in np.arange(3,12+1)},
        20:{R: 0.999 for R in np.arange(4,12+1)},
    }
    ps_hf = {
        N: {} for N in Ns
    }
    runs = {
        2: False,
        6: False,
        12: False,
        20: False,
    }
    run_machinery(omega, Ns, ps_ho, ps_hf, runs)

    # A little lower
    omega = 0.5
    Ns = [2,6,12,20]
    ps_ho = {
        2: {},
        6: {i: 0.9999 for i in range(5,12+1)},
        12: {i: 0.9999 for i in range(3,12+1)},
        20: {i: 0.9999 for i in range(4,12+1)},
    }
    ps_hf = {
        2: {},
        6: {},
        12: {},
        20: {5: 0.2, 6: 0.2, 7: 0.999, 8: 0.999, 9: 0.999, 10: 0.999, 11: 0.999, 12: 0.999},
    }
    runs = {
        2: False,
        6: False,
        12: False,
        20: False,
    }
    # run_machinery(omega, Ns, ps_ho, ps_hf, runs)
