from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import RHF
from ..CC.CCD import RCCD

# To make tables
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib as pl

def run_vary_R(N=2, omega=1.0, Rs = [1,2,3], run=False):
    Rs = np.array(Rs)

    Ehfs = np.zeros_like(Rs).astype(float)
    Eccds = np.zeros_like(Rs).astype(float)
    Eccdshf = np.zeros_like(Rs).astype(float)
    path = pl.Path(__file__).parent / pl.Path(f"data/dfN{N}")

    if run:
        for i, R in tqdm(enumerate(Rs)):
            ho = HarmonicsOscillator(N=N, R=R, omega=omega, spinrestricted=True)
            ho.calculate_OB()
            ho.calculate_TB()

            rccd = RCCD(ho).run()
            Eccds[i] = rccd.evaluate_energy()

            rhf = RHF(ho).run()
            Ehfs[i] = rhf.evaluate_energy()
            ho = rhf.perform_basis_change(ho)
            
            rccd = RCCD(ho).run()
            Eccdshf[i] = rccd.evaluate_energy()

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
    df = df.style.hide(axis="index").format(precision=4)

    # position_float

    table_df = df.to_latex(hrules=True, position_float="centering", position="H", column_format="ccccc")
    print(table_df)
if __name__ == '__main__':
    Rs = np.arange(1,4)
    run_vary_R(Rs=Rs, run=False)