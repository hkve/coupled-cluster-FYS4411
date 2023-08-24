from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD, RCCD
from ..CC.fastCCD import fastCCD

import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import pandas as pd
import time

def end(start):
    elapsed = time.time_ns() - start
    return elapsed *1e-6 # ms

omega = 1.0
ho = HarmonicsOscillator(N=6, R=7, omega=omega, spinrestricted=True)
ho.calculate_OB()
ho.load_TB("ho1.0.npz")


ho.restricted_to_unrestricted()

start = time.time_ns()
hf = HF(ho).run()
t = end(start)
print(f"HF {t = } ms")

ho = hf.perform_basis_change(ho)

start = time.time_ns()
ccd = CCD(ho).run()
t = end(start)
print(f"CCD {t = } ms")

print(ccd.evaluate_energy(), t)

start = time.time_ns()
ccd = fastCCD(ho).run()
t = end(start)
print(f"CCD {t = } ms")

print(ccd.evaluate_energy())