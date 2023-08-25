from ..basis.HarmonicsOscillator import HarmonicsOscillator
from ..HF.HF import HF, RHF
from ..CC.CCD import CCD, RCCD
from ..CC.fastCCD import fastCCD

from ..CC.CCSD import CCSD
from ..CC.fastCCSD import fastCCSD

import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import pandas as pd
import time

N, R = 2, 7
vocal = False

def end(start):
    elapsed = time.time_ns() - start
    return elapsed *1e-6 # ms

omega = 1.0
ho = HarmonicsOscillator(N=N, R=R, omega=omega, spinrestricted=True)
ho.calculate_OB()
ho.load_TB("ho1.0.npz")


ho.restricted_to_unrestricted()

start = time.time_ns()
hf = HF(ho).run()
t = end(start)
print(f"HF {t = } ms")

ho = hf.perform_basis_change(ho)

# start = time.time_ns()
# ccd = CCD(ho).run()
# t = end(start)
# print(f"CCD {t = } ms")
# print(ccd.evaluate_energy())

# start = time.time_ns()
# ccd = fastCCD(ho).run()
# t = end(start)
# print(f"fastCCD {t = } ms")
# print(ccd.evaluate_energy())

start = time.time_ns()
ccsd = CCSD(ho).run(vocal=vocal)
t = end(start)
print("")
print(f"CCSD {t = } ms")
print(f"CCSD {ccsd.evaluate_energy()}")
print("")

start = time.time_ns()
ccsd = fastCCSD(ho).run(vocal=vocal)
t = end(start)

print("")
print(f"fastCCSD {t = } ms")
print(f"fastCCSD {ccsd.evaluate_energy()}")
print("")

import coupled_cluster.ccsd as CCHyQD
from coupled_cluster.mix import AlphaMixer, DIIS
from quantum_systems import GeneralOrbitalSystem, BasisSet

basis = BasisSet(l=ho.L_, dim=3, anti_symmetrized_u=True, includes_spin=True)
basis.h = ho.h
basis.u = ho.v

sys = GeneralOrbitalSystem(n=N, basis_set=basis)

start = time.time_ns()
meth = CCHyQD.CCSD(sys, verbose=vocal, mixer=AlphaMixer)
E_HyQD = meth.compute_energy()
t = end(start)
print("")
print(f"HyQD CCSD {t = } ms")
print(f"HyQD CCSD = {E_HyQD}")
print("")

from IPython import embed
embed()