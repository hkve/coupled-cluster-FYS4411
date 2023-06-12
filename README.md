# Coupled Cluster Doubles for the closed shell 2D Harmonic Oscillator

This repository contains code developed for the second project in the course FYS4411 – Computational Physics II: Quantum Mechanical Systems. An implementation of Hartree-Fock (HF) and Coupled Cluster using doubles excitations (CCD) was made, in both an unrestricted and restricted scheme. These were applied with s-type hydrogen orbitals for basic calculations on Helium and Beryllium, in addition to closed shell calculations on the two-dimensional Harmonic Oscillator.

## Requirements
To run the basics of the HF and CCD calculations, the following packages are required
- `matplotlib>=3.5.1`
- `numpy>=1.21.5`
- `pandas>=1.4.4`
- `scipy>=1.8.0`
- `seaborn>=0.12.2`

The Harmonic Oscillator two-body matrix elements were calculated using [Quantum systems](https://github.com/HyQD/quantum-systems) 
- `quantum_systems>=0.2.6`

To run calculations of Hydrogen matrix elements and the CCD equations, a working implementation of [Drudge](https://github.com/tschijnmo/drudge) is required. 

## Usage and Structure
The `Python` code is structured like a package, where you can run specific source files from the main directory following

```Bash
python3 -m src.path.to.file.filename
```
The `src` directory contains multiple subdirectories

- `basis`: A basic class structure to handle one and two-body matrix elements in both restricted and unrestricted schemes. Derived classes are for specific systems, such as present in `Hydrogen.py` and `HarmonicOscillator.py`. 
- `HF`: Contains a Hartree-Fock base class, in addition to derived classes for restricted and unrestricted schemes.
- `CC`: Contains a Coupled Cluster base class, in addition to derived classes for restricted and unrestricted schemes for doubles excitations.
- `analysis`: Where plots, tables and other analysis for the article is performed. 
- `tests`: Runs test for the aforementioned code.
- `drudge`: Contains Drudge scripts used for calculating CCD equations in restricted and unrestricted schemes.
- `notebooks`: Contains a single notebook used for calculating Hydrogen matrix elements.
- `cpputils`: Deprecated C++ code which was mixed into python using the [PyBind11](https://github.com/pybind/pybind11) module. This was removed due to clever NumPy tricks making it redundant.  
