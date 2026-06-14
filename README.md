# Peregrine

__Project is under construction__

Software for building transition states in a high-throuput fashion. Input TS geometry and SMILES => Output TS structures. Enables reactivity screening and building datasets for MLPs and AI/ML applications.

# The big idea proposition in 30 seconds

Chemist needs to optimise reaction for an industrial process. They will need to explore 100s of possibilities to build a map get to a final solution, a very resource intensive process when done in the laboratory.

Building that same map can be done computationally, using a fraction of the resources versus map making in the laboratory.

However, experimental chemists typically lack the programming and computational expertise to build maps computationally, this is where our no-code software idea comes in.


# Installing Dependancies for chem-env, the python environment for Peregrine
```
conda env create -f environment.yml --name chem-env
```
Python Version 3.12 and all packages are downloaded within the conda environment that must be called "chem-env". All packages are managed with __Miniforge__, works well across different platforms and has BSD 3-clause license, handy for commercial use. 

## Chemistry Related Dependencies
__RDKit__ is a critical package that would be very hard to replace. Very good at handling if `smiles` and `SMARTS` strings, machine readable information for chemical structures. Has atom mapping caperbilities, heavily utilised in this project by mapping `SMARTS` that is used build transition states with the `Molecule` class.

__Openbabel__ has extreamly useful functionality for assigning 3D coordinates to `smiles` strings and thus converting them to `.xyz` files or our prefered file `.mol2` files. `.mol2` files are very useful sincy they contain not only the cartesian coordinates of the atoms but also the bonding, and atom properties information. Openbabel has optimising functions to optimise molecules with the Universal Force Field (UFF).

__xyzgraph__ and __RCgraph__ are excellent packages developed by Alister Goodfellow, really handy for characterising transtion states and assigning bonds to XYZ coordinates.

__ASE__ (Atomic Simulation Environment) Can be thought of as an adapter for linking packages that calculate molecular properties to interface with python. For example, it can interface with xTB to optimise molecules, or maybe a Machine Learning Potential (MLP). It would be faster to interface with xTB direct in the command line but ASE can do constrained optimisations and Nudge Elastic Band (NEB) calculations to find Transition States (TS).

## Python Classics
__Numpy__, __Pandas__, and __Matplotlib__ are the holy trinity of basic scientific python programming. Numpy and pandas are by far the most useful, matplotlib is occasionally used to visulise results and trying to understand certain problems better but it is not integral for the project at all. But I like to have it installed just incase.

## Add ons that may or may not be useful in the final product
__Scipy__ and __scikit-learn__

## GUI Related Dependencies
__PyVista__ an excellent toolkit for 3D modelling of out molecules. __PyQt5__ Don't quite know how to install the exact version but it should not matter to much. __VTK__ (Visulisation Tool Kit).

## Notes on PySCF
PySCF is a python package that can run DFT calculations and optimisations. It will be used to generate the training data for the Machine Learning Potential (MLP) and inital guesses for SCF convergence. It is an apache license so we can use it commercially to. PySCF is very customisable so that means we can incorperate our machine learning models into it to massively speed up these calculations, this will give us a huge edge commericaly. PySCF is currently not included in as it is unable to run on windows and is not essiental for the workflows being built on it. It runs on linux and will be used in a distrabutive computing fashion.


# Unit Testing with PyTest
To test project, call `pytest` in the root directory of `Project`. All testing input and outputs are saved in the folder `Project/tests`.

```
conda install conda-forge::pytest -y
