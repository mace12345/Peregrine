import numpy as np
from pathlib import Path
from copy import deepcopy

from peregrine.molecule import Molecule
from peregrine.atom import Atom

def test_write_PySCF_RHF_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/benzene.mol",
        "r",
    ) as f:
        benzene_str = f.read()
        f.close()
    benzene = Molecule.ReadMolString(benzene_str)
    benzene_pyscf_str = benzene.WritePySCFInput()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/benzene_calc.py",
        "w",
    ) as f:
        f.write(benzene_pyscf_str)
        f.close()

def test_write_PySCF_RKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_cation.mol",
        "r",
    ) as f:
        water_cation_str = f.read()
        f.close()
    water_cation = Molecule.ReadMolString(water_cation_str)
    water_cation.Identifier = "WaterCation"
    water_cation_pyscf_str = water_cation.WritePySCFInput(
        method="wb97m-v",
        basisset="def2svp",
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/water_cation_calc.py",
        "w",
    ) as f:
        f.write(water_cation_pyscf_str)
        f.close()

def test_write_PySCF_ROKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/water_triplet.mol",
        "r",
    ) as f:
        water_triplet_str = f.read()
        f.close()
    water_triplet = Molecule.ReadMolString(water_triplet_str)
    water_triplet.Identifier = "WaterTriplet"
    water_triplet_pyscf_str = water_triplet.WritePySCFInput(
        method="M06-L",
        basisset="def2svp",
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/water_triplet_calc.py",
        "w",
    ) as f:
        f.write(water_triplet_pyscf_str)
        f.close()

def test_write_PySCF_UKS_input():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/InitialTests/BIHGEE-S4_opt0.mol",
        "r",
    ) as f:
        BIHGEE_str = f.read()
        f.close()
    BIHGEE = Molecule.ReadMolString(BIHGEE_str)
    BIHGEE.Identifier = "BIHGEE-S4"
    BIHGEE_pyscf_str = BIHGEE.WritePySCFInput(
        method="r2SCAN",
        basisset="def2svp",
        restricted=False,
        grid_density=3,
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/PySCFTests/BIHGEE-S4_calc.py",
        "w",
    ) as f:
        f.write(BIHGEE_pyscf_str)
        f.close()