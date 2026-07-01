import numpy as np
from pathlib import Path
from copy import deepcopy

from peregrine.molecule import Molecule
from peregrine.atom import Atom

def test_write_PySCF_input():
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