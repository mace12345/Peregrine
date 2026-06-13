import numpy as np
from pathlib import Path
from copy import deepcopy

from Peregrine.moleculeset import MoleculeSet
from Peregrine.molecule import Molecule
from Peregrine.atom import Atom


def test_molecule_ReadORCA6Output():
    pass


def test_molecule_ReadORCA6OutputGradients():
    molObj = Molecule.ReadORCA6OutputGradients(
        ORCA_output_filepath=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_PBE0-def2-TZVP-Opt-Freq-DEFGRID3/BIHGII-S4.out",
    )
