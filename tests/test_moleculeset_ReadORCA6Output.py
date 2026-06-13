import numpy as np
from pathlib import Path
from copy import deepcopy

from Peregrine.moleculeset import MoleculeSet
from Peregrine.molecule import Molecule
from Peregrine.atom import Atom


def test_molecule_ReadORCA6Output():
    pass


def test_molecule_ReadORCA6OutputGradients():
    molObj_list = Molecule.ReadORCA6OutputGradients(
        ORCA_output_filepath=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_PBE0-def2-TZVP-Opt-Freq-DEFGRID3/BIHGII-S4.out",
    )
    # Test first atom of first molecule object and molecule energies
    atomObj = molObj_list[0].AtomsList[0]
    assert round(atomObj.Gradient[0], 5) == -0.00087
    assert round(atomObj.Gradient[1], 5) == 0.00208
    assert round(atomObj.Gradient[2], 5) == 0.00307
    assert round(atomObj.Coordinates[0], 5) == -0.63470
    assert round(atomObj.Coordinates[1], 5) == 2.86860
    assert round(atomObj.Coordinates[2], 5) == 2.27080
    # Test last atom of last molecule object and molecule energies
    atomObj = molObj_list[-1].AtomsList[-1]
    assert atomObj.Gradient[0] == 0.000012555
    assert atomObj.Gradient[1] == 0.000025513
    assert atomObj.Gradient[2] == -0.000005766
    assert round(atomObj.Coordinates[0], 5) == 4.48027
    assert round(atomObj.Coordinates[1], 5) == 4.10131
    assert round(atomObj.Coordinates[2], 5) == 6.88683
