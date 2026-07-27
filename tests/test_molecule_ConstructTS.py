import numpy as np
from pathlib import Path
from copy import deepcopy

from peregrine.molecule import Molecule
from peregrine.atom import Atom

xtb_binary_path = "C:/Users/samue/Desktop/xtb-bleed-windows/bin/"


def test_molecule_ConstructTS_diels_alder():
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_Reac.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj_Reac = Molecule.ReadMolString(molObj_str)
    molObj_Reac.AtomsList[20].FormalCharge = -1
    molObj_Reac.AtomsList[21].FormalCharge = 1
    molObj_Reac.ChangeBond(
        NewBondOrder=2,
        AtomIndices=[20, 21],
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_TS.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj_TS = Molecule.ReadMolString(molObj_str)
    molObj_TS.AtomsList[20].FormalCharge = -1
    molObj_TS.AtomsList[21].FormalCharge = 1
    molObj_TS.ChangeBond(
        NewBondOrder=2,
        AtomIndices=[20, 21],
    )
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/TS/BH9/Pericyclic32_Prod.mol",
        "r",
    ) as f:
        molObj_str = f.read()
        f.close()
    molObj_Prod = Molecule.ReadMolString(molObj_str)

    print(molObj_Reac.WriteSMARTSString())
    print(molObj_TS.WriteSMARTSString())
    print(molObj_Prod.WriteSMARTSString())
