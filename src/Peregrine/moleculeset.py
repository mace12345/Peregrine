import numpy as np

from .atom import Atom
from .molecule import Molecule


class MoleculeSet:
    def __init__(self):
        self.MoleculeDict = {}

    def ReadXYZFile(self):
        pass

    def WriteXYZFile(self):
        pass

    def ReadMolFile(self):
        pass

    def WriteMolFile(self):
        main_mol_str = ""
        for Identifier in self.MoleculeDict:
            mol_str = ""
            molObj = self.MoleculeDict[Identifier]
            # Opening Identifier Line, Header block, and blank comment line
            mol_str += f"{Identifier}\nPeregrine Generated .MOL File\n\n"

        pass

    def ReadMol2File(self):
        pass

    def WriteMol2File(self):
        pass
