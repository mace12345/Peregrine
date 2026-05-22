import numpy as np
from .atom import Atom


class Molecule:
    def __init__(
        self,
        Identifier: str,
        AtomsList: list[Atom],
        BondMatrix: np.ndarray,
    ):
        """
        Molecule class to represent a molecule
        """
        # Essential Attributes
        self.Identifier = Identifier
        self.AtomsList = AtomsList
        self.BondMatrix = BondMatrix

        # Derived Attributes
        self.ConnectivityMatrix = np.floor_divide(
            self.BondMatrix,
            self.BondMatrix,
            out=np.zeros_like(self.BondMatrix),
            where=(self.BondMatrix != 0),
        )
        self.AtomsDict = {Atom.Label: Atom for Atom in AtomsList}
        self.NumberOfAtoms = len(self.AtomsList)

        # Check Attributes
        if self.BondMatrix.shape != (self.NumberOfAtoms, self.NumberOfAtoms):
            raise ValueError(
                f"bond_matrix shape {self.bond_matrix.shape} does not match number of atoms ({self.NumberOfAtoms})"
            )
        if not np.array_equal(self.BondMatrix, self.BondMatrix.T):
            raise ValueError("bond_matrix must be symmetric")
        if not np.any(np.diag(self.BondMatrix) != 0):
            raise ValueError("bond_matrix must have zero diagonal")
        if len(self.AtomsList) == 0:
            raise ValueError("No atoms in AtomsList")
