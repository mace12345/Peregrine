import numpy as np
from .atom import Atom


class Molecule:
    def __init__(
        self,
        Identifier: str,
        AtomsList: list[Atom],
        BondMatrix: np.ndarray | None,
    ):
        """
        Molecule class to represent a molecule
        """
        # Essential Attributes
        self.Identifier = Identifier
        self.AtomsList = AtomsList
        self.BondMatrix = BondMatrix
        if self.BondMatrix is None:
            self.BondMatrix = np.zeros((len(self.AtomsList), len(self.AtomsList)))

        # Derived Basic Attributes
        self.ConnectivityMatrix = np.floor_divide(
            self.BondMatrix,
            self.BondMatrix,
            out=np.zeros_like(self.BondMatrix),
            where=(self.BondMatrix != 0),
        )
        self.AtomsDict = {Atom.Label: Atom for Atom in AtomsList}
        self.NumberOfAtoms = len(self.AtomsList)
        self.NumberOfBonds = int(self.ConnectivityMatrix.sum().sum() / 2)
        # Assign substructure indices based on connectivity
        self.NormaliseSubstructureIndicies()

        # Check Attributes
        if self.BondMatrix.shape != (self.NumberOfAtoms, self.NumberOfAtoms):
            raise ValueError(
                f"bond_matrix shape {self.bond_matrix.shape} does not match number of atoms ({self.NumberOfAtoms})"
            )
        if not np.array_equal(self.BondMatrix, self.BondMatrix.T):
            raise ValueError("bond_matrix must be symmetric")
        if np.any(np.diag(self.BondMatrix) != 0):
            print(self.BondMatrix)
            raise ValueError("bond_matrix must have zero diagonal")
        if len(self.AtomsList) == 0:
            raise ValueError("No atoms in AtomsList")

    def NormaliseSubstructureIndicies(self):
        """
        Assign substructure indices to atoms based on connectivity.
        Connected components are assigned the same substructure index.

        Written by claude haiku 4.5
        """
        visited = np.zeros(self.NumberOfAtoms, dtype=bool)
        substructure_index = 1

        for atom_idx in range(self.NumberOfAtoms):
            if not visited[atom_idx]:
                # Perform depth-first search to find all connected atoms
                self._dfs_assign_substructure(atom_idx, visited, substructure_index)
                substructure_index += 1

        self.NumberOfSubstructures = substructure_index - 1

    def _dfs_assign_substructure(
        self, atom_idx: int, visited: np.ndarray, substructure_index: int
    ):
        """
        Depth-first search to assign substructure index to connected atoms.

        Parameters:
            atom_idx (int): Index of the current atom.
            visited (np.ndarray): Boolean array tracking visited atoms.
            substructure_index (int): The substructure index to assign.

        Written by claude haiku 4.5
        """
        visited[atom_idx] = True
        self.AtomsList[atom_idx].SubstructureIndex = substructure_index

        # Find all atoms connected to the current atom
        connected_atoms = np.where(self.ConnectivityMatrix[atom_idx] != 0)[0]

        for connected_idx in connected_atoms:
            if not visited[connected_idx]:
                self._dfs_assign_substructure(
                    connected_idx, visited, substructure_index
                )
