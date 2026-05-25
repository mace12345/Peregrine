import numpy as np
from typing import Self
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import rdkit.Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import SanitizeMol, SanitizeFlags
from rdkit import RDLogger
import rdkit

from .atom import Atom, ATOMIC_MASSES


class Molecule:
    def __init__(
        self,
        Identifier: str,
        AtomsList: list[Atom],
        BondOrderMatrix: np.ndarray | None,
    ):
        """
        Initialize a Molecule instance.

        Creates a new Molecule object with the given atoms and bonding information.
        Automatically derives connectivity information, assigns substructure indices,
        and validates the integrity of the molecular structure.

        Parameters:
            Identifier (str): A unique identifier or name for the molecule
                             (e.g., "Water", "Benzene", "SMILES_string").
            AtomsList (list[Atom]): A list of Atom objects representing all atoms
                                    in the molecule. Must not be empty.
            BondOrderMatrix (np.ndarray | None): An (n x n) symmetric matrix where
                                                 element [i,j] represents the bond
                                                 order between atoms i and j.
                                                 - Pass None to create an empty
                                                   molecule with no bonds.
                                                 - Diagonal must be zero (no self-bonds).
                                                 - Must be symmetric.

        Attributes (automatically derived):
            ConnectivityMatrix (np.ndarray): Binary connectivity matrix (1 if bonded,
                                            0 otherwise).
            AtomsDict (dict): Mapping of atom labels to Atom objects for quick lookup.
            NumberOfAtoms (int): Total number of atoms.
            NumberOfBonds (int): Total number of bonds (counted from connectivity).
            NumberOfSubstructures (int): Number of disconnected molecular fragments.
            FormalCharge (int): Total formal charge of the molecule.
            Multiplicity (int): Spin multiplicity (2S+1) of the molecule.

        Raises:
            ValueError: If BondOrderMatrix shape doesn't match NumberOfAtoms.
            ValueError: If BondOrderMatrix is not symmetric.
            ValueError: If BondOrderMatrix has non-zero diagonal.
            ValueError: If AtomsList is empty.

        Examples:
            # Water molecule (disconnected atoms)
            mol = Molecule(
                Identifier="Water",
                AtomsList=[
                    Atom("H1", "H", np.array([0, 0, 0])),
                    Atom("O1", "O", np.array([1, 0, 0])),
                    Atom("H2", "H", np.array([2, 0, 0])),
                ],
                BondOrderMatrix=None  # No bonds
            )

            # Water molecule with bonds
            bonds = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]
            ])
            mol = Molecule("Water", atoms, bonds)

        Notes:
            - Substructure indices are automatically assigned based on connectivity.
            - If BondOrderMatrix is None, a zero matrix is created.
            - Formal charges and multiplicities should be set on individual atoms.
        """
        # Essential Attributes
        self.Identifier = Identifier
        self.AtomsList = AtomsList
        self.BondOrderMatrix = BondOrderMatrix
        if self.BondOrderMatrix is None:
            self.BondOrderMatrix = np.zeros((len(self.AtomsList), len(self.AtomsList)))

        # Derived Basic Attributes
        self.DeriveBasicAttributes()

        # Check Attributes
        if self.BondOrderMatrix.shape != (self.NumberOfAtoms, self.NumberOfAtoms):
            raise ValueError(
                f"bond_matrix shape {self.bond_matrix.shape} does not match number of atoms ({self.NumberOfAtoms})"
            )
        if not np.array_equal(self.BondOrderMatrix, self.BondOrderMatrix.T):
            raise ValueError("bond_matrix must be symmetric")
        if np.any(np.diag(self.BondOrderMatrix) != 0):
            print(self.BondOrderMatrix)
            raise ValueError("bond_matrix must have zero diagonal")
        if len(self.AtomsList) == 0:
            raise ValueError("No atoms in AtomsList")

    def DeriveBasicAttributes(self):
        """
        Derive and calculate basic molecular attributes from bond order matrix and atom list.

        This method computes several fundamental properties of the molecule based on the
        bonding information (BondOrderMatrix) and atomic composition (AtomsList). It is
        automatically called during Molecule initialization.

        Derived Attributes:
            ConnectivityMatrix (np.ndarray): Binary connectivity matrix where element [i,j]
                                            is 1 if atoms i and j are bonded, 0 otherwise.
            AtomsDict (dict): Dictionary mapping atom labels to Atom objects for O(1) lookup.
            NumberOfAtoms (int): Total count of atoms in the molecule.
            NumberOfBonds (int): Total count of unique bonds (counts each bond once).
            FormalCharge (int): Sum of formal charges across all atoms.
            Multiplicity (int): Total spin multiplicity (2S+1) of the molecule.
            NumberOfSubstructures (int): Number of disconnected molecular fragments
                                        (assigned by NormaliseSubstructureIndicies).

        Returns:
            None (modifies instance attributes in-place)

        Notes:
            - ConnectivityMatrix is derived from BondOrderMatrix by converting all non-zero
              bond orders to 1.
            - NumberOfBonds is calculated as half the sum of ConnectivityMatrix elements
              (since the matrix is symmetric).
            - Substructure indices are assigned based on connectivity using depth-first search.

        See Also:
            NormaliseSubstructureIndicies(): Assigns substructure indices to atoms.
            GetFormalCharge(): Calculates total formal charge.
            GetMultiplicity(): Calculates total spin multiplicity.
        """
        self.ConnectivityMatrix = np.floor_divide(
            self.BondOrderMatrix,
            self.BondOrderMatrix,
            out=np.zeros_like(self.BondOrderMatrix),
            where=(self.BondOrderMatrix != 0),
        )
        self.NumberOfBonds = int(self.ConnectivityMatrix.sum().sum() / 2)
        self.NormaliseAtomLabels()
        self.AtomsDict = {
            Atom.Label: [idx, Atom] for idx, Atom in enumerate(self.AtomsList)
        }
        self.NumberOfAtoms = len(self.AtomsList)
        self.GetFormalCharge()
        self.GetMultiplicity()
        self.NormaliseSubstructureIndicies()

    def NormaliseAtomLabels(self):
        """
        Normalize atom labels to a standard format and calculate molecular mass.

        Renames all atoms in the molecule to follow a consistent labeling convention:
        ElementSymbol + Count (e.g., H1, H2, C1, O1, N1). Atoms of the same element
        are numbered sequentially starting from 1. Also calculates the total molecular
        mass by summing atomic masses.

        This method is automatically called during Molecule initialization via
        DeriveBasicAttributes().

        Derived Attributes:
            MolecularMass (float): Total molecular mass in amu, calculated as the sum
                                  of all atomic masses. Updated each time this method
                                  is called.

        Modified Attributes:
            Each Atom.Label is updated to the normalized format (e.g., "H1", "C2", "O1").

        Returns:
            None (modifies instance and Atom objects in-place)

        Examples:
            Before normalization: atoms might have arbitrary labels like "Atom_1", "H_alpha"
            After normalization: labels become H1, C1, O1, etc.

            For ethane (C2H6):
            - C atoms get labeled: C1, C2
            - H atoms get labeled: H1, H2, H3, H4, H5, H6
            - MolecularMass = 2 * 12.011 + 6 * 1.008 = 30.070 amu

        Notes:
            - Labels are generated based on atomic symbol and sequential count,
              not on substructure or connectivity.
            - This method should be called after all atoms are added to the molecule.
            - Running this method multiple times will reset all labels and recalculate mass.
            - The count for each element restarts from 1 (not cumulative across elements).

        See Also:
            DeriveBasicAttributes(): Calls this method along with other initialization methods.
        """
        atomic_symbol_count_dict = {}
        self.MolecularMass = 0
        for atomObj in self.AtomsList:
            if atomObj.AtomicSymbol not in atomic_symbol_count_dict:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] = 1
                atomObj.Label = f"{atomObj.AtomicSymbol}1"
            else:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] += 1
                atomObj.Label = f"{atomObj.AtomicSymbol}{atomic_symbol_count_dict[atomObj.AtomicSymbol]}"
            self.MolecularMass += ATOMIC_MASSES[atomObj.AtomicSymbol]
        self.MolecularMass = round(self.MolecularMass, 2)

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

    def GetFormalCharge(self):
        self.FormalCharge = 0
        for atomObj in self.AtomsList:
            self.FormalCharge += atomObj.FormalCharge

    def GetMultiplicity(self):
        unpaired_electrons = 0
        for atomObj in self.AtomsList:
            if atomObj.Multiplicity == 1:
                continue
            else:
                unpaired_electrons += (atomObj.Multiplicity - 1) / 2
        self.Multiplicity = (2 * unpaired_electrons) + 1

    def WriteMolString(self):
        """
        Generate a .MOL file string in V3000 format.

        The V3000 format is a modern, extended version of the .MOL file format used to represent
        molecular structures. This method constructs a complete MOL file as a single string
        containing:
        - Molecule identifier and header information
        - Atom block with element symbols, 3D coordinates, formal charges, and multiplicities
        - Bond block with bond orders and connectivity information
        - Substructure information

        Parameters:
            None

        Returns:
            str: A complete V3000 .MOL file string representation of the molecule,
                 including all atoms, bonds, and molecular properties. Can be written
                 directly to a .mol file.

        Examples:
            mol_str = molecule.WriteMolString()
            with open("molecule.mol", "w") as f:
                f.write(mol_str)

        Notes:
            - Bond order 1.5 is represented as aromatic bonds (type 4 in V3000)
            - Formal charges and multiplicities are only included if non-zero
            - Coordinates should be in Angstroms
            - The method assumes all atoms and bond information are already set
        """
        mol_str = ""
        # Opening Identifier Line, Header block, and blank comment line
        mol_str += f"{self.Identifier}\nPeregrine Generated .MOL File\n\n"
        # CTAB begin block, counts line, and begin atoms line
        mol_str += f" 0 0 0 0 0 999 V3000\nM V30 BEGIN CTAB\nM V30 COUNTS {self.NumberOfAtoms} {self.NumberOfBonds} {self.NumberOfSubstructures} 0 0\nM V30 BEGIN ATOM\n"
        # specify atoms
        for idx, atomObj in enumerate(self.AtomsList):
            mol_str += f"M V30 {idx+1} {atomObj.AtomicSymbol} {round(atomObj.Coordinates[0], 10)} {round(atomObj.Coordinates[1], 10)} {round(atomObj.Coordinates[2], 10)} 0"
            if atomObj.Multiplicity != 1:
                mol_str += f" RAD={atomObj.Multiplicity}"
            if atomObj.FormalCharge != 0:
                mol_str += f" CHG={atomObj.FormalCharge}"
            mol_str += "\n"
        # End atom and begin bonds
        mol_str += "M V30 END ATOM\nM V30 BEGIN BOND\n"
        # specify bonds
        idx = 1
        for i_idx in range(self.NumberOfAtoms):
            for j_idx in range(i_idx + 1, self.NumberOfAtoms):
                BondOrder = self.BondOrderMatrix[i_idx][j_idx]
                if BondOrder == 0:
                    continue
                elif BondOrder % 1 == 0.5:
                    if BondOrder == 1.5:  # Likely Aromatic bond
                        mol_str += f"M V30 {idx} 4 {i_idx+1} {j_idx+1}\n"
                    else:
                        print(
                            "Will need to go back and deal with aromaticity etc correctly"
                        )
                else:
                    mol_str += f"M V30 {idx} {int(BondOrder)} {i_idx+1} {j_idx+1}\n"
                idx += 1
        mol_str += "M V30 END BOND\nM V30 END CTAB\nM END\n"
        return mol_str

    def WriteXYZBlock(self):
        xyz_block = ""
        for atomObj in self.AtomsList:
            xyz_block += f"{atomObj.AtomicSymbol} {atomObj.Coordinates[0]} {atomObj.Coordinates[1]} {atomObj.Coordinates[2]}\n"
        return xyz_block

    def WriteXYZString(self):
        xyz_str = f"{self.NumberOfAtoms}\nIdentifier={self.Identifier} FormalCharge={self.FormalCharge} Multiplicity={self.Multiplicity}\n"
        xyz_str += self.WriteXYZBlock()
        return xyz_str

    def MoleculeToSMILES(self):
        """
        MoleculeToSMILES converts a custom Molecule object into a SMILES string.
        Parameters:
            allHsExplicit (bool): If True, all hydrogen atoms are explicitly included in the SMILES string.
        Returns:
            str: The SMILES representation of the molecule.
        """
        # Create an empty RDKit molecule
        rdkit_mol = Chem.RWMol()

        # Add atoms to the RDKit molecule
        atom_indices = []
        for atomObj in self.AtomsList:
            rdkit_atom = Chem.Atom(atomObj.AtomicSymbol)
            rdkit_atom.SetFormalCharge(atomObj.FormalCharge)
            atom_idx = rdkit_mol.AddAtom(rdkit_atom)
            atom_indices.append(atom_idx)

        # Add bonds based on the connectivity matrix
        if self.ConnectivityMatrix is not None:
            for i in range(self.NumberOfAtoms):
                for j in range(i + 1, self.NumberOfAtoms):
                    if self.ConnectivityMatrix[i][j] > 0:  # Bond exists
                        bond_type = self.GetRDKitBondType(self.BondOrderMatrix[i][j])
                        rdkit_mol.AddBond(i, j, bond_type)

        # Finalize the molecule
        rdmolops.Kekulize(rdkit_mol, clearAromaticFlags=True)
        return Chem.MolToSmiles(rdkit_mol)

    @classmethod
    def ReadMolString(cls, mol_string: str) -> "Molecule":
        """
        Parse a V3000 .MOL file string and create a Molecule object.

        This method reads a V3000 format MOL file string and reconstructs the molecular
        structure by extracting atoms and bond information. It is the inverse operation
        of WriteMolString().

        Parameters:
            mol_string (str): A complete V3000 .MOL file string containing molecule
                             identifier, atom block, and bond block.

        Returns:
            Molecule: A new Molecule object populated with atoms and bond order matrix
                     from the parsed MOL string.

        Raises:
            ValueError: If the MOL string format is invalid or missing required sections.
            IndexError: If atom or bond indices are out of range.

        Examples:
            mol_str = molecule.WriteMolString()
            new_molecule = Molecule.ReadMolString(mol_str)

        Notes:
            - Parses V3000 format only (not V2000)
            - Bond order 4 in V3000 (aromatic) is converted to 1.5
            - Optional properties like charge (CHG) and multiplicity (RAD) are extracted
            - Missing properties default to zero or standard values
        """
        lines = mol_string.strip().split("\n")

        # Extract identifier from first line
        identifier = lines[0].strip()

        # Find atom and bond sections
        atom_begin_idx = None
        atom_end_idx = None
        bond_begin_idx = None
        bond_end_idx = None

        for idx, line in enumerate(lines):
            if "M V30 BEGIN ATOM" in line:
                atom_begin_idx = idx + 1
            elif "M V30 END ATOM" in line:
                atom_end_idx = idx
            elif "M V30 BEGIN BOND" in line:
                bond_begin_idx = idx + 1
            elif "M V30 END BOND" in line:
                bond_end_idx = idx

        if atom_begin_idx is None or atom_end_idx is None:
            raise ValueError("MOL string missing ATOM section")

        # Parse atoms
        atoms_list = []
        atom_indices = {}  # Map MOL indices to list indices

        for idx in range(atom_begin_idx, atom_end_idx):
            line = lines[idx].strip()
            if not line.startswith("M V30"):
                continue
            else:
                parts = line.replace("M V30", "").split()
            mol_idx = int(parts[0])
            atom_symbol = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])

            formal_charge = 0
            multiplicity = 1

            # Parse optional properties
            for i in range(6, len(parts)):
                if parts[i].startswith("CHG="):
                    formal_charge = int(parts[i].split("=")[1])
                elif parts[i].startswith("RAD="):
                    multiplicity = int(parts[i].split("=")[1])

            atom = Atom(
                Label=f"{atom_symbol}{mol_idx}",
                AtomicSymbol=atom_symbol,
                Coordinates=np.array([x, y, z]),
                FormalCharge=formal_charge,
                Multiplicity=multiplicity,
            )
            atom_indices[mol_idx] = len(atoms_list)
            atoms_list.append(atom)

        # Parse bonds
        num_atoms = len(atoms_list)
        bond_order_matrix = np.zeros((num_atoms, num_atoms))

        if bond_begin_idx is not None and bond_end_idx is not None:
            for idx in range(bond_begin_idx, bond_end_idx):
                line = lines[idx].strip()
                if not line.startswith("M V30"):
                    continue
                else:
                    parts = line.replace("M V30", "").split()
                bond_type = int(parts[1])
                atom1_idx = atom_indices[int(parts[2])]
                atom2_idx = atom_indices[int(parts[3])]

                # Convert bond type 4 (aromatic) to 1.5
                bond_order = 1.5 if bond_type == 4 else float(bond_type)

                bond_order_matrix[atom1_idx][atom2_idx] = bond_order
                bond_order_matrix[atom2_idx][atom1_idx] = bond_order

        return cls(identifier, atoms_list, bond_order_matrix)

    @classmethod
    def ReadSMILES(cls, SMILES: str) -> "Molecule":
        pass

    def AddAtom(
        self,
        AtomicSymbol: str,
        Coordinates: np.ndarray,
        Label: str | None,
        FormalCharge: int = 0,
        Multiplicity: int = 1,
        SubstructureIndex: int = 1,
    ):
        self.AtomsList.append(
            Atom(
                AtomicSymbol=AtomicSymbol,
                Coordinates=Coordinates,
                Label=Label,
                FormalCharge=FormalCharge,
                Multiplicity=Multiplicity,
                SubstructureIndex=SubstructureIndex,
            )
        )
        self.BondOrderMatrix = np.pad(self.BondOrderMatrix, ((0, 1), (0, 1)))
        self.DeriveBasicAttributes()

    def AddBond(
        self,
        AtomLabels: list[str] | None = None,
        AtomIndicies: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
        BondOrder: float = 1,
    ):
        if AtomIndicies is not None:
            atomIdx1, atomIdx2 = AtomIndicies
        elif AtomLabels is not None:
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
        elif AtomObjects is not None:
            atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
            atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
        else:
            raise ValueError(
                "AddBond requires AtomLabels, AtomIndicies, or AtomObjects"
            )
        self.BondOrderMatrix[atomIdx1][atomIdx2] = BondOrder
        self.BondOrderMatrix[atomIdx2][atomIdx1] = BondOrder
        self.ConnectivityMatrix = np.floor_divide(
            self.BondOrderMatrix,
            self.BondOrderMatrix,
            out=np.zeros_like(self.BondOrderMatrix),
            where=(self.BondOrderMatrix != 0),
        )
        self.NumberOfBonds = int(self.ConnectivityMatrix.sum().sum() / 2)
        self.NormaliseSubstructureIndicies()

    def AddMolecule(
        self,
        MoleculeToAdd: Self,
    ):
        og_NumberOfAtoms = deepcopy(self.NumberOfAtoms)
        # Add Atoms
        for atomObj in MoleculeToAdd.AtomsList:
            self.AddAtom(
                AtomicSymbol=atomObj.AtomicSymbol,
                Coordinates=atomObj.Coordinates,
                FormalCharge=atomObj.FormalCharge,
                Multiplicity=atomObj.Multiplicity,
                Label=atomObj.Label,
            )
        # Add Bonds
        for atomIdx1 in range(MoleculeToAdd.NumberOfAtoms):
            new_atomIdx1 = atomIdx1 + og_NumberOfAtoms
            for atomIdx2 in range(MoleculeToAdd.NumberOfAtoms):
                new_atomIdx2 = atomIdx2 + og_NumberOfAtoms
                if MoleculeToAdd.BondOrderMatrix[atomIdx1][atomIdx2] != 0:
                    self.AddBond(
                        AtomIndicies=[
                            new_atomIdx1,
                            new_atomIdx2,
                        ],
                        BondOrder=MoleculeToAdd.BondOrderMatrix[atomIdx1][atomIdx2],
                    )

    def RemoveMolecule(
        self,
        SMILES: str | None=None,
        SubstructureIndex: int | None=None,
        AtomIndicies: list[int] | None=None,
    ):
        pass

    def RemoveBond(self):
        pass

    def RemoveAtom(self):
        pass

    def TranslateMolecule(
        self,
        TranslationVector: np.ndarray,
        Displacement: float,
    ):
        TranslationVector = TranslationVector / np.linalg.norm(TranslationVector)
        TranslationVector = TranslationVector * Displacement
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates + TranslationVector
    
    def GetRotationMatrix(self, rotation_axis: np.array, theta: float):
        """
        pass for now
        """
        x, y, z = rotation_axis[0], rotation_axis[1], rotation_axis[2]
        rotation_matrix = np.array(
            [
                [
                    np.cos(theta) + (x**2) * (1 - np.cos(theta)),
                    x * y * (1 - np.cos(theta)) - z * np.sin(theta),
                    x * z * (1 - np.cos(theta)) + y * np.sin(theta),
                ],
                [
                    y * x * (1 - np.cos(theta)) + z * np.sin(theta),
                    np.cos(theta) + (y**2) * (1 - np.cos(theta)),
                    y * z * (1 - np.cos(theta)) - x * np.sin(theta),
                ],
                [
                    z * x * (1 - np.cos(theta)) - y * np.sin(theta),
                    z * y * (1 - np.cos(theta)) + x * np.sin(theta),
                    np.cos(theta) + (z**2) * (1 - np.cos(theta)),
                ],
            ]
        ).reshape((3, 3))
        return rotation_matrix

    def RotateMolecule(
        self,
        RotationVector: np.ndarray,
        RotationAngle: float,
    ):
        # Find geometric midpoint of molecule
        geometric_midpoint = np.array([0.0, 0.0, 0.0])
        for atomObj in self.AtomsList:
            geometric_midpoint += atomObj.Coordinates
        geometric_midpoint = geometric_midpoint / self.NumberOfAtoms
        # Translate to molecule to origin
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates - geometric_midpoint
        # Rotate molecule atom by atom
        RotationMatrix = self.GetRotationMatrix(
            rotation_axis=RotationVector/np.linalg.norm(RotationVector),
            theta=RotationAngle,
        )
        for atomObj in self.AtomsList:
            atomObj.Coordinates = RotationMatrix @ atomObj.Coordinates
        # Translate back to original position
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates + geometric_midpoint