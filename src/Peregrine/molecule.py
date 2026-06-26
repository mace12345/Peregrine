# fmt: on
from typing import Self
from copy import deepcopy
from pathlib import Path
import os
import warnings
import subprocess

import numpy as np
from scipy.spatial import ConvexHull

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import rdkit.Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import SanitizeMol, SanitizeFlags
from rdkit import RDLogger
import rdkit
from rdkit import rdBase

from openbabel import pybel
from openbabel import openbabel as ob

from ase import Atoms as aseAtoms
from dscribe.descriptors import SOAP

from xyzgraph import build_graph

from .atom import Atom

RDKIT_BONDTYPE_TRANSLATION = {
    1: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    2: Chem.BondType.DOUBLE,
    2.5: Chem.BondType.TWOANDAHALF,
    3: Chem.BondType.TRIPLE,
    3.5: Chem.BondType.THREEANDAHALF,
    4: Chem.BondType.QUADRUPLE,
    4.5: Chem.BondType.FOURANDAHALF,
    5: Chem.BondType.QUINTUPLE,
    5.5: Chem.BondType.FIVEANDAHALF,
    6: Chem.BondType.HEXTUPLE,
}


# === Helper functions ===


def _XYZBlockToAtomsList(
    xyz_block: str, template_molObj: "Molecule | None" = None
) -> tuple[list[Atom], int]:
    symbols = []
    coords = []
    for line in xyz_block.split("\n"):
        parts = line.split()
        if not parts:
            continue
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))

    coord_array = np.array(coords, dtype=float)  # shape (N, 3)

    AtomsList = []
    if template_molObj is None:
        for symbol, row in zip(symbols, coord_array):
            AtomsList.append(Atom(AtomicSymbol=symbol, Coordinates=row))
    else:
        template_atoms = template_molObj.AtomsList
        for symbol, row, t_atom in zip(symbols, coord_array, template_atoms):
            AtomsList.append(
                Atom(
                    AtomicSymbol=symbol,
                    Coordinates=row,
                    FormalCharge=t_atom.FormalCharge,
                    Multiplicity=t_atom.Multiplicity,
                    GetAtomAttributes=False,
                )
            )

    return AtomsList, len(AtomsList)


def _GradBlockInToAtomsList(AtomsList: list[Atom], grad_block: str) -> list[Atom]:
    for line in grad_block.split("\n"):
        line = line.split()
        idx = int(line[0]) - 1
        AtomsList[idx].Gradient = np.array(
            [
                float(line[3]),
                float(line[4]),
                float(line[5]),
            ]
        )
    return AtomsList


def _BondBlockToBondOrderMatrix(bond_block: str, AtomsListLen: int) -> np.ndarray:
    BondOrderMatrix = np.zeros((AtomsListLen, AtomsListLen))
    bonds_list = bond_block.split("B(")[1:]
    for line in bonds_list:
        left, _, right = line.partition(",")
        idx1 = int(left.partition("-")[0])
        idx2 = int(right.partition("-")[0])
        mayer_BO = float(line.rpartition(":")[2])
        # round to nearest 0.5, floor at 1.0
        BO = max(round(mayer_BO * 2) / 2, 1.0)
        BondOrderMatrix[idx1, idx2] = BO
        BondOrderMatrix[idx2, idx1] = BO
    return BondOrderMatrix, len(bonds_list)


def _GetCalculatedEnergies(orca_string: str, check_final_energies: bool) -> dict:
    en_output_dict = {
        "Electronic Energy": None,
        "Enthalpy": None,
        "Entropy": None,
        "Gibbs Free Energy": None,
    }

    before, sep, after = orca_string.rpartition("FINAL SINGLE POINT ENERGY")
    if sep:
        en_output_dict["Electronic Energy"] = float(after.partition("\n")[0])

    if check_final_energies:
        before, sep, after = orca_string.partition("Total Enthalpy")
        if sep:
            en_output_dict["Enthalpy"] = float(
                after.partition("...")[2].partition("Eh")[0]
            )

        before, sep, after = orca_string.partition("Final entropy term")
        if sep:
            en_output_dict["Entropy"] = float(
                after.partition("...")[2].partition("Eh")[0]
            )

        before, sep, after = orca_string.partition("Final Gibbs free energy")
        if sep:
            en_output_dict["Gibbs Free Energy"] = float(
                after.partition("...")[2].partition("Eh")[0]
            )

    return en_output_dict


class Molecule:
    def __init__(
        self,
        Identifier: str,
        AtomsList: list[Atom],
        BondOrderMatrix: np.ndarray | None,
        DeriveAttributes: bool = True,
        CheckMolObj: bool = True,
        UpdateAtomLabels: bool = True,
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
        if DeriveAttributes == True:
            self.DeriveBasicAttributes(UpdateAtomLabels=UpdateAtomLabels)

        # Optional SMILES Attributes
        self.AssociatedMoleculeSMILES = None

        # Check Attributes
        if CheckMolObj == True:
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

        # Calculated Attributes
        self.num_basis_functions: int | None = None
        self.electronic_energy: float | None = None
        self.enthalpy: float | None = None
        self.entropy: float | None = None
        self.gibbs_free_energy: float | None = None
        self.vibrational_frequencies: list[float] | None = None

    def DeriveBasicAttributes(
        self,
        UpdateAtomLabels: bool = True,
        UpdateSubstructureIndices: bool = True,
    ):
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
        # Calculate Atomic Valence
        for idx, atomObj in enumerate(self.AtomsList):
            atomObj.Valence = self.BondOrderMatrix[idx].sum()
        self.NormaliseAtomLabels(UpdateAtomLabels=UpdateAtomLabels)
        self.AtomsDict = {
            Atom.Label: [idx, Atom] for idx, Atom in enumerate(self.AtomsList)
        }
        self.NumberOfAtoms = len(self.AtomsList)
        self.GetFormalCharge()
        self.GetMultiplicity()
        if UpdateSubstructureIndices == True:
            self.NormaliseSubstructureIndicies()

    def DeriveMoleculeSMILES(self):
        # Split substructuures into their own molecule objects
        Components = self.SplitMoleculeIntoComponents(UpdateAtomLabels=False)
        for component in Components:
            SMILES_str = component.WriteSMILESString()
            for atomObj in component.AtomsList:
                self.AtomsDict[atomObj.Label][1].AssociatedSMILES = SMILES_str
        self.AssociatedMoleculeSMILES = self.WriteSMILESString()

    def SplitMoleculeIntoComponents(self, UpdateAtomLabels: bool = True) -> list[Self]:
        # Split substructuures into their own molecule objects
        new_Molecules = []
        substructure_dict = {i + 1: [] for i in range(self.NumberOfSubstructures)}
        for atomObj in self.AtomsList:
            substructure_dict[atomObj.SubstructureIndex] += [deepcopy(atomObj)]
        for substructure_idx in substructure_dict:
            new_AtomsList = substructure_dict[substructure_idx]
            new_Identifier = self.Identifier + f"_{substructure_idx}"
            new_BondOrderMatrix = np.zeros((len(new_AtomsList), len(new_AtomsList)))
            for new_atomIdx1, atomObj1 in enumerate(new_AtomsList):
                old_atomIdx1 = self.AtomsDict[atomObj1.Label][0]
                for new_atomIdx2, atomObj2 in enumerate(new_AtomsList):
                    old_atomIdx2 = self.AtomsDict[atomObj2.Label][0]
                    if self.BondOrderMatrix[old_atomIdx1][old_atomIdx2] != 0:
                        new_BondOrderMatrix[new_atomIdx1][new_atomIdx2] = (
                            self.BondOrderMatrix[old_atomIdx1][old_atomIdx2]
                        )
            new_Molecules.append(
                Molecule(
                    Identifier=new_Identifier,
                    AtomsList=new_AtomsList,
                    BondOrderMatrix=new_BondOrderMatrix,
                    UpdateAtomLabels=UpdateAtomLabels,
                )
            )
        return new_Molecules

    def NormaliseAtomLabels(self, UpdateAtomLabels: bool = True):
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
                if UpdateAtomLabels == True:
                    atomObj.Label = f"{atomObj.AtomicSymbol}1"
            else:
                atomic_symbol_count_dict[atomObj.AtomicSymbol] += 1
                if UpdateAtomLabels == True:
                    atomObj.Label = f"{atomObj.AtomicSymbol}{atomic_symbol_count_dict[atomObj.AtomicSymbol]}"
            self.MolecularMass += atomObj.AtomicMass
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

    # === Get Molecule properties ===

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

    def GetCentreOfMass(self) -> np.ndarray:
        centre = np.array([0.0, 0.0, 0.0])
        for atomObj in self.AtomsList:
            centre += atomObj.Coordinates * atomObj.AtomicMass
        return centre / self.MolecularMass

    def GetMoleculeRadius(self) -> float:
        radius = 0
        mid_point = self.GetCentreOfMass()
        for atomObj in self.AtomsList:
            test_radius = (
                np.linalg.norm(atomObj.Coordinates - mid_point) + atomObj.AtomicRadii
            )
            if test_radius > radius:
                radius = test_radius
        return radius

    def GetMoleculeVolume(self) -> float:
        points = [atomObj.Coordinates for atomObj in self.AtomsList]
        hull = ConvexHull(points)
        return round(hull.volume, 2)

    def GetAromaticAtoms(
        self,
        MolecularMechanicsPreOpt: bool = False,
        SemiEmpiricalxTBPreOpt: bool = False,
    ):
        components = self.SplitMoleculeIntoComponents(UpdateAtomLabels=False)
        for component in components:
            if component.NumberOfAtoms < 6:
                continue
            # Optimse component so xyzgraph correctly identifies aromatic atoms
            if MolecularMechanicsPreOpt == True:
                component.OptimiseGeometry(
                    MolecularMechanics=MolecularMechanicsPreOpt,
                )
            if SemiEmpiricalxTBPreOpt == True:
                component.OptimiseGeometry(
                    SemiEmpiricalxTB=SemiEmpiricalxTBPreOpt,
                )
            # Convert to xyz file
            xyz_string = component.WriteXYZString()
            with open(
                f"{Path(__file__).parent}/{component.Identifier}_temp.xyz", "w"
            ) as f:
                f.write(xyz_string)
                f.close()
            G_full = build_graph(
                atoms=f"{Path(__file__).parent}/{component.Identifier}_temp.xyz",
                charge=component.FormalCharge,
                multiplicity=component.Multiplicity,
            )
            os.remove(f"{Path(__file__).parent}/{component.Identifier}_temp.xyz")
            # Flatten all aromatic rings into a single set of atom indices
            aromatic_atoms = {
                idx for ring in G_full.graph.get("aromatic_rings", []) for idx in ring
            }
            # Get atom label from atom indices
            # call main AtomsDict and set atom to aromatic
            for aromatic_index in aromatic_atoms:
                self.AtomsDict[component.AtomsList[aromatic_index].Label][
                    1
                ].IsAromatic = True
        for atomObj in self.AtomsList:
            if atomObj.IsAromatic is None:
                atomObj.IsAromatic = False

    def GetBondAngle(
        self,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
    ) -> float:
        """
        Calculate the bond angle between three atoms in radians.

        The angle is measured at the central atom (second atom in the input list).
        For example, GetBondAngle(AtomLabels=['H1', 'C1', 'H2']) calculates the
        H1-C1-H2 bond angle with C1 as the central atom.

        Parameters:
            AtomLabels (list[str] | None): Labels of three atoms (e.g., ['H1', 'C1', 'H2'])
            AtomIndices (list[int] | None): Indices of three atoms in AtomsList
            AtomObjects (list[Atom] | None): Direct references to three Atom objects

        Returns:
            float: The bond angle in degrees, rounded to 2 decimal places

        Raises:
            ValueError: If not exactly 3 atoms provided, or invalid atom specification
            IndexError: If atom indices are out of bounds
            ValueError: If atoms are at same location (degenerate geometry)

        Examples:
            # Using atom labels
            angle = molecule.GetBondAngle(AtomLabels=['H1', 'O1', 'H2'])

            # Using atom indices
            angle = molecule.GetBondAngle(AtomIndexs=[0, 1, 2])
        """
        # Determine atom indices
        if AtomIndices is not None:
            if len(AtomIndices) != 3:
                raise ValueError("AtomIndexs must contain exactly 3 indices")
            atomIdx1, atomIdx2, atomIdx3 = AtomIndices
            if not all(0 <= idx < self.NumberOfAtoms for idx in AtomIndices):
                raise IndexError("Atom indices out of bounds")
        elif AtomLabels is not None:
            if len(AtomLabels) != 3:
                raise ValueError("AtomLabels must contain exactly 3 labels")
            if not all(label in self.AtomsDict for label in AtomLabels):
                raise ValueError("One or more atom labels not found in molecule")
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
            atomIdx3 = self.AtomsDict[AtomLabels[2]][0]
        elif AtomObjects is not None:
            if len(AtomObjects) != 3:
                raise ValueError("AtomObjects must contain exactly 3 objects")
            try:
                atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
                atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
                atomIdx3 = self.AtomsDict[AtomObjects[2].Label][0]
            except KeyError as e:
                raise ValueError(f"Atom object not found in molecule: {e}")
        else:
            raise ValueError(
                "GetBondAngle requires AtomLabels, AtomIndexs, or AtomObjects"
            )

        # Get atom coordinates
        # The central atom is the second one (index 1)
        central_atom = self.AtomsList[atomIdx2]
        atom1 = self.AtomsList[atomIdx1]
        atom3 = self.AtomsList[atomIdx3]

        # Create vectors from central atom to the other two atoms
        vector1 = atom1.Coordinates - central_atom.Coordinates
        vector3 = atom3.Coordinates - central_atom.Coordinates

        # Calculate the angle using dot product
        dot_product = np.dot(vector1, vector3)
        magnitude1 = np.linalg.norm(vector1)
        magnitude3 = np.linalg.norm(vector3)

        # Avoid division by zero
        if magnitude1 == 0 or magnitude3 == 0:
            raise ValueError("Bond angle undefined: atoms at same location")

        cos_angle = dot_product / (magnitude1 * magnitude3)

        # Clamp to [-1, 1] to avoid numerical errors in arccos
        cos_angle = np.clip(cos_angle, -1, 1)

        # Calculate angle in radians
        return np.arccos(cos_angle)

    # === Get atomic descriptors ===

    def GetSOAPDescriptors(
        self,
        RadiusCutOff: float = 5.0,
        NumRadialBasisFunctions: int = 8,
        MaxDegreeSphericalHarm: int = 6,
        AtomicSymbols: list[str] | None = None,
        periodic: bool = False,
    ):
        """
        Using DScribe python package to calculate atomic SOAP descriptors for MLP training.

        DScribe SOAP() object is initialised
        Molecule object is converted to ASE Atoms object
        ASE atoms object is used to feed into SOAP() object and calculate SOAP descriptors

        Keyword arguments:
            RadiusCutOff -- MLPs are based on atomic centred clusters, so how many atoms will be included in the defined radius for the soap descriptor (default = 5 angstrom)
            NumRadialBasisFunctions --
            MaxDegreeSphericalHarm --
            AtomicSymbols -- Chemical elements used to construct descriptor (species in DScribe) (default is the chemical elements that exists in the molObj)
            periodic -- Is the ASE Atoms object structure periodic or not (default = False)
        """
        if AtomicSymbols is None:
            AtomicSymbols = list({atomObj.AtomicSymbol for atomObj in self.AtomsList})
        soap = SOAP(
            species=AtomicSymbols,
            r_cut=RadiusCutOff,
            n_max=NumRadialBasisFunctions,
            l_max=MaxDegreeSphericalHarm,
            periodic=periodic,
        )
        aseMolObj = self.MoleculeToASEMolecule()
        for idx, atomObj in enumerate(self.AtomsList):
            atomObj.SOAPDescriptor = list(
                float(i) for i in soap.create(aseMolObj, centers=[idx])[0]
            )

    # === Read/Write files & SMILES/SMARTS & convert molecule objects ===

    def EquivelentMoleculeInchi(self, SMILES1: str, SMILES2: str) -> bool:
        SMILES1_rdkitObj = Chem.MolFromSmiles(SMILES1)
        SMILES2_rdkitObj = Chem.MolFromSmiles(SMILES2)
        if SMILES1_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMILES string: {SMILES1}")
            return False
        if SMILES2_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMILES string: {SMILES2}")
            return False
        else:
            return Chem.MolToInchi(SMILES1_rdkitObj) == Chem.MolToInchi(
                SMILES2_rdkitObj
            )

    def SMARTSMatchesSMILES(self, SMILES: str, SMARTS: str) -> tuple:
        SMILES_rdkitObj = Chem.MolFromSmiles(SMILES)
        SMARTS_rdkitObj = Chem.MolFromSmarts(SMARTS)
        if SMILES_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMILES string: {SMILES}")
            return False
        if SMARTS_rdkitObj is None:
            print(f"Could not generate rdkitObj from SMARTS string: {SMARTS}")
            return False
        matches = SMILES_rdkitObj.GetSubstructMatches(SMARTS_rdkitObj)
        return matches

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
            if atomObj.SMARTSCentre == True:
                mol_str += f" SMC={1}"
            if atomObj.Multiplicity != 1:
                mol_str += f" RAD={atomObj.Multiplicity}"
            if atomObj.FormalCharge != 0:
                mol_str += f" CHG={atomObj.FormalCharge}"
            if atomObj.Gradient is not None:
                mol_str += f" XGD={atomObj.Gradient[0]} YGD={atomObj.Gradient[1]} ZGD={atomObj.Gradient[2]}"
            if atomObj.SOAPDescriptor is not None:
                mol_str += f" SPD={str(atomObj.SOAPDescriptor).replace(" ", "")}"
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
                    elif BondOrder == 2.5:
                        mol_str += f"M V30 {idx} 2 {i_idx+1} {j_idx+1}\n"
                    elif BondOrder == 3.5:
                        mol_str += f"M V30 {idx} 3 {i_idx+1} {j_idx+1}\n"
                else:
                    mol_str += f"M V30 {idx} {int(BondOrder)} {i_idx+1} {j_idx+1}\n"
                idx += 1
        mol_str += "M V30 END BOND\nM V30 END CTAB\nM END\n"
        # Add properties
        if self.electronic_energy is not None:
            mol_str += f"> <Electronic Energy (Eh)>\n{self.electronic_energy}\n"
        if self.gibbs_free_energy is not None:
            mol_str += f"> <Gibbs Free Energy (Eh)>\n{self.gibbs_free_energy}\n"
        if self.enthalpy is not None:
            mol_str += f"> <Enthalpy (Eh)>\n{self.enthalpy}\n"
        if self.entropy is not None:
            mol_str += f"> <Entropy (Eh)>\n{self.entropy}\n"
        return mol_str

    def WriteXYZBlock(self):
        xyz_block = ""
        for atomObj in self.AtomsList:
            xyz_block += f"{atomObj.AtomicSymbol} {atomObj.Coordinates[0]} {atomObj.Coordinates[1]} {atomObj.Coordinates[2]}\n"
        return xyz_block

    def WriteXYZString(self) -> str:
        xyz_str = f"{self.NumberOfAtoms}\nIdentifier={self.Identifier} FormalCharge={self.FormalCharge} Multiplicity={self.Multiplicity}\n"
        xyz_str += self.WriteXYZBlock()
        return xyz_str

    def WriteSMILESString(self) -> str:
        # Create an empty RDKit molecule
        rdkit_mol = Chem.RWMol()
        # Add atoms to the RDKit molecule
        for atomObj in self.AtomsList:
            rdkit_atom = Chem.Atom(atomObj.AtomicSymbol)
            rdkit_atom.SetFormalCharge(atomObj.FormalCharge)
            atom_idx = rdkit_mol.AddAtom(rdkit_atom)
        # Add bonds based on the connectivity matrix
        if self.ConnectivityMatrix is not None:
            for i in range(self.NumberOfAtoms):
                for j in range(i + 1, self.NumberOfAtoms):
                    if self.ConnectivityMatrix[i][j] > 0:  # Bond exists
                        bond_type = RDKIT_BONDTYPE_TRANSLATION[
                            self.BondOrderMatrix[i][j]
                        ]
                        rdkit_mol.AddBond(i, j, bond_type)
        # Finalize the molecule and make SMILES string
        rdmolops.Kekulize(rdkit_mol, clearAromaticFlags=True)
        SMILES_str = Chem.MolToSmiles(rdkit_mol)
        return SMILES_str

    def WriteSMARTSString(
        self,
        HandleAromaticity: bool = True,
    ) -> str:
        # Create initial SMARTS string
        # Convert molObj to rdKitMolObj
        rdkitMolObj = Chem.EditableMol(Chem.Mol())
        molObj_to_rdkitMolObj_atomIdx_dict = {}
        rdkitMolObj_to_molObj_atomIdx_dict = {}
        for atomObj_idx, atomObj in enumerate(self.AtomsList):
            if atomObj.SMARTSCentre == True:
                rdkitAtomObj = Chem.Atom(atomObj.AtomicSymbol)
                rdkitAtomObj.SetFormalCharge(atomObj.FormalCharge)
                rdkitAtomObj.SetNumRadicalElectrons(atomObj.Multiplicity - 1)
                rdkitAtomObj_idx = rdkitMolObj.AddAtom(rdkitAtomObj)
                molObj_to_rdkitMolObj_atomIdx_dict[atomObj_idx] = rdkitAtomObj_idx
                rdkitMolObj_to_molObj_atomIdx_dict[rdkitAtomObj_idx] = atomObj_idx
        for i in range(self.NumberOfAtoms):
            if i in molObj_to_rdkitMolObj_atomIdx_dict:
                for j in range(i + 1, self.NumberOfAtoms):
                    if j in molObj_to_rdkitMolObj_atomIdx_dict:
                        if self.BondOrderMatrix[i][j] != 0:
                            rdkitMolObj.AddBond(
                                molObj_to_rdkitMolObj_atomIdx_dict[i],
                                molObj_to_rdkitMolObj_atomIdx_dict[j],
                                RDKIT_BONDTYPE_TRANSLATION[self.BondOrderMatrix[i][j]],
                            )
        rdkitMolObj = rdkitMolObj.GetMol()
        # convert rdkitMolObj to SMARTS string
        for rdkitAtomObj in rdkitMolObj.GetAtoms():
            rdkitAtomObj.SetAtomMapNum(rdkitAtomObj.GetIdx())
        SMARTS = Chem.MolToSmarts(rdkitMolObj)
        # Add 0th index to smarts
        for sub_SMARTS in SMARTS.split("]"):
            if ":" not in sub_SMARTS:
                new_sub_SMARTS = sub_SMARTS + ":0]"
                old_sub_SMARTS = sub_SMARTS + "]"
                break
        SMARTS = SMARTS.replace(old_sub_SMARTS, new_sub_SMARTS)

        # Create dictionary of molObj atomIdx dict to atoms SMARTS pattern
        # Using replace function on strings old SMARTS patterns can be swapped for new ones
        # The dictionary created contains all the old SMARTS pattern
        Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict = {}
        for sub_SMARTS in SMARTS.split("]"):
            if sub_SMARTS == "":
                continue
            sub_SMARTS = sub_SMARTS.split("[")[-1]
            rdkitAtomObj_idx = int(sub_SMARTS.split(":")[-1])
            Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                f"[{sub_SMARTS}]"
            )

        # Edit SMARTS string
        # Collect rdkit atom properties from own molObj
        # Make new SMARTS patterns based on the molObj atoms
        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict = {}
        if HandleAromaticity == True:
            self.GetAromaticAtoms()
        for rdkitAtomObj_idx in rdkitMolObj_to_molObj_atomIdx_dict:

            atomObj_idx = rdkitMolObj_to_molObj_atomIdx_dict[rdkitAtomObj_idx]
            atomObj = self.AtomsList[atomObj_idx]
            valence = int(self.BondOrderMatrix[atomObj_idx].sum())
            SMARTSAtom = atomObj.SMARTSAtom

            if SMARTSAtom is not None:
                New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                    f"{SMARTSAtom}:{rdkitAtomObj_idx}"
                )
                continue

            # Construct SMARTS for both all radicals and aromatic radicals
            if atomObj.Multiplicity == 2:
                valence = int(self.BondOrderMatrix[atomObj_idx].sum())
                if atomObj.IsAromatic == None or atomObj.IsAromatic == False:
                    if atomObj.FormalCharge == 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[#{atomObj.AtomicNumber}v{valence}+0:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge > 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[#{atomObj.AtomicNumber}v{valence}+{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge < 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[#{atomObj.AtomicNumber}v{valence}-{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                elif atomObj.IsAromatic == True:
                    if atomObj.FormalCharge == 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[{atomObj.AtomicSymbol.lower()}v{valence}+0:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge > 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[{atomObj.AtomicSymbol.lower()}v{valence}+{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                    elif atomObj.FormalCharge < 0:
                        New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                            f"[{atomObj.AtomicSymbol.lower()}v{valence}-{atomObj.FormalCharge}:{rdkitAtomObj_idx}]"
                        )
                continue

            # Construct SMARTS for aromatic atoms
            if atomObj.IsAromatic == True:
                old_smarts_pattern = Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict[
                    rdkitAtomObj_idx
                ]
                New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[rdkitAtomObj_idx] = (
                    old_smarts_pattern.replace(
                        f"#{atomObj.AtomicNumber}", atomObj.AtomicSymbol.lower()
                    )
                )

        # Swap out old SMARTS patterns with new SMARTS patterns
        for new_rdkitAtomObj_idx in New_rdkitAtomObjIdx_to_SMARTS_pattern_dict:
            old_smarts_pattern = Old_rdkitAtomObjIdx_to_SMARTS_pattern_dict[
                new_rdkitAtomObj_idx
            ]
            new_smarts_pattern = New_rdkitAtomObjIdx_to_SMARTS_pattern_dict[
                new_rdkitAtomObj_idx
            ]
            SMARTS = SMARTS.replace(old_smarts_pattern, new_smarts_pattern)

        return SMARTS

    def WriteORCAInput(self):
        pass

    def MoleculeToRDKitMol(self):
        pass

    def MoleculeToASEMolecule(self) -> aseAtoms:
        ASEMolecule = aseAtoms(
            symbols=[atomObj.AtomicSymbol for atomObj in self.AtomsList],
            positions=[tuple(atomObj.Coordinates) for atomObj in self.AtomsList],
        )
        ASEMolecule.info["spin_multiplicity"] = self.Multiplicity
        ASEMolecule.info["charge"] = self.FormalCharge
        return ASEMolecule

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

            SOAPDescriptor = None
            SMARTSCentre = False
            formal_charge = 0
            multiplicity = 1
            # Parse optional properties
            # including gradient of atom
            Gradient = np.array([None, None, None])
            for i in range(6, len(parts)):
                if parts[i].startswith("CHG="):
                    formal_charge = int(parts[i].split("=")[1])
                elif parts[i].startswith("RAD="):
                    multiplicity = int(parts[i].split("=")[1])
                elif parts[i].startswith("XGD="):
                    Gradient[0] = float(parts[i].split("=")[1])
                elif parts[i].startswith("YGD="):
                    Gradient[1] = float(parts[i].split("=")[1])
                elif parts[i].startswith("ZGD="):
                    Gradient[2] = float(parts[i].split("=")[1])
                elif parts[i] == "SMC=1":
                    SMARTSCentre = True
                elif parts[i].startswith("SPD="):
                    SOAPDescriptor = [
                        float(i)
                        for i in parts[i]
                        .split("=")[1]
                        .replace("[", "")
                        .replace("]", "")
                        .split(",")
                    ]
            atomObj = Atom(
                Label=f"{atom_symbol}{mol_idx}",
                AtomicSymbol=atom_symbol,
                Coordinates=np.array([x, y, z]),
                FormalCharge=formal_charge,
                Multiplicity=multiplicity,
                SMARTSCentre=SMARTSCentre,
                SOAPDescriptor=SOAPDescriptor,
            )
            if Gradient[0] is None and Gradient[1] is None and Gradient[2] is None:
                pass
            else:
                atomObj.Gradient = Gradient

            atom_indices[mol_idx] = len(atoms_list)
            atoms_list.append(atomObj)

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

        # Create molecule object
        molObj = cls(identifier, atoms_list, bond_order_matrix)

        # Parse for molecule properties
        property_map = {
            "> <Electronic Energy (Eh)>": "electronic_energy",
            "> <Gibbs Free Energy (Eh)>": "gibbs_free_energy",
            "> <Enthalpy (Eh)>": "enthalpy",
            "> <Entropy (Eh)>": "entropy",
        }
        for idx, line in enumerate(lines):
            attr = property_map.get(line.strip())
            if attr is not None and idx + 1 < len(lines):
                setattr(molObj, attr, float(lines[idx + 1].strip()))

        return molObj

    @classmethod
    def ReadMol2String(cls, mol2_string: str) -> "Molecule":
        """
        Reads a `.mol2` format string and parses the molecular data into a Molecule object.

        The method:
        - Parses atom and bond information from the mol2 string.
        - Constructs Atom and Molecule objects.
        - Builds connectivity and bond order matrices.
        - Handles specific logic for "am" bond types (e.g., C-N → single bond, C-O → double bond).

        Args:
            mol2_string (str): A single molecule's mol2 format string content.

        Returns:
            Molecule: A Molecule object with parsed atoms, bonds, and connectivity information.

        Raises:
            ValueError: If the mol2 string format is invalid or missing required sections.

        Example:
            mol2_content = "...mol2 string content..."
            molecule = Molecule.ReadMol2String(mol2_content)
        """
        # Parse molecule info section
        molucule_info_string = mol2_string.split("@<TRIPOS>ATOM\n")[0]
        molecule_atom_string = mol2_string.split("@<TRIPOS>ATOM\n")[-1].split("@")[0]
        molecule_bond_string = mol2_string.split("@<TRIPOS>BOND\n")[-1].split("@")[0]

        identifier = molucule_info_string.split("\n")[0]
        atom_bond_number = [
            i for i in molucule_info_string.split("\n")[1].split(" ") if i != ""
        ]
        number_of_atoms = int(atom_bond_number[0])
        number_of_bonds = int(atom_bond_number[1])
        number_of_substructures = int(atom_bond_number[2])

        # Parse atoms
        molecule_atom_list = [
            [i for i in j.split(" ") if i != ""]
            for j in molecule_atom_string.split("\n")
        ]

        atoms_list = []
        for atom in molecule_atom_list:
            if len(atom) == 0:
                continue
            atoms_list.append(
                Atom(
                    Label=atom[1],
                    Coordinates=np.array(
                        [
                            float(atom[2]),
                            float(atom[3]),
                            float(atom[4]),
                        ]
                    ),
                    AtomicSymbol=atom[5].split(".")[0],
                    SubstructureIndex=atom[6],
                    FormalCharge=int(float(atom[8])),
                )
            )

        # Parse bonds
        molecule_bond_list = [
            [i for i in j.split(" ") if i != ""]
            for j in molecule_bond_string.split("\n")
        ]

        if len(molecule_bond_list[0]) == 4:
            bond_order_matrix = np.zeros((number_of_atoms, number_of_atoms))
            for bond in molecule_bond_list:
                if len(bond) == 0:
                    continue
                atom1_index = int(bond[1])
                atom2_index = int(bond[2])
                bond_type = bond[3]

                # Handle "am" bond type logic
                if bond_type == "am":
                    if (
                        atoms_list[atom1_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom2_index - 1].AtomicSymbol == "N"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 1
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 1
                    elif (
                        atoms_list[atom2_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom1_index - 1].AtomicSymbol == "N"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 1
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 1
                    if (
                        atoms_list[atom1_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom2_index - 1].AtomicSymbol == "O"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 2
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 2
                    elif (
                        atoms_list[atom2_index - 1].AtomicSymbol == "C"
                        and atoms_list[atom1_index - 1].AtomicSymbol == "O"
                    ):
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = 2
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = 2
                else:
                    # Standard bond type mapping would require access to bond_types_to_bond_order_dict
                    # For now, use standard mapping
                    bond_order_dict = {
                        "1": 1,
                        "2": 2,
                        "3": 3,
                        "ar": 1.5,
                        "du": 1,
                        "un": 1,
                        "nc": 0,
                    }
                    if bond_type in bond_order_dict:
                        bond_order = bond_order_dict[bond_type]
                        bond_order_matrix[atom1_index - 1][atom2_index - 1] = bond_order
                        bond_order_matrix[atom2_index - 1][atom1_index - 1] = bond_order
        else:
            bond_order_matrix = np.array([[0]])

        # Create Molecule object
        mol_obj = cls(
            Identifier=identifier,
            AtomsList=atoms_list,
            BondOrderMatrix=bond_order_matrix,
        )

        # Handle multiplicities if present in mol2 string
        if "Multiplicities: " in molucule_info_string:
            molucule_mult_string = (
                molucule_info_string.split("Multiplicities: ")[1].split("},")[0] + "}"
            )
            molucule_mult_string = molucule_mult_string.replace(" ", "")
            molucule_mult_dict = eval(molucule_mult_string)
            for atomLabel in molucule_mult_dict:
                atomObj = mol_obj.AtomsDict[atomLabel][1]
                atomObj.Multiplicity = molucule_mult_dict[atomLabel]

        return mol_obj

    @classmethod
    def ReadXYZFile(
        cls, xyz_file: str, identifier: str, charge: int, multiplicity: int
    ) -> "Molecule":
        G_full = build_graph(
            atoms=xyz_file,
            charge=charge,
            multiplicity=multiplicity,
        )
        # Flatten all aromatic rings into a single set of atom indices
        aromatic_atoms = {
            idx for ring in G_full.graph.get("aromatic_rings", []) for idx in ring
        }
        AtomsList = [
            Atom(
                AtomicSymbol=d["symbol"],
                Coordinates=np.array(d["position"]),
                FormalCharge=d["formal_charge"],
            )
            for i, d in G_full.nodes(data=True)
        ]
        for aromatic_index in aromatic_atoms:
            AtomsList[aromatic_index].IsAromatic = True
        BondOrderMatrix = np.zeros((len(AtomsList), len(AtomsList)))
        for i, j, d in G_full.edges(data=True):
            BondOrderMatrix[i][j] = d["bond_order"]
            BondOrderMatrix[j][i] = d["bond_order"]
        molObj = Molecule(
            Identifier=identifier,
            AtomsList=AtomsList,
            BondOrderMatrix=BondOrderMatrix,
        )
        if molObj.Multiplicity != multiplicity:
            for atomObj in molObj.AtomsList:
                atom_valence_electron_count = (
                    atomObj.Valence
                    + atomObj.AtomicValenceElectronCount
                    + (-1 * atomObj.FormalCharge)
                )
                if atom_valence_electron_count % 2 == 1:
                    atomObj.Multiplicity = 2
            molObj.GetMultiplicity()
            if molObj.Multiplicity != multiplicity:
                print("Need to improve this multiplicity assigning function")
        return molObj

    @classmethod
    def ReadSMILESString(cls, SMILES: str) -> "Molecule":
        pass

    @classmethod
    def ReadORCA6Output(cls, ORCA_output_filepath: str) -> "Molecule":
        #
        pass

    @classmethod
    def ReadORCA6OutputGradients(
        cls, ORCA_output_filepath: str, template_molObj: "Molecule | None" = None
    ) -> list["Molecule"]:
        """
        Important Note: If template molecule not provided func will place overall multiplicity and charge on first atom in the atomlist index
        """
        # TODO: Raise Errors when template object does not match up with ORCA molecule file
        with open(ORCA_output_filepath, "r") as f:
            orca_file = f.read()
            f.close()
        Identifier = ORCA_output_filepath.split("/")[-1].split(".")[0]
        orca_file_geom_opt_steps = orca_file.split("GEOMETRY OPTIMIZATION CYCLE")[1:]
        num_opt_step = len(orca_file_geom_opt_steps)
        charge_mult = [
            int(i)
            for i in orca_file.split("> *xyz ")[1].split("\n")[0].split(" ")
            if i != ""
        ]
        molObj_list = []
        prev_BondOrderMatrix = None
        prev_NumberOfBonds = None
        check_final_energies = False
        for opt_step_idx, opt_step in enumerate(orca_file_geom_opt_steps):
            # Get XYZ coordinates
            xyz_block = opt_step.rpartition(
                "CARTESIAN COORDINATES (ANGSTROEM)\n---------------------------------\n"
            )[2].partition("\n\n")[0]
            AtomsList, NumberOfAtoms = _XYZBlockToAtomsList(xyz_block, template_molObj)
            # Get Mayer bond orders
            if template_molObj is None:
                parts = opt_step.split("Mayer bond orders larger than 0.100000")
                if len(parts) > 1:
                    bond_block = parts[-1].split("\n\n")[0]
                    BondOrderMatrix, NumberOfBonds = _BondBlockToBondOrderMatrix(
                        bond_block, len(AtomsList)
                    )
                    prev_BondOrderMatrix = BondOrderMatrix
                    prev_NumberOfBonds = NumberOfBonds
                else:
                    BondOrderMatrix = prev_BondOrderMatrix
                    NumberOfBonds = prev_NumberOfBonds
            else:
                BondOrderMatrix = template_molObj.BondOrderMatrix
                NumberOfBonds = template_molObj.NumberOfBonds
            # Get cartesian gradients
            parts = opt_step.split("CARTESIAN GRADIENT\n------------------\n\n")
            if len(parts) > 1:
                grad_block = parts[-1].split("\n\n", 1)[0]
                AtomsList = _GradBlockInToAtomsList(AtomsList, grad_block)
            molObj = Molecule(
                Identifier=f"{Identifier}_opt{opt_step_idx}",
                AtomsList=AtomsList,
                BondOrderMatrix=BondOrderMatrix,
                DeriveAttributes=False,
                CheckMolObj=False,
            )
            molObj.NumberOfAtoms = NumberOfAtoms
            molObj.NumberOfBonds = NumberOfBonds
            molObj.NumberOfSubstructures = 0
            if template_molObj is None:
                molObj.AtomsList[0].FormalCharge = charge_mult[0]
                molObj.AtomsList[0].Multiplicity = charge_mult[1]
                molObj.FormalCharge = charge_mult[0]
                molObj.Multiplicity = charge_mult[1]
            # Get molecule energies
            if opt_step_idx + 1 == num_opt_step:
                check_final_energies = True
            calc_en_dict = _GetCalculatedEnergies(
                opt_step, check_final_energies=check_final_energies
            )
            molObj.electronic_energy = calc_en_dict["Electronic Energy"]
            molObj.enthalpy = calc_en_dict["Enthalpy"]
            molObj.entropy = calc_en_dict["Entropy"]
            molObj.gibbs_free_energy = calc_en_dict["Gibbs Free Energy"]
            molObj_list.append(molObj)
        return molObj_list

    def XYZFileToCoords(self, xyz_file: str) -> list[list[str]]:
        with open(xyz_file, "r") as f:
            xyz_file = f.read()
            f.close()
        return [
            [coor for coor in line.split(" ") if coor != ""]
            for line in xyz_file.split("\n")[2:]
        ]

    def ReadXYZFileMapCoords(self, xyz_file: str):
        xyz_file_list = self.XYZFileToCoords(xyz_file=xyz_file)
        for line, atomObj in zip(xyz_file_list, self.AtomsList):
            atomObj.Coordinates = np.array(
                [
                    float(line[1]),
                    float(line[2]),
                    float(line[3]),
                ]
            )

    # === Edit Molecule functions ===

    def AddAtom(
        self,
        AtomicSymbol: str,
        Coordinates: np.ndarray,
        Label: str | None,
        FormalCharge: int = 0,
        Multiplicity: int = 1,
        SubstructureIndex: int = 1,
        UpdateAtomLabels: bool = True,
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
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
        BondOrder: float = 1,
    ):
        if AtomIndices is not None:
            atomIdx1, atomIdx2 = AtomIndices
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
                        AtomIndices=[
                            new_atomIdx1,
                            new_atomIdx2,
                        ],
                        BondOrder=MoleculeToAdd.BondOrderMatrix[atomIdx1][atomIdx2],
                    )

    def RemoveMolecule(
        self,
        SMILES: str | None = None,
        SMARTS: str | None = None,
        SubstructureIndex: int | None = None,
    ):
        """
        SMILES: Checks to see if molecule is equivelent to SMILES
        SMARTS: Checks to see if molecule contains SMARTS
        """
        if SMILES is not None:
            for component in self.SplitMoleculeIntoComponents(UpdateAtomLabels=False):
                comp_SMILES = component.WriteSMILESString()
                with rdBase.BlockLogs():
                    if (
                        self.EquivelentMoleculeInchi(
                            SMILES,
                            comp_SMILES,
                        )
                        == True
                    ):
                        AtomLabels_to_remove = [
                            atomObj.Label for atomObj in component.AtomsList
                        ]
                        for AtomLabel in AtomLabels_to_remove:
                            self.RemoveAtom(
                                AtomLabel=AtomLabel,
                                UpdateAtomLabels=False,
                                UpdateSubstructureIndices=False,
                            )
        elif SMARTS is not None:
            for component in self.SplitMoleculeIntoComponents(UpdateAtomLabels=False):
                comp_SMILES = component.WriteSMILESString()
                with rdBase.BlockLogs():
                    matches = self.SMARTSMatchesSMILES(comp_SMILES, SMARTS)
                    if matches != ():
                        AtomLabels_to_remove = [
                            atomObj.Label for atomObj in component.AtomsList
                        ]
                        for AtomLabel in AtomLabels_to_remove:
                            self.RemoveAtom(
                                AtomLabel=AtomLabel,
                                UpdateAtomLabels=False,
                                UpdateSubstructureIndices=False,
                            )
        elif SubstructureIndex is not None:
            atom_labels = []
            for atomObj in self.AtomsList:
                if atomObj.SubstructureIndex == SubstructureIndex:
                    atom_labels.append(atomObj.Label)
            for atom_label in atom_labels:
                self.RemoveAtom(
                    AtomLabel=atom_label,
                    UpdateAtomLabels=False,
                    UpdateSubstructureIndices=False,
                )
        self.DeriveBasicAttributes()

    def RemoveBond(
        self,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
    ):
        """
        Remove a bond between two atoms in the molecule.

        Parameters:
            AtomLabels (list[str] | None): Labels of the two atoms (e.g., ['H1', 'C1'])
            AtomIndices (list[int] | None): Indices of the two atoms in AtomsList
            AtomObjects (list[Atom] | None): Direct references to the two Atom objects

        Raises:
            ValueError: If no identifier provided or invalid atom specification
            IndexError: If atom indices are out of bounds
            ValueError: If no bond exists between the specified atoms

        Notes:
            - Removing a bond may change the number of substructures if it disconnects
            previously bonded atoms
            - Derived attributes (NumberOfBonds, NumberOfSubstructures) are updated automatically
        """
        # Determine atom indices
        if AtomIndices is not None:
            if len(AtomIndices) != 2:
                raise ValueError("AtomIndices must contain exactly 2 indices")
            atomIdx1, atomIdx2 = AtomIndices
            if not (
                0 <= atomIdx1 < self.NumberOfAtoms
                and 0 <= atomIdx2 < self.NumberOfAtoms
            ):
                raise IndexError(f"Atom indices out of bounds: {atomIdx1}, {atomIdx2}")
        elif AtomLabels is not None:
            if len(AtomLabels) != 2:
                raise ValueError("AtomLabels must contain exactly 2 labels")
            if AtomLabels[0] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[0]}' not found")
            if AtomLabels[1] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[1]}' not found")
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
        elif AtomObjects is not None:
            if len(AtomObjects) != 2:
                raise ValueError("AtomObjects must contain exactly 2 objects")
            try:
                atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
                atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
            except KeyError as e:
                raise ValueError(f"Atom object not found in molecule: {e}")
        else:
            raise ValueError(
                "RemoveBond requires AtomLabels, AtomIndices, or AtomObjects"
            )

        # Check if bond exists
        if self.BondOrderMatrix[atomIdx1][atomIdx2] == 0:
            raise ValueError(
                f"No bond exists between atoms at indices {atomIdx1} and {atomIdx2}"
            )

        # Remove bond by setting to 0
        self.BondOrderMatrix[atomIdx1][atomIdx2] = 0
        self.BondOrderMatrix[atomIdx2][atomIdx1] = 0
        self.ConnectivityMatrix[atomIdx1][atomIdx2] = 0
        self.ConnectivityMatrix[atomIdx2][atomIdx1] = 0
        self.NumberOfBonds -= 1
        self.NormaliseSubstructureIndicies()

    def RemoveAtom(
        self,
        AtomLabel: str | None = None,
        AtomIndex: int | None = None,
        AtomObject: Atom | None = None,
        UpdateAtomLabels: bool = True,
        UpdateSubstructureIndices: bool = True,
    ):
        """
        Remove an atom from the molecule and all its associated bonds.

        Parameters:
            AtomLabel (str | None): Label of the atom to remove (e.g., 'H1', 'C2')
            AtomIndex (int | None): Index of the atom in AtomsList
            AtomObject (Atom | None): Direct reference to the Atom object

        Raises:
            ValueError: If no identifier provided or atom not found in molecule
            IndexError: If AtomIndex is out of bounds

        Notes:
            - Removing an atom automatically removes all bonds involving it
            - Substructure indices are recalculated after removal
            - Derived attributes (MolecularMass, NumberOfBonds, etc.) are updated
        """
        # Determine which atom to remove
        if AtomIndex is not None:
            if not 0 <= AtomIndex < self.NumberOfAtoms:
                raise IndexError(
                    f"Atom index {AtomIndex} out of bounds (0-{self.NumberOfAtoms-1})"
                )
            atom_idx_to_remove = AtomIndex
        elif AtomLabel is not None:
            if AtomLabel not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabel}' not found in molecule")
            atom_idx_to_remove = self.AtomsDict[AtomLabel][0]
        elif AtomObject is not None:
            try:
                atom_idx_to_remove = self.AtomsDict[AtomObject.Label][0]
            except KeyError:
                raise ValueError(
                    f"Atom object with label '{AtomObject.Label}' not found in molecule"
                )
        else:
            raise ValueError("Must provide AtomLabel, AtomIndex, or AtomObject")

        # Remove atom from AtomsList
        self.AtomsList.pop(atom_idx_to_remove)

        # Remove row and column from bond order matrix
        self.BondOrderMatrix = np.delete(
            self.BondOrderMatrix, atom_idx_to_remove, axis=0
        )
        self.BondOrderMatrix = np.delete(
            self.BondOrderMatrix, atom_idx_to_remove, axis=1
        )

        # Recalculate all derived attributes
        self.DeriveBasicAttributes(
            UpdateAtomLabels=UpdateAtomLabels,
            UpdateSubstructureIndices=UpdateSubstructureIndices,
        )

    def ChangeAtom(
        self,
        NewAtomicSymbol: str,
        NewFormalCharge: int = 0,
        NewMultiplicity: int = 1,
        AtomLabel: str | None = None,
        AtomIndex: int | None = None,
        UpdateAtomLabels: bool = True,
    ):
        """
        Change the atomic symbol of an atom in the molecule.

        Parameters:
            NewAtomicSymbol (str): The new atomic symbol (e.g., 'C', 'N', 'O')
            AtomLabel (str | None): Label of the atom to change (e.g., 'H1', 'C2')
            AtomIndex (int | None): Index of the atom to change in AtomsList

        Raises:
            ValueError: If neither AtomLabel nor AtomIndex provided, if AtomLabel not found,
                    if AtomIndex out of bounds, or if NewAtomicSymbol is invalid
        """

        # Validate that exactly one identifier is provided
        if AtomLabel is None and AtomIndex is None:
            raise ValueError("Must provide either AtomLabel or AtomIndex")

        # Get atom object
        if AtomLabel is not None:
            if AtomLabel not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabel}' not found in molecule")
            atomObj = self.AtomsDict[AtomLabel][1]
        else:  # AtomIndex is not None
            if not 0 <= AtomIndex < self.NumberOfAtoms:
                raise ValueError(
                    f"Atom index {AtomIndex} out of bounds (0-{self.NumberOfAtoms-1})"
                )
            atomObj = self.AtomsList[AtomIndex]

        # Change atom and update molecular properties
        atomObj.AtomicSymbol = NewAtomicSymbol
        atomObj.Update()
        self.DeriveBasicAttributes(
            UpdateAtomLabels=UpdateAtomLabels
        )  # Updates MolecularMass

    def ChangeBond(
        self,
        NewBondOrder: float,
        AtomLabels: list[str] | None = None,
        AtomIndices: list[int] | None = None,
        AtomObjects: list[Atom] | None = None,
    ):
        """
        Change the bond order between two atoms in the molecule.

        Parameters:
            NewBondOrder (float): The new bond order (e.g., 1.0, 1.5, 2.0, 3.0)
                                 - 1.0: Single bond
                                 - 1.5: Aromatic bond
                                 - 2.0: Double bond
                                 - 3.0: Triple bond
            AtomLabels (list[str] | None): Labels of the two atoms (e.g., ['C1', 'C2'])
            AtomIndices (list[int] | None): Indices of the two atoms in AtomsList
            AtomObjects (list[Atom] | None): Direct references to the two Atom objects

        Raises:
            ValueError: If no identifier provided, invalid atom specification, or no bond exists
            IndexError: If atom indices are out of bounds
            ValueError: If NewBondOrder is invalid (negative or zero)

        Notes:
            - Setting NewBondOrder to 0 is equivalent to RemoveBond()
            - Changing bond order does not affect substructure connectivity (only presence/absence)
            - Derived attributes are minimally updated for efficiency
        """

        # Validate bond order
        if NewBondOrder <= 0:
            raise ValueError(f"Bond order must be positive, got {NewBondOrder}")

        # Determine atom indices
        if AtomIndices is not None:
            if len(AtomIndices) != 2:
                raise ValueError("AtomIndices must contain exactly 2 indices")
            atomIdx1, atomIdx2 = AtomIndices
            if not (
                0 <= atomIdx1 < self.NumberOfAtoms
                and 0 <= atomIdx2 < self.NumberOfAtoms
            ):
                raise IndexError(f"Atom indices out of bounds: {atomIdx1}, {atomIdx2}")
        elif AtomLabels is not None:
            if len(AtomLabels) != 2:
                raise ValueError("AtomLabels must contain exactly 2 labels")
            if AtomLabels[0] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[0]}' not found")
            if AtomLabels[1] not in self.AtomsDict:
                raise ValueError(f"Atom label '{AtomLabels[1]}' not found")
            atomIdx1 = self.AtomsDict[AtomLabels[0]][0]
            atomIdx2 = self.AtomsDict[AtomLabels[1]][0]
        elif AtomObjects is not None:
            if len(AtomObjects) != 2:
                raise ValueError("AtomObjects must contain exactly 2 objects")
            try:
                atomIdx1 = self.AtomsDict[AtomObjects[0].Label][0]
                atomIdx2 = self.AtomsDict[AtomObjects[1].Label][0]
            except KeyError as e:
                raise ValueError(f"Atom object not found in molecule: {e}")
        else:
            raise ValueError(
                "ChangeBond requires AtomLabels, AtomIndices, or AtomObjects"
            )

        # Check if bond exists
        if self.BondOrderMatrix[atomIdx1][atomIdx2] == 0:
            raise ValueError(
                f"No bond exists between atoms at indices {atomIdx1} and {atomIdx2}"
            )

        # Update bond order
        self.BondOrderMatrix[atomIdx1][atomIdx2] = NewBondOrder
        self.BondOrderMatrix[atomIdx2][atomIdx1] = NewBondOrder

    # === Translate and Rotate Molecule ===

    def TranslateMolecule(
        self,
        TranslationVector: np.ndarray,
        Displacement: float,
    ):
        TranslationVector = TranslationVector / np.linalg.norm(TranslationVector)
        TranslationVector = TranslationVector * abs(Displacement)
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
            rotation_axis=RotationVector / np.linalg.norm(RotationVector),
            theta=RotationAngle,
        )
        for atomObj in self.AtomsList:
            atomObj.Coordinates = RotationMatrix @ atomObj.Coordinates
        # Translate back to original position
        for atomObj in self.AtomsList:
            atomObj.Coordinates = atomObj.Coordinates + geometric_midpoint

    # === Optimise Geometries Functions ===

    def LennardJonesPotential(
        self,
        sigma_a: float,
        sigma_b: float,
        coordinates_a: np.array,
        coordinates_b: np.array,
        epsilon_a=1,
        epsilon_b=1,
    ):
        epsilon = (epsilon_a * epsilon_b) ** 0.5
        sigma = (sigma_a + sigma_b) / 2
        r = np.linalg.norm(coordinates_a - coordinates_b)
        V_r = 4 * epsilon * (((sigma / r) ** 12) - ((sigma / r) ** 6))
        return V_r

    def LennardJonesGradient(
        self,
        sigma_a: float,
        sigma_b: float,
        coordinates_a: np.array,
        coordinates_b: np.array,
        epsilon_a=1,
        epsilon_b=1,
        step=0.1,
    ):
        # Calculate energy gradient between coordinates
        direction_vector = coordinates_a - coordinates_b
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        translation_vector = direction_vector * step
        step0_LJPotEn = self.LennardJonesPotential(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            coordinates_a=coordinates_a,
            coordinates_b=coordinates_b,
            epsilon_a=epsilon_a,
            epsilon_b=epsilon_b,
        )
        step1_LJPotEn = self.LennardJonesPotential(
            sigma_a=sigma_a,
            sigma_b=sigma_b,
            coordinates_a=coordinates_a,
            coordinates_b=coordinates_b + translation_vector,
            epsilon_a=epsilon_a,
            epsilon_b=epsilon_b,
        )
        gradient = step0_LJPotEn - step1_LJPotEn
        return gradient, direction_vector

    def CalculateTotalLennardJonesPotential(
        self,
        ForcesDict: dict,
    ) -> float:
        total_LJ_pot = 0
        for Identifier1 in ForcesDict:
            for Identifier2 in ForcesDict:
                if Identifier1 == Identifier2:
                    continue
                LJ_pot = self.LennardJonesPotential(
                    sigma_a=ForcesDict[Identifier1]["Radius"],
                    sigma_b=ForcesDict[Identifier2]["Radius"],
                    coordinates_a=ForcesDict[Identifier1]["Centre of Mass"],
                    coordinates_b=ForcesDict[Identifier2]["Centre of Mass"],
                )
                total_LJ_pot += LJ_pot
        return total_LJ_pot

    def MoveSubStructures_SimpleLJ(
        self,
        ForcesDict: dict,
    ):
        for Identifier in ForcesDict:
            translation_vector = (
                ForcesDict[Identifier]["Displacement"]
                * ForcesDict[Identifier]["Direction"]
                * -1
            )
            for atomObj in self.AtomsList:
                if str(atomObj.SubstructureIndex) == Identifier.split("_")[-1]:
                    atomObj.Coordinates += translation_vector

    def ForcesDict_SimpleLJ(
        self, max_step_size: float = 0.1, time_step: float = 1.0
    ) -> dict:
        ForcesDict = {}
        components = self.SplitMoleculeIntoComponents(UpdateAtomLabels=False)
        for component in components:
            ForcesDict[component.Identifier] = {
                "Centre of Mass": component.GetCentreOfMass(),
                "Radius": component.GetMoleculeRadius(),
                "Molecular Mass": component.MolecularMass,
            }
        # Append forces dict with LJ potential gradient and direction of force
        for Identifier1 in ForcesDict:
            force_vector = np.array([0.0, 0.0, 0.0])
            for Identifier2 in ForcesDict:
                if Identifier1 == Identifier2:
                    continue
                else:
                    grad, dir_vec = self.LennardJonesGradient(
                        sigma_a=ForcesDict[Identifier1]["Radius"],
                        sigma_b=ForcesDict[Identifier2]["Radius"],
                        coordinates_a=ForcesDict[Identifier1]["Centre of Mass"],
                        coordinates_b=ForcesDict[Identifier2]["Centre of Mass"],
                    )
                    force_vector += dir_vec * grad
            force_mag = np.linalg.norm(force_vector)
            distance_travel = (force_mag * (time_step**2)) / ForcesDict[Identifier1][
                "Molecular Mass"
            ]
            if distance_travel > max_step_size:
                distance_travel = max_step_size
            ForcesDict[Identifier1]["Displacement"] = distance_travel
            ForcesDict[Identifier1]["Direction"] = force_vector / np.linalg.norm(
                force_vector
            )
        return ForcesDict

    def OptimiseGeometry_SimpleLJ(
        self,
        n_steps: int = 100,
        max_en_diff: float = 1e-1,
        max_step_size: float = 0.1,
        time_step: float = 1.0,
    ) -> float:
        # Calculate initial forces that the substructures place on each other
        ForcesDict = self.ForcesDict_SimpleLJ()
        OG_LJ_en = self.CalculateTotalLennardJonesPotential(
            ForcesDict=ForcesDict,
        )
        for _ in range(n_steps):
            self.MoveSubStructures_SimpleLJ(
                ForcesDict=ForcesDict,
            )
            ForcesDict = self.ForcesDict_SimpleLJ(
                max_step_size=max_step_size, time_step=time_step
            )
            NEW_LJ_en = self.CalculateTotalLennardJonesPotential(
                ForcesDict=ForcesDict,
            )
            en_diff = abs(OG_LJ_en - NEW_LJ_en)
            if en_diff < max_en_diff:
                break
            OG_LJ_en = NEW_LJ_en
        return OG_LJ_en

    def OptimiseGeometry_UFF(
        self,
        fixed_atoms: list[int] | None = None,
        max_steps: int = 700,
        energy_tol: float = 1e-6,
        force_field: str = "UFF",
        suppress_warnings: bool = True,
    ) -> float:
        if suppress_warnings:
            ob.obErrorLog.SetOutputLevel(0)
        # Read pybel file
        temp_mol_str = self.WriteMolString()
        with open(f"{Path(__file__).parent}/{self.Identifier}_temp.mol", "w") as f:
            f.write(temp_mol_str)
            f.close()
        molPybelObj = pybel.readfile(
            "mol", f"{Path(__file__).parent}/{self.Identifier}_temp.mol"
        )
        molPybelObj = next(molPybelObj)
        os.remove(f"{Path(__file__).parent}/{self.Identifier}_temp.mol")
        # define which bonds and atoms are aromatic
        obmol = molPybelObj.OBMol
        obmol.PerceiveBondOrders()
        obmol.SetAromaticPerceived(False)
        # Set up constraints
        if fixed_atoms:
            constrs = ob.OBFFConstraints()
            for atom_idx in fixed_atoms:
                constrs.AddAtomConstraint(atom_idx + 1)
        # Set up force field
        ff = ob.OBForceField.FindForceField(force_field)
        if not ff:
            raise ValueError(f"Could not find {force_field} forcefield")
        # Setup minimization
        if fixed_atoms:
            ff.Setup(molPybelObj.OBMol, constrs)
            ff.SetConstraints(constrs)
        else:
            ff.Setup(molPybelObj.OBMol)
        # Run minimization
        max_steps = int((max_steps) / 4) + 1
        ff.ConjugateGradients(max_steps, energy_tol)
        ff.SteepestDescent(max_steps, energy_tol)
        ff.ConjugateGradients(max_steps, energy_tol)
        ff.SteepestDescent(max_steps, energy_tol)
        # Update coordinates
        ff.GetCoordinates(molPybelObj.OBMol)
        for ob_atom, atomObj in zip(molPybelObj, self.AtomsList):
            atomObj.Coordinates = np.array(ob_atom.coords)
        return ff.Energy()

    def OptimiseGeometry_gxTB(
        self,
        xtb_binary_path: str,
        solvent_model: str | None = None,
        solvent: str | None = None,
        opt_tol: str | None = None,
        opt_cycles: int | None = None,
        xtb_method: str = "gxtb",
        fixed_atoms: list[int] | None = None,
    ):
        xyz_string = self.WriteXYZString()
        cmd = [
            f"{xtb_binary_path}/xtb",
            f"{Path(__file__).parent}/{self.Identifier}_temp.xyz",
        ]
        with open(f"{Path(__file__).parent}/{self.Identifier}_temp.xyz", "w") as f:
            f.write(xyz_string)
            f.close()

        if fixed_atoms:
            atom_string = ""
            for atom_idx in fixed_atoms[:-1]:
                atom_string += f"{int(atom_idx+1)}, "
            atom_string += f"{int(fixed_atoms[-1]+1)}"
            input_string = f"""$fix
    atoms: {atom_string}
$end
"""
            with open(f"{Path(__file__).parent}/xtb.inp", "w") as f:
                f.write(input_string)
                f.close()
            cmd += [
                "--input",
                f"{Path(__file__).parent}/xtb.inp",
            ]

        cmd += [
            f"--{xtb_method}",
            "--opt",
            opt_tol,
            "--charge",
            str(self.FormalCharge),
            "--uhf",
            str(self.Multiplicity - 1),
        ]

        if solvent_model is not None and solvent is not None:
            cmd += [
                f"--{solvent_model}",
                solvent,
            ]
        if opt_cycles is not None:
            cmd += [
                "--cycles",
                str(opt_cycles),
            ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=f"{Path(__file__).parent}",
        )
        if result.returncode != 0:
            print(result.stderr)
            return result.stdout

        # Read output xyz files and update coordinates
        self.ReadXYZFileMapCoords(xyz_file=f"{Path(__file__).parent}/xtbopt.xyz")

        # Remove all output files
        for stringObj in [
            "charges",
            "energy",
            "gradient",
            "wbo",
            "xtbrestart",
            "xtbtopo.mol",
            "temp_input_xtb.engrad",
            "temp_input_xtb.xyz",
            "xtblast.xyz",
            "xtbopt.log",
            "xtbopt.xyz",
            ".xtboptok",
            "xtb.inp",
            f"{self.Identifier}_temp.xyz",
        ]:
            try:
                os.remove(f"{Path(__file__).parent}/{stringObj}")
            except FileNotFoundError:
                pass
            except PermissionError:
                pass

    def OptimiseGeometry(
        self,
        SimpleLennardJonesPotential: bool | None = None,
        SimpleLennardJonesPotential_settings: dict | None = None,
        MolecularMechanics: bool | None = None,
        MolecularMechanics_settings: dict | None = None,
        SemiEmpiricalxTB: bool | None = None,
        SemiEmpiricalxTB_settings: dict | None = None,
        xtb_binary_path: str | None = None,
    ):
        lj_defaults = {
            "Max Steps": 100,
            "Max Energy Difference": 1e-1,
            "Max Step Size": 0.1,
            "Time Step": 1,
        }
        mm_defaults = {
            "Max Steps": 700,
            "Max Energy Difference": 1e-6,
            "Method": "UFF",
            "ConstrainedAtomLabels": None,
            "ConstrainedAtomIndices": None,
        }
        xtb_defaults = {
            "Solvent Model": None,
            "Solvent": None,
            "Optimisation Level": "tight",
            "Optimisation Cycles": None,
            "xTB Method": "gxtb",
            "ConstrainedAtomLabels": None,
            "ConstrainedAtomIndices": None,
        }

        lj_settings = {**lj_defaults, **(SimpleLennardJonesPotential_settings or {})}
        mm_settings = {**mm_defaults, **(MolecularMechanics_settings or {})}
        xtb_settings = {**xtb_defaults, **(SemiEmpiricalxTB_settings or {})}
        if SimpleLennardJonesPotential == True:
            self.OptimiseGeometry_SimpleLJ(
                n_steps=lj_settings["Max Steps"],
                max_en_diff=lj_settings["Max Energy Difference"],
                max_step_size=lj_settings["Max Step Size"],
                time_step=lj_settings["Time Step"],
            )
        if MolecularMechanics == True:
            if mm_settings["ConstrainedAtomLabels"] is not None:
                fixed_atoms = [
                    self.AtomsDict[Label][0]
                    for Label in mm_defaults["ConstrainedAtomIndices"]
                ]
            elif mm_settings["ConstrainedAtomIndices"] is not None:
                fixed_atoms = mm_settings["ConstrainedAtomIndices"]
            else:
                fixed_atoms = None
            self.OptimiseGeometry_UFF(
                fixed_atoms=fixed_atoms,
                max_steps=mm_settings["Max Steps"],
                energy_tol=mm_settings["Max Energy Difference"],
                force_field=mm_settings["Method"],
            )
        if SemiEmpiricalxTB == True:
            if xtb_settings["ConstrainedAtomLabels"] is not None:
                fixed_atoms = [
                    self.AtomsDict[Label][0]
                    for Label in mm_defaults["ConstrainedAtomIndices"]
                ]
            elif xtb_settings["ConstrainedAtomIndices"] is not None:
                fixed_atoms = mm_settings["ConstrainedAtomIndices"]
            else:
                fixed_atoms = None
            self.OptimiseGeometry_gxTB(
                xtb_binary_path=xtb_binary_path,
                solvent_model=xtb_settings["Solvent Model"],
                solvent=xtb_settings["Solvent"],
                opt_tol=xtb_settings["Optimisation Level"],
                opt_cycles=xtb_settings["Optimisation Cycles"],
                xtb_method=xtb_settings["xTB Method"],
                fixed_atoms=fixed_atoms,
            )
