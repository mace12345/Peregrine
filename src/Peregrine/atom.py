import numpy as np

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

ATOMIC_MASSES = {
    "H": 1.008,
    "He": 4.0026,
    "Li": 6.94,
    "Be": 9.0122,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "Ne": 20.180,
    "Na": 22.990,
    "Mg": 24.305,
    "Al": 26.982,
    "Si": 28.085,
    "P": 30.974,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.95,
    "K": 39.098,
    "Ca": 40.078,
    "Sc": 44.956,
    "Ti": 47.867,
    "V": 50.942,
    "Cr": 51.996,
    "Mn": 54.938,
    "Fe": 55.845,
    "Co": 58.933,
    "Ni": 58.693,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.630,
    "As": 74.922,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.468,
    "Sr": 87.62,
    "Y": 88.906,
    "Zr": 91.224,
    "Nb": 92.906,
    "Mo": 95.95,
    "Tc": 97.0,
    "Ru": 101.07,
    "Rh": 102.91,
    "Pd": 106.42,
    "Ag": 107.87,
    "Cd": 112.41,
    "In": 114.82,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.60,
    "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91,
    "Ba": 137.33,
    "La": 138.91,
    "Ce": 140.12,
    "Pr": 140.91,
    "Nd": 144.24,
    "Pm": 145.0,
    "Sm": 150.36,
    "Eu": 151.96,
    "Gd": 157.25,
    "Tb": 158.93,
    "Dy": 162.50,
    "Ho": 164.93,
    "Er": 167.26,
    "Tm": 168.93,
    "Yb": 173.05,
    "Lu": 174.97,
    "Hf": 178.49,
    "Ta": 180.95,
    "W": 183.84,
    "Re": 186.21,
    "Os": 190.23,
    "Ir": 192.22,
    "Pt": 195.08,
    "Au": 196.97,
    "Hg": 200.59,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.98,
    "Po": 209.0,
    "At": 210.0,
    "Rn": 222.0,
    "Fr": 223.0,
    "Ra": 226.0,
    "Ac": 227.0,
    "Th": 232.04,
    "Pa": 231.04,
    "U": 238.03,
    "Np": 237.0,
    "Pu": 244.0,
    "Am": 243.0,
    "Cm": 247.0,
    "Bk": 247.0,
    "Cf": 251.0,
    "Es": 252.0,
    "Fm": 257.0,
    "Md": 258.0,
    "No": 259.0,
    "Lr": 266.0,
    "Rf": 267.0,
    "Db": 268.0,
    "Sg": 269.0,
    "Bh": 270.0,
    "Hs": 269.0,
    "Mt": 278.0,
    "Ds": 281.0,
    "Rg": 282.0,
    "Cn": 285.0,
    "Nh": 286.0,
    "Fl": 289.0,
    "Mc": 289.0,
    "Lv": 293.0,
    "Ts": 294.0,
    "Og": 294.0,
}

ATOMIC_VDW_RADII = {
    "H": 1.20,
    "He": 1.43,
    "Li": 2.12,
    "Be": 1.98,
    "B": 1.91,
    "C": 1.77,
    "N": 1.66,
    "O": 1.50,
    "F": 1.46,
    "Ne": 1.58,
    "Na": 2.50,
    "Mg": 2.51,
    "Al": 2.25,
    "Si": 2.19,
    "P": 1.90,
    "S": 1.89,
    "Cl": 1.82,
    "Ar": 1.83,
    "K": 2.73,
    "Ca": 2.62,
    "Sc": 2.58,
    "Ti": 2.46,
    "V": 2.42,
    "Cr": 2.45,
    "Mn": 2.45,
    "Fe": 2.44,
    "Co": 2.40,
    "Ni": 2.40,
    "Cu": 2.38,
    "Zn": 2.39,
    "Ga": 2.32,
    "Ge": 2.29,
    "As": 1.88,
    "Se": 1.82,
    "Br": 1.86,
    "Kr": 2.25,
    "Rb": 3.21,
    "Sr": 2.84,
    "Y": 2.75,
    "Zr": 2.52,
    "Nb": 2.56,
    "Mo": 2.45,
    "Tc": 2.44,
    "Ru": 2.46,
    "Rh": 2.44,
    "Pd": 2.15,
    "Ag": 2.53,
    "Cd": 2.49,
    "In": 2.43,
    "Sn": 2.42,
    "Sb": 2.47,
    "Te": 1.99,
    "I": 2.04,
    "Xe": 2.06,
    "Cs": 3.48,
    "Ba": 3.03,
    "La": 2.98,
    "Ce": 2.88,
    "Pr": 2.92,
    "Nd": 2.95,
    "Sm": 2.90,
    "Eu": 2.87,
    "Gd": 2.83,
    "Tb": 2.79,
    "Dy": 2.87,
    "Ho": 2.81,
    "Er": 2.83,
    "Tm": 2.79,
    "Yb": 2.80,
    "Lu": 2.74,
    "Hf": 2.63,
    "Ta": 2.53,
    "W": 2.57,
    "Re": 2.49,
    "Os": 2.48,
    "Ir": 2.41,
    "Pt": 2.29,
    "Au": 2.32,
    "Hg": 2.45,
    "Tl": 2.47,
    "Pb": 2.60,
    "Bi": 2.54,
    "Ac": 2.80,
    "Th": 2.93,
    "Pa": 2.88,
    "U": 2.71,
    "Np": 2.82,
    "Pu": 2.81,
    "Am": 2.83,
    "Cm": 3.05,
    "Bk": 3.40,
    "Cf": 3.05,
    "Es": 2.70,
}


class Atom:
    def __init__(
        self,
        AtomicSymbol: str,
        Coordinates: np.ndarray,
        Label: str | None,
        FormalCharge: int = 0,
        Multiplicity: int = 1,
        SubstructureIndex: int = 1,
    ):
        """
        Initialize an Atom instance.

        Creates a new Atom object representing a single atom in a molecular structure
        with its position, properties, and connectivity information.

        Parameters:
            Label (str): Unique identifier/label for the atom (e.g., "H1", "C2", "O_alpha").
            AtomicSymbol (str): Chemical element symbol (e.g., "H", "C", "N", "O").
                               Must be a valid element in ATOMIC_MASSES and ATOMIC_NUMBERS.
            Coordinates (np.ndarray): 3D coordinates as a numpy array [x, y, z] in Angstroms.
            FormalCharge (int, optional): Formal charge on the atom. Default is 0.
            Multiplicity (int, optional): Spin multiplicity (2S+1) of the atom.
                                         Default is 1 (singlet/paired electrons).
            SubstructureIndex (int, optional): Index of the molecular substructure/fragment
                                             to which the atom belongs. Default is 1.
                                             Automatically reassigned by Molecule.NormaliseSubstructureIndicies().

        Attributes (auto-populated):
            AtomicMass (float): Atomic mass in amu, retrieved from ATOMIC_MASSES.
            AtomicNumber (int): Atomic number (proton count), retrieved from ATOMIC_NUMBERS.

        Raises:
            KeyError: If AtomicSymbol is not found in ATOMIC_MASSES or ATOMIC_NUMBERS.

        Examples:
            atom_h = Atom("H1", "H", np.array([0.0, 0.0, 0.0]))
            atom_c = Atom("C1", "C", np.array([1.5, 0.0, 0.0]), FormalCharge=0)
            atom_o = Atom("O1", "O", np.array([3.0, 0.0, 0.0]), FormalCharge=-1)

        Notes:
            - Coordinates should be in Angstroms (typical for molecular structures).
            - Multiplicity (2S+1) for common cases: 1 (all paired), 2 (1 unpaired), 3 (2 unpaired).
            - SubstructureIndex is initially set but will be recalculated when added to a Molecule.
        """
        self.Label = Label
        self.Coordinates = Coordinates
        self.AtomicSymbol = AtomicSymbol
        self.FormalCharge = FormalCharge
        self.SubstructureIndex = SubstructureIndex
        self.Multiplicity = Multiplicity
        self.AssociatedAtomSMILES = None
        self.Update()

    def Update(self):
        """
        Refresh atomic properties from lookup tables.

        Re-retrieves the atomic mass and atomic number for this atom based on its
        AtomicSymbol. Useful if the ATOMIC_MASSES or ATOMIC_NUMBERS dictionaries
        have been modified after the atom was created, or to ensure the values are
        current if external data has changed.

        Returns:
            None (modifies instance attributes in-place)

        Raises:
            KeyError: If AtomicSymbol is not found in ATOMIC_MASSES or ATOMIC_NUMBERS.

        Notes:
            - This method is automatically called during __init__(), so manual calls
              are rarely necessary.
            - Typically only needed if you modify the ATOMIC_MASSES or ATOMIC_NUMBERS
              dictionaries after instantiation.
        """
        self.AtomicMass = ATOMIC_MASSES[self.AtomicSymbol]
        self.AtomicNumber = ATOMIC_NUMBERS[self.AtomicSymbol]
        self.AtomicRadii = ATOMIC_VDW_RADII[self.AtomicSymbol]
