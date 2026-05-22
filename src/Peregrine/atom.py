import numpy as np


class Atom:
    def __init__(
        self,
        Label: str,
        AtomicSymbol: str,
        Coordinates: np.ndarray,
        FormalCharge: int = 0,
        Multiplicity: int = 1,
        SubstructureIndex: int = 1,
    ):
        """
        Initialize an Atom instance.

        Parameters:
            Label (str): Label of the atom.
            AtomicSymbol (str): Atomic symbol of the atom.
            Coordinates (np.ndarray): 3D coordinates of the atom as a simple vector array.
            FormalCharge (int): Formal charge of the atom.
            Multiplicity (int): Spin multiplicity of the atom (2S+1).
            SubstructureIndex (int): Index of the substructure to which the atom belongs.
        """
        self.Label = Label
        self.Coordinates = Coordinates
        self.AtomicSymbol = AtomicSymbol
        self.FormalCharge = FormalCharge
        self.SubstructureIndex = SubstructureIndex
        self.Multiplicity = Multiplicity
