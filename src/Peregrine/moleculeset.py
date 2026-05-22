import numpy as np

class Atom:
    def __init__(
        self,
        Label: str,
        Coordinates: np.ndarray,
        SybylType: str,
        AtomicSymbol: str,
        FormalCharge: int = 0,
        Multiplicity: int = 1,
        SubstructureIndex: int = 1,
        SubstructureName: str = "SUB1",
    ):
