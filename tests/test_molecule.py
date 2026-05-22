import numpy as np

from Peregrine.molecule import Molecule
from Peregrine.atom import Atom


def test_atom_1():
    atom = Atom(
        Label="H1",
        AtomicSymbol="H",
        Coordinates=np.array([0, 0, 0]),
    )
