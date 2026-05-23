import numpy as np

from Peregrine.molecule import Molecule
from Peregrine.atom import Atom


def test_atom_initialization():
    """Test basic atom initialization with required parameters."""
    atomObj = Atom(
        Label="H1",
        AtomicSymbol="H",
        Coordinates=np.array([0, 0, 0]),
    )
    assert atomObj.Label == "H1"
    assert atomObj.AtomicSymbol == "H"
    np.testing.assert_array_equal(atomObj.Coordinates, np.array([0, 0, 0]))
    assert atomObj.FormalCharge == 0
    assert atomObj.Multiplicity == 1
    assert atomObj.SubstructureIndex == 1


def test_atom_with_optional_params():
    """Test atom initialization with optional parameters."""
    atomObj = Atom(
        Label="O1",
        AtomicSymbol="O",
        Coordinates=np.array([1.0, 2.0, 3.0]),
        FormalCharge=-2,
        Multiplicity=3,
        SubstructureIndex=2,
    )
    assert atomObj.FormalCharge == -2
    assert atomObj.Multiplicity == 3
    assert atomObj.SubstructureIndex == 2


def test_molecule_initialization():
    """Test basic molecule initialization with a water molecule"""
    molObj = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(Label="H1", AtomicSymbol="H", Coordinates=np.array([-1, 0, 0])),
            Atom(Label="O1", AtomicSymbol="O", Coordinates=np.array([0, 0, 0])),
            Atom(Label="H2", AtomicSymbol="H", Coordinates=np.array([0, 1, 0])),
        ],
        BondMatrix=None,
    )
    assert molObj.Identifier == "Water"
    assert molObj.NumberOfAtoms == 3
    assert molObj.NumberOfBonds == 0
    assert len(molObj.AtomsDict) == 3
    assert molObj.BondMatrix.shape == (3, 3)
    assert molObj.ConnectivityMatrix.shape == (3, 3)
    assert molObj.NumberOfSubstructures == 3

    molObj = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(Label="H1", AtomicSymbol="H", Coordinates=np.array([-1, 0, 0])),
            Atom(Label="O1", AtomicSymbol="O", Coordinates=np.array([0, 0, 0])),
            Atom(Label="H2", AtomicSymbol="H", Coordinates=np.array([0, 1, 0])),
        ],
        BondMatrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert molObj.NumberOfSubstructures == 2

    molObj = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(Label="H1", AtomicSymbol="H", Coordinates=np.array([-1, 0, 0])),
            Atom(Label="O1", AtomicSymbol="O", Coordinates=np.array([0, 0, 0])),
            Atom(Label="H2", AtomicSymbol="H", Coordinates=np.array([0, 1, 0])),
        ],
        BondMatrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
    )
    assert molObj.NumberOfSubstructures == 1
