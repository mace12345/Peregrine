import numpy as np
from pathlib import Path
from copy import deepcopy

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
        BondOrderMatrix=None,
    )
    assert molObj.Identifier == "Water"
    assert molObj.NumberOfAtoms == 3
    assert molObj.NumberOfBonds == 0
    assert len(molObj.AtomsDict) == 3
    assert molObj.BondOrderMatrix.shape == (3, 3)
    assert molObj.ConnectivityMatrix.shape == (3, 3)
    assert molObj.NumberOfSubstructures == 3

    molObj = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(Label="H1", AtomicSymbol="H", Coordinates=np.array([-1, 0, 0])),
            Atom(Label="O1", AtomicSymbol="O", Coordinates=np.array([0, 0, 0])),
            Atom(Label="H2", AtomicSymbol="H", Coordinates=np.array([0, 1, 0])),
        ],
        BondOrderMatrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]
        ),
    )
    assert molObj.NumberOfSubstructures == 2
    assert molObj.NumberOfBonds == 1
    assert molObj.FormalCharge == 0
    assert molObj.Multiplicity == 1

    molObj = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(Label="H1", AtomicSymbol="H", Coordinates=np.array([-1, 0, 0])),
            Atom(
                Label="O1",
                AtomicSymbol="O",
                Coordinates=np.array([0, 0, 0]),
                Multiplicity=2,
                FormalCharge=1,
            ),
            Atom(Label="H2", AtomicSymbol="H", Coordinates=np.array([0, 1, 0])),
        ],
        BondOrderMatrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
    )
    assert molObj.FormalCharge == 1
    assert molObj.Multiplicity == 2

    molObj = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(
                Label="H1", AtomicSymbol="H", Coordinates=np.array([-1.09, 0.05, 0.02])
            ),
            Atom(
                Label="O1",
                AtomicSymbol="O",
                Coordinates=np.array([0.00123, 0.0034, 0.0087]),
                FormalCharge=1,
            ),
            Atom(
                Label="H2",
                AtomicSymbol="H",
                Coordinates=np.array([0.08, 1.00064, 0.076]),
            ),
            Atom(
                Label="H2",
                AtomicSymbol="H",
                Coordinates=np.array([0.0062, 0.078, 1.088]),
            ),
        ],
        BondOrderMatrix=np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 1],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
            ]
        ),
    )
    assert molObj.NumberOfSubstructures == 1
    assert molObj.NumberOfBonds == 3
    assert molObj.FormalCharge == 1
    assert molObj.Multiplicity == 1


def test_write_molecule():
    water_triplet = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(Label="H1", AtomicSymbol="H", Coordinates=np.array([-1, 0, 0])),
            Atom(
                Label="O1",
                AtomicSymbol="O",
                Coordinates=np.array([0, 0, 0]),
                Multiplicity=2,
                FormalCharge=1,
            ),
            Atom(Label="H2", AtomicSymbol="H", Coordinates=np.array([0, 1, 0])),
        ],
        BondOrderMatrix=np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ),
    )
    wt_mol_str = water_triplet.WriteMolString()
    with open(f"{Path(__file__).parent}/water_triplet.mol", "w") as f:
        f.write(wt_mol_str)
        f.close()

    water_cation = Molecule(
        Identifier="Water",
        AtomsList=[
            Atom(
                Label="H1", AtomicSymbol="H", Coordinates=np.array([-1.09, 0.05, 0.02])
            ),
            Atom(
                Label="O1",
                AtomicSymbol="O",
                Coordinates=np.array([0.00123, 0.0034, 0.0087]),
                FormalCharge=1,
            ),
            Atom(
                Label="H2",
                AtomicSymbol="H",
                Coordinates=np.array([0.08, 1.00064, 0.076]),
            ),
            Atom(
                Label="H2",
                AtomicSymbol="H",
                Coordinates=np.array([0.0062, 0.078, 1.088]),
            ),
        ],
        BondOrderMatrix=np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        ),
    )
    wc_mol_str = water_cation.WriteMolString()
    with open(f"{Path(__file__).parent}/water_cation.mol", "w") as f:
        f.write(wc_mol_str)
        f.close()

    benzene = Molecule(
        Identifier="Benzene",
        AtomsList=[
            Atom(
                Label="C1",
                AtomicSymbol="C",
                Coordinates=np.array([-1.2131, -0.6884, 0.0004]),
            ),
            Atom(
                Label="C2",
                AtomicSymbol="C",
                Coordinates=np.array([-1.2028, 0.7064, 0.0001]),
            ),
            Atom(
                Label="C3",
                AtomicSymbol="C",
                Coordinates=np.array([-0.0103, -1.3948, 0.0002]),
            ),
            Atom(
                Label="C4",
                AtomicSymbol="C",
                Coordinates=np.array([0.0104, 1.3948, 0.0001]),
            ),
            Atom(
                Label="C5",
                AtomicSymbol="C",
                Coordinates=np.array([1.2028, -0.7063, 0.0006]),
            ),
            Atom(
                Label="C6",
                AtomicSymbol="C",
                Coordinates=np.array([1.2131, 0.6884, 0.0004]),
            ),
        ],
        BondOrderMatrix=np.array(
            [
                [0, 1.5, 1.5, 0, 0, 0],
                [1.5, 0, 0, 1.5, 0, 0],
                [1.5, 0, 0, 0, 1.5, 0],
                [0, 1.5, 0, 0, 0, 1.5],
                [0, 0, 1.5, 0, 0, 1.5],
                [0, 0, 0, 1.5, 1.5, 0],
            ]
        ),
    )
    benzene_str = benzene.WriteMolString()
    with open(f"{Path(__file__).parent}/benzene_stripped.mol", "w") as f:
        f.write(benzene_str)
        f.close()


def test_read_molecule():
    with open(f"{Path(__file__).parent}/water_triplet.mol", "r") as f:
        wt_mol_string = f.read()
        f.close()
    water_triplet = Molecule.ReadMolString(wt_mol_string)
    assert water_triplet.FormalCharge == 1
    assert water_triplet.Multiplicity == 2
    assert water_triplet.NumberOfSubstructures == 1
    assert water_triplet.NumberOfBonds == 2
    assert water_triplet.NumberOfAtoms == 3
    assert water_triplet.MolecularMass == 18.02

    with open(f"{Path(__file__).parent}/water_cation.mol", "r") as f:
        wc_mol_string = f.read()
        f.close()
    water_cation = Molecule.ReadMolString(wc_mol_string)
    assert water_cation.FormalCharge == 1
    assert water_cation.Multiplicity == 1
    assert water_cation.NumberOfSubstructures == 2
    assert water_cation.NumberOfBonds == 2
    assert water_cation.NumberOfAtoms == 4
    assert water_cation.MolecularMass == 19.02


def test_add_atoms_and_bonds_to_molecule():
    with open(f"{Path(__file__).parent}/benzene_stripped.mol", "r") as f:
        benzene_string = f.read()
        f.close()
    benzene = Molecule.ReadMolString(benzene_string)
    for idx, atomObj in enumerate(deepcopy(benzene.AtomsList)):
        coor = atomObj.Coordinates
        norm_coor = coor / np.linalg.norm(coor)
        H_coor = norm_coor * 3
        benzene.AddAtom(
            AtomicSymbol="H",
            Coordinates=H_coor,
            Label=None,
        )
        assert benzene.NumberOfSubstructures == 2 + idx
    benzene.AddBond(AtomLabels=["C1", "H1"])
    assert benzene.NumberOfSubstructures == 6
    benzene.AddBond(AtomIndicies=[1, 7])
    benzene.AddBond(
        AtomObjects=[
            benzene.AtomsList[2],
            benzene.AtomsList[8],
        ]
    )
    assert benzene.NumberOfSubstructures == 4
    benzene.AddBond(AtomLabels=["C4", "H4"])
    benzene.AddBond(AtomLabels=["C5", "H5"])
    benzene.AddBond(AtomLabels=["C6", "H6"])
    benzene_str = benzene.WriteMolString()
    with open(f"{Path(__file__).parent}/benzene.mol", "w") as f:
        f.write(benzene_str)
        f.close()
    assert benzene.FormalCharge == 0
    assert benzene.Multiplicity == 1
    assert benzene.NumberOfSubstructures == 1
    assert benzene.NumberOfBonds == 12
    assert benzene.NumberOfAtoms == 12
    assert benzene.MolecularMass == 78.11


def test_add_molecule_to_molecule():
    with open(f"{Path(__file__).parent}/benzene.mol", "r") as f:
        benzene_string = f.read()
        f.close()
    benzene = Molecule.ReadMolString(benzene_string)
    with open(f"{Path(__file__).parent}/water_cation.mol", "r") as f:
        wc_mol_string = f.read()
        f.close()
    water_cation = Molecule.ReadMolString(wc_mol_string)
    water_cation.TranslateMolecule(
        TranslationVector=np.array([0, 0, 2]),
        Displacement=3,
    )
    water_cation.RotateMolecule(
        RotationVector=np.array([0,1,1]),
        RotationAngle=1,
    )
    benzene.AddMolecule(
        MoleculeToAdd=water_cation,
    )
    benzene_string = benzene.WriteMolString()
    with open(f"{Path(__file__).parent}/benzene_water_cation.mol", "w") as f:
        f.write(benzene_string)
        f.close()
    assert benzene.FormalCharge == 1
    assert benzene.Multiplicity == 1
    assert benzene.NumberOfSubstructures == 3
    assert benzene.NumberOfBonds == 14
    assert benzene.NumberOfAtoms == 16
    assert benzene.MolecularMass == 97.14


def test_remove_molecule_from_molecule():
    pass