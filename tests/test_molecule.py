import numpy as np
from pathlib import Path
from copy import deepcopy

from Peregrine.molecule import Molecule
from Peregrine.atom import Atom

xtb_binary_path = "C:/Users/samue/Desktop/xtb-bleed-windows/bin"


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
    benzene.AddBond(AtomIndices=[1, 7])
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
        RotationVector=np.array([0, 1, 1]),
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


def test_SplitMoleculeIntoComponents():
    # Benzene, Water, and Proton
    with open(f"{Path(__file__).parent}/benzene_water_cation.mol", "r") as f:
        benzene_str = f.read()
        f.close()
    benzene_water_cat = Molecule.ReadMolString(benzene_str)
    benzene_water_cat.AtomsDict["O1"][1].FormalCharge = 0
    benzene_water_cat.AtomsDict["H9"][1].FormalCharge = 1
    components = benzene_water_cat.SplitMoleculeIntoComponents()
    assert components[0].MolecularMass == 78.11
    assert components[0].NumberOfAtoms == 12
    assert components[0].NumberOfBonds == 12
    assert components[0].NumberOfSubstructures == 1
    assert components[0].FormalCharge == 0
    assert components[0].Multiplicity == 1
    assert components[0].WriteSMILESString() == "[H]C1=C([H])C([H])=C([H])C([H])=C1[H]"
    assert components[1].MolecularMass == 18.02
    assert components[1].NumberOfAtoms == 3
    assert components[1].NumberOfBonds == 2
    assert components[1].NumberOfSubstructures == 1
    assert components[1].FormalCharge == 0
    assert components[1].Multiplicity == 1
    assert components[1].WriteSMILESString() == "[H]O[H]"
    assert components[2].MolecularMass == 1.01
    assert components[2].NumberOfAtoms == 1
    assert components[2].NumberOfBonds == 0
    assert components[2].NumberOfSubstructures == 1
    assert components[2].FormalCharge == 1
    assert components[2].Multiplicity == 1
    assert components[2].WriteSMILESString() == "[H+]"


def test_DeriveMoleculeSmiles():
    # Benzene, Water, and Proton
    with open(f"{Path(__file__).parent}/benzene_water_cation.mol", "r") as f:
        benzene_str = f.read()
        f.close()
    benzene_water_cat = Molecule.ReadMolString(benzene_str)
    benzene_water_cat.AtomsDict["O1"][1].FormalCharge = 0
    benzene_water_cat.AtomsDict["H9"][1].FormalCharge = 1
    benzene_water_cat.DeriveMoleculeSMILES()
    assert (
        benzene_water_cat.AssociatedMoleculeSMILES
        == "[H+].[H]C1=C([H])C([H])=C([H])C([H])=C1[H].[H]O[H]"
    )


def test_ChangeAtom():
    with open(f"{Path(__file__).parent}/benzene_water_cation.mol", "r") as f:
        benzene_str = f.read()
        f.close()
    benzene_water_cat = Molecule.ReadMolString(benzene_str)
    components = benzene_water_cat.SplitMoleculeIntoComponents()
    benzene = components[0]
    benzene.ChangeAtom(
        AtomLabel="C1",
        NewAtomicSymbol="N",
    )
    benzene.ChangeAtom(
        AtomIndex=2,
        NewAtomicSymbol="N",
    )
    benzene.TranslateMolecule(TranslationVector=np.array([0, 0, 1]), Displacement=6)
    benzene_water_cat.AddMolecule(benzene)
    benzene_water_cat.AtomsDict["O1"][1].FormalCharge = 0
    benzene_water_cat.AtomsDict["H9"][1].FormalCharge = 1
    benzene_water_cat_str = benzene_water_cat.WriteMolString()
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "w") as f:
        f.write(benzene_water_cat_str)
        f.close()
    assert benzene_water_cat.MolecularMass == 179.24
    assert benzene_water_cat.NumberOfAtoms == 28
    assert benzene_water_cat.NumberOfBonds == 26
    assert benzene_water_cat.NumberOfSubstructures == 4
    assert benzene_water_cat.FormalCharge == 1
    assert benzene_water_cat.Multiplicity == 1
    assert benzene.MolecularMass == 82.11


def test_RemoveAtom():
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        aromatic_sandwich_str = f.read()
        f.close()
    aromatic_sandwich = Molecule.ReadMolString(aromatic_sandwich_str)
    aromatic_sandwich.RemoveAtom(AtomLabel="C7")
    aromatic_sandwich.RemoveAtom(AtomIndex=21)
    aromatic_sandwich.RemoveAtom(AtomObject=aromatic_sandwich.AtomsDict["H10"][1])
    aromatic_sandwich_str = aromatic_sandwich.WriteMolString()
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "w") as f:
        f.write(aromatic_sandwich_str)
        f.close()
    assert aromatic_sandwich.MolecularMass == 165.22
    assert aromatic_sandwich.NumberOfAtoms == 25
    assert aromatic_sandwich.NumberOfBonds == 22
    assert aromatic_sandwich.NumberOfSubstructures == 4
    assert aromatic_sandwich.FormalCharge == 1
    assert aromatic_sandwich.Multiplicity == 1


def test_RemoveBond():
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        aromatic_sandwich_str = f.read()
        f.close()
    aromatic_sandwich = Molecule.ReadMolString(aromatic_sandwich_str)
    aromatic_sandwich.RemoveBond(AtomLabels=["C7", "C9"])
    aromatic_sandwich.RemoveBond(AtomIndices=[20, 19])
    assert aromatic_sandwich.NumberOfBonds == 20
    assert aromatic_sandwich.NumberOfSubstructures == 6
    aromatic_sandwich.AddBond(AtomLabels=["C7", "C9"])
    aromatic_sandwich.AddBond(AtomIndices=[20, 19], BondOrder=2)
    aromatic_sandwich.AddBond(AtomIndices=[16, 18])
    assert aromatic_sandwich.NumberOfBonds == 23
    assert aromatic_sandwich.NumberOfSubstructures == 4
    aromatic_sandwich_str = aromatic_sandwich.WriteMolString()
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "w") as f:
        f.write(aromatic_sandwich_str)
        f.close()


def test_ChangeBond():
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        aromatic_sandwich_str = f.read()
        f.close()
    aromatic_sandwich = Molecule.ReadMolString(aromatic_sandwich_str)
    aromatic_sandwich.ChangeBond(NewBondOrder=1, AtomIndices=[16, 17])
    aromatic_sandwich.ChangeBond(NewBondOrder=1, AtomIndices=[19, 17])
    aromatic_sandwich.ChangeBond(NewBondOrder=2, AtomIndices=[16, 18])
    aromatic_sandwich_str = aromatic_sandwich.WriteMolString()
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "w") as f:
        f.write(aromatic_sandwich_str)
        f.close()


def test_RemoveMolecule():
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        aromatic_sandwich_str = f.read()
        f.close()
    aromatic_sandwich = Molecule.ReadMolString(aromatic_sandwich_str)
    aromatic_sandwich.RemoveMolecule(SMILES="O")
    aromatic_sandwich.RemoveMolecule(SMARTS="[#6][#6][#6][#6]")
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        aromatic_sandwich_str = f.read()
        f.close()
    aromatic_sandwich_2 = Molecule.ReadMolString(aromatic_sandwich_str)
    aromatic_sandwich_2.TranslateMolecule(
        TranslationVector=np.array([0, 0, -1]), Displacement=4
    )
    aromatic_sandwich.AddMolecule(MoleculeToAdd=aromatic_sandwich_2)
    assert aromatic_sandwich.MolecularMass == 234.3
    assert aromatic_sandwich.NumberOfAtoms == 35
    assert aromatic_sandwich.NumberOfBonds == 32
    assert aromatic_sandwich.NumberOfSubstructures == 6
    assert aromatic_sandwich.FormalCharge == 2
    assert aromatic_sandwich.Multiplicity == 1
    aromatic_sandwich.RemoveMolecule(SubstructureIndex=4)
    assert aromatic_sandwich.NumberOfSubstructures == 5
    aromatic_sandwich_str = aromatic_sandwich.WriteMolString()
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "w") as f:
        f.write(aromatic_sandwich_str)
        f.close()


def test_OptimiseGeometry():
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        aromatic_sandwich_str = f.read()
        f.close()
    aromatic_sandwich = Molecule.ReadMolString(aromatic_sandwich_str)
    aromatic_sandwich.OptimiseGeometry(
        SimpleLennardJonesPotential=True,
        SimpleLennardJonesPotential_settings={"Max Steps": 150},
        MolecularMechanics=True,
        SemiEmpiricalxTB=True,
        SemiEmpiricalxTB_settings={
            "Solvent Model": "gbe",
            "Solvent": "ethanol",
            "Optimisation Cycles": 2,
        },
        xtb_binary_path=xtb_binary_path,
    )
    aromatic_sandwich_str = aromatic_sandwich.WriteMolString()
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "w") as f:
        f.write(aromatic_sandwich_str)
        f.close()


def test_ReadXYZFile():
    # Radical Rearrangment
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/01_3R.xyz",
        identifier="RadPentene",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsList[0].SMARTSCentre = True
    molObj.AtomsList[1].SMARTSCentre = True
    molObj.AtomsList[2].SMARTSCentre = True
    molObj.AtomsList[3].SMARTSCentre = True
    molObj.AtomsList[4].SMARTSCentre = True
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalRearrangment_Reac.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/01_3TS.xyz",
        identifier="RadPentene",
        charge=0,
        multiplicity=2,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalRearrangment_TS.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/01_3P.xyz",
        identifier="RadPentene",
        charge=0,
        multiplicity=2,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalRearrangment_Prod.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Radical Addition
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/01_9P.xyz",
        identifier="RadThiol",
        charge=0,
        multiplicity=2,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalAddition_Prod.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/01_9TS.xyz",
        identifier="RadThiol",
        charge=0,
        multiplicity=2,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalAddition_TS.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalAddition_Reac.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Pericyclic [4+2]
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/02_11P.xyz",
        identifier="Peri42",
        charge=0,
        multiplicity=1,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/Pericyclic42_Prod.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/02_11R.xyz",
        identifier="Peri42",
        charge=0,
        multiplicity=1,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/Pericyclic42_Reac.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/02_11TS.xyz",
        identifier="Peri42",
        charge=0,
        multiplicity=1,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/Pericyclic42_TS.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Pericyclic [3+2]
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/02_56P.xyz",
        identifier="Peri32",
        charge=0,
        multiplicity=1,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/Pericyclic32_Prod.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/02_56TS.xyz",
        identifier="Peri32",
        charge=0,
        multiplicity=1,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/Pericyclic32_TS.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.OptimiseGeometry(
        MolecularMechanics=True,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/Pericyclic32_Reac.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()

    # Halogen Abstraction
    molObj = Molecule.ReadXYZFile(
        xyz_file=f"{Path(__file__).parent}/TS/BH9/03_7TS.xyz",
        identifier="HalAbstract",
        charge=0,
        multiplicity=2,
    )
    molObj.AtomsDict["C1"][1].FormalCharge = 0
    molObj.AtomsDict["F1"][1].FormalCharge = 0
    molObj.AtomsDict["F1"][1].Multiplicity = 1
    molObj.AtomsDict["S1"][1].FormalCharge = 2
    with open(f"{Path(__file__).parent}/TS/BH9/HalAbstract_TS.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()
    molObj.AddBond(AtomIndices=[1, 0])
    molObj.RemoveBond(AtomIndices=[13, 1])
    molObj.OptimiseGeometry(
        SimpleLennardJonesPotential=True,
        MolecularMechanics=True,
    )
    with open(f"{Path(__file__).parent}/TS/BH9/HalAbstract_Reac.mol", "w") as f:
        f.write(molObj.WriteMolString())
        f.close()


def test_AtomIsAromatic():
    with open(f"{Path(__file__).parent}/AromaticSandwich.mol", "r") as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    molObj.RemoveAtom(AtomIndex=22)
    molObj.RemoveAtom(AtomIndex=0)
    molObj.GetAromaticAtoms()
    assert molObj.AtomsList[0].IsAromatic == True
    assert molObj.AtomsList[7].IsAromatic == False


def test_WriteSMARTSString():
    # Radical Rearrangement Reaction
    with open(f"{Path(__file__).parent}/TS/BH9/RadicalRearrangment_Reac.mol", "r") as f:
        molObj_str = f.read()
        f.close()
    molObj = Molecule.ReadMolString(molObj_str)
    SMARTS_1 = molObj.WriteSMARTSString()
