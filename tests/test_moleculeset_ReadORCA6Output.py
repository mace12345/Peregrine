import numpy as np
from pathlib import Path
from copy import deepcopy

from Peregrine.moleculeset import MoleculeSet
from Peregrine.molecule import Molecule
from Peregrine.atom import Atom


def test_molecule_ReadORCA6Output():
    pass


def test_molecule_ReadORCA6OutputGradients():
    molObj_list = Molecule.ReadORCA6OutputGradients(
        ORCA_output_filepath=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_PBE0-def2-TZVP-Opt-Freq-DEFGRID3/BIHGII-S4.out",
    )
    # Test first atom of first molecule object and molecule energies
    atomObj = molObj_list[0].AtomsList[0]
    assert round(atomObj.Gradient[0], 5) == -0.00087
    assert round(atomObj.Gradient[1], 5) == 0.00208
    assert round(atomObj.Gradient[2], 5) == 0.00307
    assert round(atomObj.Coordinates[0], 5) == -0.63470
    assert round(atomObj.Coordinates[1], 5) == 2.86860
    assert round(atomObj.Coordinates[2], 5) == 2.27080
    assert molObj_list[0].electronic_energy == -8600.751252244099
    assert molObj_list[0].gibbs_free_energy == None
    assert molObj_list[0].entropy == None
    assert molObj_list[0].enthalpy == None
    # Test last atom of last molecule object and molecule energies
    atomObj = molObj_list[-1].AtomsList[-1]
    assert atomObj.Gradient[0] == 0.000012555
    assert atomObj.Gradient[1] == 0.000025513
    assert atomObj.Gradient[2] == -0.000005766
    assert round(atomObj.Coordinates[0], 5) == 4.48027
    assert round(atomObj.Coordinates[1], 5) == 4.10131
    assert round(atomObj.Coordinates[2], 5) == 6.88683
    assert molObj_list[-1].electronic_energy == -8601.6060366043948306
    assert molObj_list[-1].gibbs_free_energy == -8601.12018681
    assert molObj_list[-1].entropy == 0.10870349
    assert molObj_list[-1].enthalpy == -8601.01148332
    # Test to see if gradient information is retained when saved as .MOL file
    for molObj in molObj_list:
        with open(
            f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/{molObj.Identifier}.mol",
            "w",
        ) as f:
            f.write(molObj.WriteMolString())
            f.close()
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/BIHGII-S4_opt0.mol",
        "r",
    ) as f:
        mol_str = f.read()
        f.close()
    molObj_opt0 = Molecule.ReadMolString(mol_str)
    with open(
        f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output/BIHGII-S4_opt24.mol",
        "r",
    ) as f:
        mol_str = f.read()
        f.close()
    molObj_opt24 = Molecule.ReadMolString(mol_str)
    # Test first atom of first molecule object and molecule energies
    atomObj = molObj_opt0.AtomsList[0]
    assert round(atomObj.Gradient[0], 5) == -0.00087
    assert round(atomObj.Gradient[1], 5) == 0.00208
    assert round(atomObj.Gradient[2], 5) == 0.00307
    assert round(atomObj.Coordinates[0], 5) == -0.63470
    assert round(atomObj.Coordinates[1], 5) == 2.86860
    assert round(atomObj.Coordinates[2], 5) == 2.27080
    assert molObj_opt0.electronic_energy == -8600.751252244099
    assert molObj_opt0.gibbs_free_energy == None
    assert molObj_opt0.entropy == None
    assert molObj_opt0.enthalpy == None
    # Test last atom of last molecule object and molecule energies
    atomObj = molObj_opt24.AtomsList[-1]
    assert atomObj.Gradient[0] == 0.000012555
    assert atomObj.Gradient[1] == 0.000025513
    assert atomObj.Gradient[2] == -0.000005766
    assert round(atomObj.Coordinates[0], 5) == 4.48027
    assert round(atomObj.Coordinates[1], 5) == 4.10131
    assert round(atomObj.Coordinates[2], 5) == 6.88683
    assert molObj_opt24.electronic_energy == -8601.6060366043948306
    assert molObj_opt24.gibbs_free_energy == -8601.12018681
    assert molObj_opt24.entropy == 0.10870349
    assert molObj_opt24.enthalpy == -8601.01148332


def test_moleculeset_ReadORCA6OptOutput():
    ms = MoleculeSet()
    ms.ReadORCA6OptOutput(
        input_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_PBE0-def2-TZVP-Opt-Freq-DEFGRID3",
        output_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S2_Gradient-Output",
    )
    ms = MoleculeSet()
    ms.ReadORCA6OptOutput(
        input_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S4_PBE0-def2-TZVP-Opt-Freq-DEFGRID3",
        output_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CoIILig-S4_Gradient-Output",
    )
    ms = MoleculeSet()
    ms.ReadORCA6OptOutput(
        input_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CrIILig-S1_PBE0-def2-TZVP-Opt-Freq-DEFGRID3",
        output_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CrIILig-S1_Gradient-Output",
    )
    ms = MoleculeSet()
    ms.ReadORCA6OptOutput(
        input_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CrIILig-S3_PBE0-def2-TZVP-Opt-Freq-DEFGRID3",
        output_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CrIILig-S3_Gradient-Output",
    )
    ms = MoleculeSet()
    ms.ReadORCA6OptOutput(
        input_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CrIILig-S5_PBE0-def2-TZVP-Opt-Freq-DEFGRID3",
        output_file_path=f"{str(Path(__file__).parent.parent).replace("\\", "/")}/data/testing_data/ReadORCA6Outputs/CrIILig-S5_Gradient-Output",
    )
