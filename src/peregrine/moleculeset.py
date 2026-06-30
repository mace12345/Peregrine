import numpy as np
import os
import time
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from .atom import Atom
from .molecule import Molecule


class MoleculeSet:
    def __init__(self):
        self.MoleculesDict: dict[str, Molecule] = {}

    # === Read in molecule information (mainly from directories) ===

    def ReadXYZFileDirectory(self):
        pass

    def WriteXYZFileDirectory(self):
        pass

    def ReadMolFileDirectory(self, mol_file_directory: str):
        mol_file_list = [
            i for i in os.listdir(mol_file_directory) if i.endswith(".mol")
        ]

        def load(mol_file):
            with open(f"{mol_file_directory}/{mol_file}") as f:
                return Molecule.ReadMolString(f.read())

        with ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
            for molObj in executor.map(load, mol_file_list):
                self.MoleculesDict[molObj.Identifier] = molObj

    def ReadMol2File(self, mol2_file: str):
        """
        Reads all `.mol2` files in a given directory and parses them into Molecule objects.

        The method:
        - Scans the input directory for all `.mol2` files.
        - Reads each file and parses molecular data.
        - Stores all molecules in MoleculesDict indexed by Identifier.
        - Provides feedback on the number of molecules loaded.

        Args:
            mol2_directory (str): Path to the directory containing `.mol2` files.

        Raises:
            FileNotFoundError: If the directory does not exist.
            IOError: If files cannot be read.

        Example:
            mol_set = MoleculeSet()
            mol_set.ReadMol2Files("./mol2_files")
            print(mol_set.MoleculesDict.keys())
        """
        with open(mol2_file, "r") as f:
            file_content = f.read()
        # Split multiple molecules in file if present
        molecule_string_list = [
            i
            for i in file_content.split("@<TRIPOS>MOLECULE\n")
            if i != "" and "@<TRIPOS>ATOM" in i
        ]
        molecule_list = []
        for molecule_string in molecule_string_list:
            mol_obj = Molecule.ReadMol2String(molecule_string)
            molecule_list.append(mol_obj)
        # Store molecules in dictionary
        self.MoleculesDict = {mol.Identifier: mol for mol in molecule_list}

    def ReadORCA6OutputDirectory(
        self,
        input_file_path: str,
        output_file_path: str,
    ):
        # TODO: test this code, it was written by Claud Haiku 4.5
        """
        Reads ORCA 6.0 optimization output files from a directory and converts them to Molecule objects.

        The method:
        - Scans the input directory for `.out` files.
        - Parses each ORCA output file into Molecule objects.
        - Stores all molecules in MoleculesDict indexed by Identifier.
        - Writes each molecule to a separate `.mol` file in the output directory.

        Args:
            input_file_path (str): Path to the directory containing ORCA `.out` files.
            output_file_path (str): Path to the directory where `.mol` files will be written.

        Raises:
            FileNotFoundError: If the input directory does not exist.
            IOError: If output files cannot be written.

        Example:
            mol_set = MoleculeSet()
            mol_set.ReadORCA6OptOutput(
                input_file_path="./orca_outputs",
                output_file_path="./mol_files"
            )
        """
        os.makedirs(output_file_path, exist_ok=True)

        # Collect all .out files from input directory
        orca_file_list = [
            f"{input_file_path}/{filename}"
            for filename in os.listdir(input_file_path)
            if filename.split(".")[-1] == "out"
        ]

        # Parse all ORCA files and collect molecules
        molecule_list = []
        for orca_file in orca_file_list:
            molecules_from_file = Molecule.ReadORCA6Output(
                ORCA_output_filepath=orca_file
            )
            molecule_list.extend(molecules_from_file)

        # Store molecules in dictionary and write to .mol files
        self.MoleculesDict = {mol.Identifier: mol for mol in molecule_list}

        for identifier, mol_obj in self.MoleculesDict.items():
            output_file = f"{output_file_path}/{identifier}.mol"
            with open(output_file, "w") as f:
                f.write(mol_obj.WriteMolString())

    def ReadORCA6OptimisationOutput(
        self,
        input_file_path: str,
        output_file_path: str,
    ):
        """
        Uses Molecule.ReadORCA6OutputGradients()
        The default of this function is that it does not fully derive molecule attributes
        """
        os.makedirs(output_file_path, exist_ok=True)
        dir_list = [
            f"{input_file_path}/{i}"
            for i in os.listdir(input_file_path)
            if i.split(".")[-1] == "out"
        ]
        NewMoleculesDict = {
            mol.Identifier: mol
            for sublist in [
                Molecule.ReadORCA6OutputGradients(
                    ORCA_output_filepath=out_file,
                    template_molObj=(
                        self.MoleculesDict[out_file.split("/")[-1].split(".")[0]]
                        if out_file.split("/")[-1].split(".")[0] in self.MoleculesDict
                        else None
                    ),
                )
                for out_file in dir_list
            ]
            for mol in sublist
        }
        for Identifier in NewMoleculesDict:
            with open(f"{output_file_path}/{Identifier}.mol", "w") as f:
                f.write(NewMoleculesDict[Identifier].WriteMolString())
                f.close()

    # === Write molecule information into directories ===

    def WriteORCA6Input(self):
        pass

    def WriteMolFileDirectory(self, mol_file_directory: str):
        os.makedirs(mol_file_directory, exist_ok=True)
        for Identifier in self.MoleculesDict:
            molObj = self.MoleculesDict[Identifier]
            with open(f"{mol_file_directory}/{Identifier}.mol", "w") as f:
                f.write(molObj.WriteMolString())
                f.close()

    # === Execute a workflow of some kind ===

    def CalculateAtomicSOAPDescriptors(
        self,
        output_mol_file_directory: str | None = None,
        output_csv_file_directory: str | None = None,
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
            periodic -- Is the ASE Atoms object structure solid state periodic or not (default = False)
        """
        if output_mol_file_directory is not None:
            os.makedirs(output_mol_file_directory, exist_ok=True)

            def process(item):
                identifier, molObj = item
                molObj_copy = deepcopy(molObj)
                molObj_copy.GetSOAPDescriptors(
                    RadiusCutOff=RadiusCutOff,
                    NumRadialBasisFunctions=NumRadialBasisFunctions,
                    MaxDegreeSphericalHarm=MaxDegreeSphericalHarm,
                    periodic=periodic,
                )
                with open(f"{output_mol_file_directory}/{identifier}.mol", "w") as f:
                    f.write(molObj_copy.WriteMolString())
                del molObj_copy

            with ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
                list(executor.map(process, self.MoleculesDict.items()))
        elif output_csv_file_directory is not None:
            print("Write code to create CSV file")
            pass
