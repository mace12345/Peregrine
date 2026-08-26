import os
import re
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import itertools


from .atom import Atom
from .molecule import Molecule

import pandas as pd

import numpy as np

def _GeneralHelper_TooSmallBondAngle(molObj: Molecule, minimum_bond_angle: float) -> bool:
    for atomObj in molObj.AtomsList:
        n_atoms = molObj.GetAtomNeighbours(AtomObject=atomObj)
        for atomObj1, atomObj2 in itertools.combinations(n_atoms, 2):
            theta = np.rad2deg(
                molObj.GetBondAngle(AtomObjects=[atomObj1, atomObj, atomObj2])
            )
            if theta <= minimum_bond_angle:
                return True
    return False

def _GeneralHelper_TooLargeBondLength(molObj: Molecule, maximum_bond_length: float) -> bool:
    for atomObj in molObj.AtomsList:
        for n_atom in molObj.GetAtomNeighbours(AtomObject=atomObj):
            length = np.linalg.norm(atomObj.Coordinates - n_atom.Coordinates)
            if length > maximum_bond_length:
                return True
    return False

class MoleculeSet:
    def __init__(self):
        self.ResultsDF: pd.DataFrame | None = None
        self.MoleculesDict: dict[str, Molecule] = {}

    def AddMolecule(self, MoleculeObject: Molecule | list[Molecule]):
        if type(MoleculeObject) == list:
            for molObj in MoleculeObject:
                self.MoleculesDict[molObj.Identifier] = molObj
        else:
            self.MoleculesDict[MoleculeObject.Identifier] = MoleculeObject

    def RemoveMolecule(self, MoleculeObject: Molecule | list[Molecule]):
        if type(MoleculeObject) == list:
            for molObj in MoleculeObject:
                del self.MoleculesDict[molObj.Identifier]
        else:
            del self.MoleculesDict[molObj.Identifier]

    def RemoveDuplicateMolecules(self):
        if self.ResultsDF is None:
            self.ResultsDF = pd.DataFrame(
                {
                    "Identifier": [molObj.Identifier for molObj in self.MoleculesDict.values()],
                    "Inchi strings": [molObj.WriteInchiString() for molObj in self.MoleculesDict.values()],
                }
            )
        else:
            print("Need to add fast functionality here")
        self.ResultsDF = self.ResultsDF.drop_duplicates(subset="Inchi strings", keep="first").reset_index(drop=True)
        identifiers = set(self.ResultsDF["Identifier"])
        self.MoleculesDict = {
            key: molObj
            for key, molObj in self.MoleculesDict.items()
            if molObj.Identifier in identifiers
        }

    def RemoveNonsensicalGeometries(
        self,
        minimum_bond_angle: float = 50,
        maximum_bond_length: float = 4,
    ):
        identifiers_to_remove = []
        for molObj in self.MoleculesDict.values():
            if _GeneralHelper_TooSmallBondAngle(molObj, minimum_bond_angle):
                identifiers_to_remove.append(molObj.Identifier)
                continue
            if _GeneralHelper_TooLargeBondLength(molObj, maximum_bond_length):
                identifiers_to_remove.append(molObj.Identifier)
                continue
        for identifier in identifiers_to_remove:
            del self.MoleculesDict[identifier]


    # === Read in molecule information (mainly from directories) ===

    def ReadXYZFileDirectory(self):
        pass

    def WriteXYZFileDirectory(self):
        pass

    @classmethod
    def ReadMolFileDirectory(cls, mol_file_directory: str) -> "MoleculeSet":
        mol_file_list = [
            i for i in os.listdir(mol_file_directory) if i.endswith(".mol")
        ]

        def load(mol_file):
            with open(f"{mol_file_directory}/{mol_file}") as f:
                return Molecule.ReadMolString(f.read())

        self = cls()
        with ThreadPoolExecutor(max_workers=int(os.cpu_count() / 2)) as executor:
            for molObj in executor.map(load, mol_file_list):
                self.MoleculesDict[molObj.Identifier] = molObj
        return self

    @classmethod
    def ReadMol2File(cls, mol2_file: str) -> "MoleculeSet":
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

        # Instantiate a new MoleculeSet
        instance = cls()
        for molecule_string in molecule_string_list:
            molObj = Molecule.ReadMol2String(molecule_string)
            if molObj is None:
                continue
            instance.MoleculesDict[molObj.Identifier] = molObj
        return instance

    @classmethod
    def ReadSMILESList(
        cls, SMILES_list: list, Identifier_List: list, AddHydrogens: bool = True
    ) -> "MoleculeSet":
        instance = cls()
        for SMILES, Identifier in zip(SMILES_list, Identifier_List):
            molObj = Molecule.ReadSMILESString(
                SMILES, Identifier, AddHydrogens=AddHydrogens
            )
            instance.MoleculesDict[molObj.Identifier] = molObj
        return instance

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

    def WriteMolFileDirectory(self, mol_file_directory: str):
        os.makedirs(mol_file_directory, exist_ok=True)

        def write(item):
            identifier, molObj = item
            with open(f"{mol_file_directory}/{identifier}.mol", "w") as f:
                f.write(molObj.WriteMolString())

        workers = max(1, (os.cpu_count() or 2) // 2)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # list() forces iteration, so exceptions propagate instead of being swallowed
            list(executor.map(write, self.MoleculesDict.items()))

    def WritePySCFInput(
        self,
        pyscf_file_directory: str,
        method: str = "hf",
        basisset: dict | str = "def2-svp",
        ecp: list[str] | None = None,
        restricted: bool = True,
        calculation_type: str = "single point",
        get_gradients: bool = True,
        get_fock_matrix: bool = True,
        max_memory: int = 1000,  # in MB
        grid_density: int = 5,
        prune_grids: None | bool = True,
        optimisation_convergence_settings: dict | None = None,
        job_scheduler_used: None | str = "SLURM",
        CPU_count: int = 4,
        max_time: int = 2880,
    ):
        os.makedirs(pyscf_file_directory, exist_ok=True)
        submit_jobs = ""
        for molObj in self.MoleculesDict.values():
            pyscf_str = molObj.WritePySCFInput(
                method=method,
                basisset=basisset,
                ecp=ecp,
                restricted=restricted,
                calculation_type=calculation_type,
                get_gradients=get_gradients,
                get_fock_matrix=get_fock_matrix,
                max_memory=max_memory,  # in MB
                grid_density=grid_density,
                prune_grids=prune_grids,
                optimisation_convergence_settings=optimisation_convergence_settings,
                CPU_count=CPU_count,
            )
            with open(
                pyscf_file_directory / f"{molObj.Identifier}.py", "w"
            ) as f:
                f.write(pyscf_str)
                f.close()
            if job_scheduler_used == "SLURM":
                slurm_str = molObj.WriteSLURMStringForPySCF(
                    job_name=molObj.Identifier,
                    CPU_count=CPU_count,
                    max_memory=max_memory,
                    max_time=max_time,
                )
                with open(
                    pyscf_file_directory / f"{molObj.Identifier}.sh", "w"
                ) as f:
                    f.write(slurm_str)
                    f.close()
                submit_jobs += f"sbatch {molObj.Identifier}.sh\n"
        if submit_jobs != "":
            with open(
                pyscf_file_directory / f"submit_jobs.sh", "w"
            ) as f:
                f.write(submit_jobs)
                f.close()

    def WriteORCAInput(
        self,
        orca_file_directory: str,
        method: str = "hf",
        basisset: str = "def2-svp",
        ORCA_commands: str = "opt freq",
        CPU_count: int = 4,
        max_memory: int = 1000,  # MB
        max_time: None | int = 2880,  # minuets
        job_scheduler_used: None | str = "SLURM",
        MPI_used: None | str = "OpenMPI",
        file_types_to_save: list[str] = [".out", ".xyz"],
    ):
        if job_scheduler_used is not None:
            job_scheduler_used = job_scheduler_used.lower()
        os.makedirs(orca_file_directory, exist_ok=True)
        submit_jobs = ""
        if self.ResultsDF is not None:
            molObj_list = [
                self.MoleculesDict[identifier]
                for identifier in self.ResultsDF[
                    self.ResultsDF["Error Code"].isna() == False
                ]["Identifier"]
            ]
        else:
            molObj_list = self.MoleculesDict.values()
        # Write jobs
        for molObj in molObj_list:
            orca_inp, queue_sh = molObj.WriteORCAInput(
                method=method,
                basisset=basisset,
                ORCA_commands=ORCA_commands,
                CPU_count=CPU_count,
                max_memory=max_memory,
                max_time=max_time,
                job_scheduler_used=job_scheduler_used,
                MPI_used=MPI_used,
                file_types_to_save=file_types_to_save,
            )
            with open(orca_file_directory / f"{molObj.Identifier}.inp", "w") as f:
                f.write(orca_inp)
                f.close()
            with open(orca_file_directory / f"{molObj.Identifier}.sh", "w") as f:
                f.write(queue_sh)
                f.close()
            if job_scheduler_used == "slurm":
                submit_jobs += f"sbatch {molObj.Identifier}.sh\n"
        with open(orca_file_directory / "submit_jobs.sh", "w") as f:
            f.write(submit_jobs)
            f.close()

    def WritePsi4Input(
        self,
        psi4_file_directory: str,
        method: str = "wb97m-d3bj",
        basisset: str = "def2-tzvppd",
        local_basissets: dict | None = None,
        ecp: dict | None = None,
        optimise_geometry: bool = False,
        get_frequency: bool = False,
        CPU_count: int = 4,
        max_memory: int = 1000,
        max_time: None | int = 2880,
        job_scheduler_used: None | str = "slurm",
        file_types_to_save: list[str] = [".out"],
        remove_negative_frequencies: bool = True,
    ):
        os.makedirs(psi4_file_directory, exist_ok=True)
        if job_scheduler_used is not None:
            job_scheduler_used = job_scheduler_used.lower()
        submit_jobs = ""

        # Resubmit previous calculations only if unsuccessful
        if self.ResultsDF is not None:
            molObj_list = [
                self.MoleculesDict[identifier]
                for identifier in self.ResultsDF[
                    self.ResultsDF["Error Code"].isna() == False
                ]["Identifier"]
            ]
            molObj_list += [
                self.MoleculesDict[identifier]
                for identifier in self.ResultsDF[
                    self.ResultsDF["Vibrational Frequency 6 (cm-1)"] < 0
                ]["Identifier"]
            ]
        else:
            molObj_list = self.MoleculesDict.values()

        # Sort local basissets
        new_local_basissets = None
        if local_basissets is not None:
            new_local_basissets = {}
            for atomic_symbols in local_basissets:
                local_basisset = local_basissets[atomic_symbols]
                atomic_symbols = [i for i in atomic_symbols.replace(" ", "").split(",") if i != ""]
                for atomic_symbol in atomic_symbols:
                    new_local_basissets[atomic_symbol] = local_basisset

        for molObj in molObj_list:

            # Sort local basissets
            if new_local_basissets is not None:
                local_basissets = {
                    atomic_symbol: new_local_basissets[atomic_symbol] 
                    for atomic_symbol in molObj.GetAtomicSymbols()
                }
                if len(local_basissets) == 0:
                    local_basissets = None
                
            orca_inp, queue_sh = molObj.WritePsi4Input(
                method=method,
                basisset=basisset,
                local_basisset=local_basissets,
                ecp=ecp,
                optimise_geometry=optimise_geometry,
                get_frequency=get_frequency,
                CPU_count=CPU_count,
                max_memory=max_memory,
                max_time=max_time,
                job_scheduler_used=job_scheduler_used,
                file_types_to_save=file_types_to_save,
            )
            with open(psi4_file_directory / f"{molObj.Identifier}.py", "w") as f:
                f.write(orca_inp)
                f.close()
            if job_scheduler_used == "slurm":
                submit_jobs += f"sbatch {molObj.Identifier}.sh\n"
                with open(psi4_file_directory / f"{molObj.Identifier}.sh", "w") as f:
                    f.write(queue_sh)
                    f.close()
            elif job_scheduler_used is None:
                submit_jobs += f"python {molObj.Identifier}.py\n"
        with open(psi4_file_directory / "submit_jobs.sh", "w") as f:
            f.write(submit_jobs)
            f.close()

    # === Read comp chem output calculations ===

    @classmethod
    def ReadORCAOutput(
        cls,
        orca_file_directory: str,
        output_mol_file_directory: str,
        template_moleculeset: "MoleculeSet | None" = None,
    ) -> "MoleculeSet":
        dir_list = os.listdir(orca_file_directory)
        # Remove unnessicary files
        # Track files to .out files to read
        files_to_remove = []
        out_files = []
        remove_patterns = [r"slurm\-", r"atom(\d+)\.out", r"\.sh"]
        out_pattern = r"\.out"
        for file in dir_list:
            # Look for files to remove
            for pattern in remove_patterns:
                if re.search(pattern, file) is not None:
                    files_to_remove.append(file)
            # Look for .out files to keep
            if (
                re.search(remove_patterns[1], file) is None
                and re.search(out_pattern, file) is not None
            ):
                out_files.append(file)
        for file in files_to_remove:
            os.remove(orca_file_directory / file)

        # Read ORCA output files
        instance = MoleculeSet()
        for out_file in sorted(out_files):
            Identifier = str(out_file).split(".")[0]
            if template_moleculeset is None:
                template_molObj = None
            else:
                template_molObj = template_moleculeset.MoleculesDict[Identifier]
            molObj = Molecule.ReadORCA6Output(
                orca_file_directory / out_file,
                template_molObj=template_molObj,
            )
            instance.MoleculesDict[molObj.Identifier] = molObj
        # Construct Results DataFrame
        instance.ResultsDF = pd.DataFrame(
            {
                "Identifier": [identifier for identifier in instance.MoleculesDict],
                "Method": [
                    molObj.calculation_method
                    for molObj in instance.MoleculesDict.values()
                ],
                "Dispersion": [
                    molObj.dispersion for molObj in instance.MoleculesDict.values()
                ],
                "Basis set":
                [
                    molObj.basisset for molObj in instance.MoleculesDict.values()
                ],
                "Number of primitive basis functions": [
                    molObj.num_prim_basis_functions
                    for molObj in instance.MoleculesDict.values()
                ],
                "RAM used per CPU core (MB)": [
                    molObj.RAM_used for molObj in instance.MoleculesDict.values()
                ],
                "Number of CPU cores used": [
                    molObj.num_CPU_used for molObj in instance.MoleculesDict.values()
                ],
                "Charge": [
                    molObj.FormalCharge for molObj in instance.MoleculesDict.values()
                ],
                "Multiplicity": [
                    molObj.Multiplicity for molObj in instance.MoleculesDict.values()
                ],
                "Error Code": [
                    molObj.error_code for molObj in instance.MoleculesDict.values()
                ],
                "wallclock time taken (seconds)": [
                    molObj.wallclock_time_sec
                    for molObj in instance.MoleculesDict.values()
                ],
                "Electronic Energy (Eh)": [
                    molObj.electronic_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Gibbs Free Energy (Eh)": [
                    molObj.gibbs_free_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Enthalpy (Eh)": [
                    molObj.enthalpy for molObj in instance.MoleculesDict.values()
                ],
                "Entropy (Eh)": [
                    molObj.entropy for molObj in instance.MoleculesDict.values()
                ],
                "Spin Contaimination (<S**2>)": [
                    molObj.spin_contamination
                    for molObj in instance.MoleculesDict.values()
                ],
                "Vibrational Frequency 6 (cm-1)": [
                    (
                        molObj.vibrational_frequencies[5][1]
                        if molObj.vibrational_frequencies is not None
                        else None
                    )
                    for molObj in instance.MoleculesDict.values()
                ],
            }
        )
        instance.ResultsDF.to_csv(str(output_mol_file_directory) + ".csv")
        # Save molObj files as V3000 .mol files
        instance.WriteMolFileDirectory(output_mol_file_directory)
        return instance

    @classmethod
    def ReadPsi4Output(
        cls,
        psi4_file_directory: str,
        output_mol_file_directory: str,
        template_moleculeset: "MoleculeSet | None" = None,
    ) -> "MoleculeSet":
        dir_list = os.listdir(psi4_file_directory)
        out_list = [i for i in dir_list if i.split(".")[-1] == "out"]
        json_list = [i for i in dir_list if i.split(".")[-1] == "json"]
        id_list = [i.split(".")[0] for i in dir_list if i.split(".")[-1] == "py"]
        id_dict = {}
        for identifier in id_list:
            if f"{identifier}.out" in out_list and f"{identifier}.meta.json" in json_list:
                id_dict[identifier] = [f"{identifier}.out", f"{identifier}.meta.json"]
            elif f"{identifier}.out" in out_list and f"{identifier}.meta.json" not in json_list:
                id_dict[identifier] = [f"{identifier}.out", None]
        # Read Psi4 .out and .meta.json files
        instance = MoleculeSet()
        for identifier in id_dict:
            out_file_name = id_dict[identifier][0]
            json_file_name = id_dict[identifier][1]
            if template_moleculeset is None:
                template_molObj = None
            else:
                template_molObj = template_moleculeset.MoleculesDict[identifier]
            molObj = Molecule.ReadPsi4Output(
                psi4_file_directory,
                out_file_name=out_file_name,
                json_file_name=json_file_name,
                template_molObj=template_molObj,
            )
            instance.MoleculesDict[molObj.Identifier] = molObj
        # Construct Results DataFrame
        instance.ResultsDF = pd.DataFrame(
            {
                "Identifier": [identifier for identifier in instance.MoleculesDict],
                "Method": [
                    molObj.calculation_method
                    for molObj in instance.MoleculesDict.values()
                ],
                "Dispersion": [
                    molObj.dispersion for molObj in instance.MoleculesDict.values()
                ],
                "Basis set":
                [
                    molObj.basisset for molObj in instance.MoleculesDict.values()
                ],
                "Number of primitive basis functions": [
                    molObj.num_prim_basis_functions
                    for molObj in instance.MoleculesDict.values()
                ],
                "RAM used (MB)": [
                    molObj.RAM_used for molObj in instance.MoleculesDict.values()
                ],
                "Number of CPU cores used": [
                    molObj.num_CPU_used for molObj in instance.MoleculesDict.values()
                ],
                "Charge": [
                    molObj.FormalCharge for molObj in instance.MoleculesDict.values()
                ],
                "Multiplicity": [
                    molObj.Multiplicity for molObj in instance.MoleculesDict.values()
                ],
                "Error Code": [
                    molObj.error_code for molObj in instance.MoleculesDict.values()
                ],
                "wallclock time taken (seconds)": [
                    molObj.wallclock_time_sec
                    for molObj in instance.MoleculesDict.values()
                ],
                "Electronic Energy (Eh)": [
                    molObj.electronic_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Gibbs Free Energy (Eh)": [
                    molObj.gibbs_free_energy
                    for molObj in instance.MoleculesDict.values()
                ],
                "Enthalpy (Eh)": [
                    molObj.enthalpy for molObj in instance.MoleculesDict.values()
                ],
                "Entropy (Eh)": [
                    molObj.entropy for molObj in instance.MoleculesDict.values()
                ],
                "Spin Contaimination (<S**2>)": [
                    molObj.spin_contamination
                    for molObj in instance.MoleculesDict.values()
                ],
                "Vibrational Frequency 6 (cm-1)": [
                    (
                        molObj.vibrational_frequencies[0]
                        if molObj.vibrational_frequencies is not None
                        else None
                    )
                    for molObj in instance.MoleculesDict.values()
                ],
            }
        )
        instance.ResultsDF.to_csv(str(output_mol_file_directory) + ".csv")
        # Save molObj files as V3000 .mol files
        instance.WriteMolFileDirectory(output_mol_file_directory)
        return instance

    
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
