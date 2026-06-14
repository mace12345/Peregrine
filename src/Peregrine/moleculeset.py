import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from .atom import Atom
from .molecule import Molecule


def _process_orca_file(out_filepath: str, output_file_path: str) -> list[str]:
    """Parse one ORCA .out file and write a .mol per molecule. Returns the IDs written."""
    mols = Molecule.ReadORCA6OutputGradients(ORCA_output_filepath=out_filepath)
    written = []
    for mol in mols:
        with open(f"{output_file_path}/{mol.Identifier}.mol", "w") as f:
            f.write(mol.WriteMolString())
        written.append(mol.Identifier)
    return written


class MoleculeSet:
    def __init__(self):
        self.MoleculeDict: dict[Molecule]

    def ReadXYZFile(self):
        pass

    def WriteXYZFile(self):
        pass

    def ReadMolFile(self):
        pass

    def WriteMolFile(self):
        main_mol_str = ""
        for Identifier in self.MoleculeDict:
            mol_str = ""
            molObj = self.MoleculeDict[Identifier]
            # Opening Identifier Line, Header block, and blank comment line
            mol_str += f"{Identifier}\nPeregrine Generated .MOL File\n\n"
            # CTAB begin block, counts line, and begin atoms line
            mol_str += f"M V30 BEGIN CTAB\nM V30 COUNTS {molObj.NumberOfAtoms} {molObj.NumberOfBonds} {molObj.NumberOfSubstructures} 0 0\nM V30 BEGIN ATOM\n"

        pass

    def ReadMol2File(self):
        pass

    def WriteMol2File(self):
        pass

    def ReadORCA6Output(self):
        # TODO: Iterate through directory looking for .out files

        # TODO: Identifier is the name of the .out file, check to see if identifier exist in MoleculeDict
        pass

    def ReadORCA6OptOutput(
        self,
        input_file_path: str,
        output_file_path: str,
    ):
        os.makedirs(output_file_path, exist_ok=True)
        dir_list = [
            f"{input_file_path}/{i}"
            for i in os.listdir(input_file_path)
            if i.split(".")[-1] == "out"
        ]

        molObjs_dict = {
            mol.Identifier: mol
            for sublist in [
                Molecule.ReadORCA6OutputGradients(ORCA_output_filepath=out_file)
                for out_file in dir_list
            ]
            for mol in sublist
        }

        for Identifier in molObjs_dict:
            with open(f"{output_file_path}/{Identifier}.mol", "w") as f:
                f.write(molObjs_dict[Identifier].WriteMolString())
                f.close()

        """worker = partial(_process_orca_file, output_file_path=output_file_path)

        with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
            written_ids = list(executor.map(worker, dir_list, chunksize=4))

        all_ids = [i for sublist in written_ids for i in sublist]
        return all_ids"""

    def WriteORCA6Output(self):
        pass
