from peregrine.atom import Atom
from peregrine.molecule import Molecule
from peregrine.moleculeset import MoleculeSet

import torch

from pathlib import Path

import numpy as np


class AtomicMLP(torch.nn.Module):
    """One atom's AEF descriptor -> one scalar atomic energy"""

    def __init__(
        self,
        n_features,
        hidden=(256, 256),
    ):
        super().__init__()
        dims, layers = [n_features, *hidden], []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [
                torch.nn.Linear(d_in, d_out),
                # torch.nn.SiLU()
            ]
        layers += [torch.nn.Linear(dims[-1], 1)]  # final scalar
        self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)  # (n_atoms, 1)


class AtomicEnergyModel(torch.nn.Module):
    def __init__(self, n_features, hidden=(256, 256)):
        super().__init__()
        # One simple MLP to rule all the atoms
        self.net = AtomicMLP(n_features, hidden)

    def forward(self, descriptors, batch_index, n_molecules):
        atomic_e = self.net(descriptors).squeeze(-1)  # (n_atoms,)
        mol_e = torch.zeros(n_molecules, device=descriptors.device)
        mol_e.index_add_(0, batch_index, atomic_e)
        return mol_e, atomic_e


if "__main__" == __name__:
    aMLP = AtomicMLP(n_features=165, hidden=(256,))
    print(aMLP.net)

    ms = MoleculeSet()
    ms.ReadMolFileDirectory(
        mol_file_directory=f"{str(Path(__file__).parent.parent.parent).replace("\\", "/")}/data/testing_data/AtomicDescriptors/CoIILig-S2_SOAP-5-2-2"
    )

    def BuildTensors(molSetObj: MoleculeSet):
        descriptors, batch_index, dft_energies, C = [], [], [], []
        # element counts use atomic number as the column index; size the matrix
        # to the largest Z you'll see (reuse the same n_elements at inference)
        for molObj_idx, Identifier in enumerate(molSetObj.MoleculesDict):
            molObj = molSetObj.MoleculesDict[Identifier]
            if molObj.electronic_energy is None:
                raise ValueError(f"{molObj.Identifier} has no electronic_energy")
            for atom in molObj.AtomsList:
                assert (
                    atom.SOAPDescriptor is not None
                ), f"{atom.Label}: no SOAPDescriptor"
                assert atom.AtomicNumber is not None, f"{atom.Label}: no AtomicNumber"
                soap = np.asarray(atom.SOAPDescriptor, dtype=np.float32)  # (165,)
                extra = np.array(
                    [atom.AtomicNumber, atom.FormalCharge, atom.Multiplicity],
                    dtype=np.float32,
                )  # (3,)
                descriptor = np.concatenate([soap, extra])  # (168,)
                descriptors.append(descriptor)
                print(len(descriptor))
                batch_index.append(molObj_idx)
            dft_energies.append(molObj.electronic_energy)
        descriptors = torch.tensor(np.stack(descriptors), device="cpu")
        batch_index = torch.tensor(batch_index, dtype=torch.long, device="cpu")
        dft_energies = torch.tensor(dft_energies, dtype=torch.float32, device="cpu")
        print(C)
        C = torch.tensor(C, dtype=torch.float32, device="cpu")
        print(C)
        return (
            descriptors,
            batch_index,
            dft_energies,
        )

    BuildTensors(ms)
