import json
from pyscf import gto
from pyscf import grad
from pyscf import dft

metadata = {}

# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''H -1.0 0.0 0.0
O 0.0 0.0 0.0
H 0.0 1.0 0.0
''',
    basis='def2svp',
    unit = 'Ang',
    output = 'WaterTriplet_PySCFOutput.log',
    verbose = 4,
    max_memory = 1000,
    charge = 1,
    spin = 1
)
metadata['Identifier'] = 'WaterTriplet'
metadata['Method Type'] = 'DFT'
metadata['Method'] = 'roks m06_l'
metadata['Basis Set'] = 'def2svp'
metadata['Charge'] = 1
metadata['Multiplicity'] = 2

pyscfMolObj_calc = dft.ROKS(pyscfMolObj)
pyscfMolObj_calc.xc = 'm06_l'
pyscfMolObj_calc.grids.level = 5
pyscfMolObj_calc.grids.prune = True
pyscfMolObj_calc.kernel()
metadata['AO Labels'] = pyscfMolObj.ao_labels()
metadata['Electronic Energy (Eh)'] = pyscfMolObj_calc.e_tot
metadata['Two Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[1]
metadata['One Electron Energy (Eh)'] = pyscfMolObj_calc.energy_elec()[0] - pyscfMolObj_calc.energy_elec()[1]
metadata['Nuclear Repulsion Energy (Eh)'] = pyscfMolObj_calc.energy_nuc()

# Get Gradients
g = pyscfMolObj_calc.Gradients()
grad = g.kernel()
metadata['Gradients (Eh/Bohr)'] = grad.tolist()

# Write Fock Matrix
import numpy as np
F = pyscfMolObj_calc.get_fock()
metadata['Fock Matrix File Name'] = 'WaterTriplet_PySCFOutput.fock'
np.savetxt('WaterTriplet_PySCFOutput.fock', F, fmt='%.16e')

# Write metadata to .json file
with open('WaterTriplet_PySCFOutput.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)