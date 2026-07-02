import json
from pyscf import gto
from pyscf import grad
from pyscf import dft

metadata = {}

# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''H -1.09 0.05 0.02
O 0.00123 0.0034 0.0087
H 0.08 1.00064 0.076
H 0.0062 0.078 1.088
''',
    basis='def2svp',
    unit = 'Ang',
    output = 'WaterCation_PySCFOutput.log',
    verbose = 4,
    max_memory = 1000,
    charge = 1,
    spin = 0
)
metadata['Identifier'] = 'WaterCation'
metadata['Method Type'] = 'DFT'
metadata['Method'] = 'rks wb97m_v'
metadata['Basis Set'] = 'def2svp'
metadata['Charge'] = 1
metadata['Multiplicity'] = 1

pyscfMolObj_calc = dft.RKS(pyscfMolObj)
pyscfMolObj_calc.xc = 'wb97m_v'
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
metadata['Fock Matrix File Name'] = 'WaterCation_PySCFOutput.fock'
np.savetxt('WaterCation_PySCFOutput.fock', F, fmt='%.16e')

# Write metadata to .json file
with open('WaterCation_PySCFOutput.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)