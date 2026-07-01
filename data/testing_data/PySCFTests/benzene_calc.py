from pyscf import gto
from pyscf import scf
from pyscf import grad
from pyscf.tools import fcidump

# Define Molecule
pyscfMolObj = gto.Mole(
    atom='''C -1.2131 -0.6884 0.0004
C -1.2028 0.7064 0.0001
C -0.0103 -1.3948 0.0002
C 0.0104 1.3948 0.0001
C 1.2028 -0.7063 0.0006
C 1.2131 0.6884 0.0004
H -2.6091649351 -1.480627435 0.0008603297
H -2.5868635378 1.5192554066 0.0002150701
H -0.0221531095 -2.9999181747 0.0004301575
H 0.0223681766 2.9999166019 0.0002150786
H 2.5869572201 -1.519095348 0.0012904675
H 2.6091649351 1.480627435 0.0008603297
''',
    basis='def2svp',
    unit = 'Ang',
    output = 'Benzene_PySCFOutput.log',
    verbose = 4,
    max_memory = 1000,
)
pyscfMolObj.charge = 0
pyscfMolObj.spin = 0

pyscfMolObj_calc = scf.RHF(pyscfMolObj)
pyscfMolObj_calc.kernel()

g = pyscfMolObj_calc.Gradients()
g.kernel()

# Write Fock Matrix
import io
import numpy as np
F = pyscfMolObj_calc.get_fock()
np.savetxt('Benzene_PySCFOutput.fock', F, fmt='%.16e')