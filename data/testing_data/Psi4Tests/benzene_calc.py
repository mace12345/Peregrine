import time

start = time.time()

import json
import platform
import resource
import psi4
import basis_set_exchange as bse
import numpy as np

T = 298.15


def get_max_rss_mb():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return raw / (1024 * 1024)
    return raw / 1024


psi4.set_output_file("Benzene.out", False)
psi4.set_memory("1000 MB")
psi4.set_num_threads(4)

metadata = {
    "Identifier": "Benzene",
    "Charge": 0,
    "Multiplicity": 1,
    "CPU cores used": 4,
}
with open("Benzene.meta.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Define psi4 molecule object
psi4MolObj = psi4.geometry(
    """
0 1
C -1.2131 -0.6884 0.0004
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

units angstrom

""",
)

# Define basis sets
element_basis_map = {
    "H": "def2-tzvppd",
    "C": "def2-tzvppd",
}
combined_basis = "\n".join(
    bse.get_basis(basisname, elements=[symbol], fmt="psi4", header=False)
    for symbol, basisname in element_basis_map.items()
)
psi4.basis_helper(
    f"""
assign mybasis
[ mybasis ]
spherical
{combined_basis}
""",
    name="mybasis",
    key="BASIS",
)
metadata["Basis Set"] = element_basis_map
jkfit_basis_map = {
    "H": "def2-universal-jkfit",
    "C": "def2-universal-jkfit",
}
combined_jkfit = "\n".join(
    bse.get_basis(basisname, elements=[symbol], fmt="psi4", header=False)
    for symbol, basisname in jkfit_basis_map.items()
)
psi4.basis_helper(
    f"""
assign myjkfit
[ myjkfit ]
spherical
{combined_jkfit}
""",
    name="myjkfit",
    key="DF_BASIS_SCF",
)
psi4.set_options(
    {
        "guess": "core",
        "scf_type": "df",
    }
)

# Set the shell restriction
psi4.set_options({"reference": "uhf"})
metadata["Method"] = "uhf wb97m-d3bj"

# Set up and run calculation

RAM = int(get_max_rss_mb())
end = time.time()
time_taken = int(round(end - start, 0))
metadata["Time Taken (s)"] = time_taken
metadata["Maximum RAM used (MB)"] = RAM
with open("Benzene.meta.json", "w") as f:
    json.dump(metadata, f, indent=2)
