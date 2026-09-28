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
    if platform.system() == 'Darwin':
        return raw / (1024 * 1024)
    return raw / 1024

psi4.set_output_file('ETncAA727.out', False)
psi4.set_memory('4000 MB')
psi4.set_num_threads(1)

metadata = {
'Identifier': 'ETncAA727',
'Charge': 2,
'Multiplicity': 1,
'CPU cores used': 1
}
with open('ETncAA727.meta.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Define psi4 molecule object
psi4MolObj = psi4.geometry('''
2 1
C -5.4412356219 8.2436698105 1.2934751355
C -5.2267318452 7.0896771559 0.7009140464
C -6.1802953779 5.9459755336 0.7182887095
C -6.6462886803 5.6242332022 -0.682546893
O -7.6055813254 4.5556535187 -0.6769630929
C -7.0977065588 3.3188591787 -0.6487690988
O -5.9135691245 3.0403027981 -0.6153478419
N -8.0899343778 2.4071435729 -0.6634529137
C -7.8002534027 0.9946963099 -0.6515237352
C -7.4361299491 0.4981859998 0.7416066528
C -7.4609709425 -1.0110868033 0.8736016811
C -6.513021834 -1.7857851551 -0.0216842234
C -5.0556722262 -1.6570611115 0.3705161658
N -4.264253254 -2.6397831166 -0.3577327817
C -3.3549366492 -3.4075658959 0.1870356396
O -3.1200819149 -3.4514704262 1.413536637
C -2.5699661232 -4.3179971862 -0.7255330439
N -1.2043171599 -4.2453600689 -0.2555295694
C -0.1662876789 -4.7131682915 -1.0616855738
O -0.3282762742 -5.0887129852 -2.1845535274
C 1.1709065423 -4.8039877809 -0.3481341512
C 1.6428983791 -6.242564647 -0.5102671638
C 0.7214447065 -7.3073718045 0.0760930758
C 0.8466074478 -7.4483094401 1.554496445
O 0.0785895578 -6.9520881035 2.373238261
O 1.8731028997 -8.1779447099 1.9181153977
N 1.0736669225 -4.3625185812 1.0372711822
C 2.3232212137 -4.4583110914 1.7852486056
C -3.1559451309 -5.7258555443 -0.5265281724
C -2.9274357734 -6.6787883713 -1.6769194345
O -2.6025084977 -6.2302874982 0.6928858714
C -4.4637996855 -0.3271121578 -0.0482451636
O -4.4327052166 0.0054917891 -1.1987769028
O -3.9269389153 0.3114417232 0.9786417525
C -3.367065629 1.616117252 0.6528688857
H -4.7272207468 9.0520912077 1.2479212493
H -6.344310269 8.4342747254 1.8588517246
H -4.3185534523 6.9268054472 0.1354617227
H -5.6970923223 5.06248164 1.1320261812
H -7.0402541811 6.1867110187 1.3424142588
H -7.1725372211 6.4703403283 -1.1188013846
H -5.8062629671 5.3510522329 -1.316604578
H -9.0397314005 2.7298470951 -0.6477644563
H -6.9740119769 0.8158781905 -1.3383103437
H -8.6840246218 0.4665855869 -1.0172409156
H -6.4694433466 0.9087796978 1.0148198351
H -8.1576233436 0.8979984602 1.4516919197
H -7.27352848 -1.2803302838 1.9148619349
H -8.47128497 -1.3516638259 0.6432110222
H -6.6322360182 -1.4617155609 -1.0531520665
H -6.7859212404 -2.8388303012 0.0304351228
H -4.9278107607 -1.8360246138 1.4335510968
H -4.3474337823 -2.5665431082 -1.3626044098
H -2.6494783919 -4.0434178091 -1.7754738336
H -0.989788899 -3.3359576653 0.1200635095
H 1.8802598588 -4.1901433542 -0.9063836143
H 1.696033901 -6.4117718217 -1.5822166622
H 2.6504552255 -6.3616207171 -0.1233352483
H -0.3134892376 -7.090574578 -0.1733702802
H 0.993076408 -8.2595642213 -0.3719507298
H 1.9091299815 -8.3165074267 2.8782623256
H 0.8503280883 -3.3844946064 1.0262619908
H 3.1573446388 -3.9809358835 1.2714189052
H 2.5822276736 -5.4981534928 1.9648384999
H 2.1799908855 -3.9736697274 2.7488224905
H -4.2360537118 -5.6091218308 -0.4009569722
H -1.8802091234 -6.8452696014 -1.8852094082
H -3.4088276492 -7.6335349521 -1.4807351838
H -3.3752738581 -6.2768853837 -2.5819675385
H -2.9781214174 -7.1010428738 0.85638155
H -2.5102792724 1.4905678254 -0.005446224
H -3.0638585009 2.0418727897 1.6042372478
H -4.121371472 2.2196665858 0.1698726441
Ca -1.2088774448 -4.9199472227 2.2185502826
O -3.0621578064 -5.0203067298 3.6475882021
H -3.7302819097 -4.5174348568 3.1655213939
H -3.4556853598 -5.3634368541 4.4558844123
O -0.9284166654 -2.5583335907 2.6114795837
H -0.5656381888 -1.961457154 3.2730992043
H -1.8286119402 -2.2712825429 2.4024323022
O -0.0790976632 -4.9607140584 4.2873698144
H 0.3155177741 -5.8363070854 4.3675100163
H -0.0481162713 -4.5392339468 5.1521850112

units angstrom
symmetry c1
noreorient
nocom
''',
)

# Define basis sets
element_basis_map = {
    'C': 'def2-svp',
    'O': 'def2-svp',
    'N': 'def2-svp',
    'H': 'def2-svp',
    'Ca': 'def2-svp',
}
combined_basis = '\n'.join(
    bse.get_basis(basisname, elements=[symbol], fmt='psi4', header=False)
    for symbol, basisname in element_basis_map.items()
)
psi4.basis_helper(f'''
assign mybasis
[ mybasis ]
spherical
{combined_basis}
''', name='mybasis', key='BASIS')
metadata['Basis Set'] = element_basis_map
jkfit_basis_map = {
    'C': 'def2-universal-jkfit',
    'O': 'def2-universal-jkfit',
    'N': 'def2-universal-jkfit',
    'H': 'def2-universal-jkfit',
    'Ca': 'def2-universal-jkfit',
}
combined_jkfit = '\n'.join(
    bse.get_basis(basisname, elements=[symbol], fmt='psi4', header=False)
    for symbol, basisname in jkfit_basis_map.items()
)
psi4.basis_helper(f'''
assign myjkfit
[ myjkfit ]
spherical
{combined_jkfit}
''', name='myjkfit', key='DF_BASIS_SCF')
psi4.set_options({
    'guess': 'core',
    'scf_type': 'df',
})

# Set the shell restriction
psi4.set_options({'reference': 'rhf'})
metadata['Method'] = 'rhf m06-2x'

# Set options
psi4.set_options({'ddx': True, 'ddx_model': 'pcm', 'ddx_solvent': 'water', 'ddx_radii_set': 'uff'})

# Set up and run calculation
try:
    e_sp, wfn = psi4.energy(
        'm06-2x',
        molecule=psi4MolObj,
        return_wfn=True,
    )
except psi4.driver.p4util.exceptions.SCFConvergenceError as exc:
    metadata['Maximum RAM used (MB)'] = int(get_max_rss_mb())
    metadata['SCF error at failure'] = {
        'iteration': exc.iteration,
        'e_conv': exc.e_conv,
        'd_conv': exc.d_conv,
    }
    failed_wfn = exc.wfn  # partial wavefunction at the point of failure
    coords_bohr = np.array(failed_wfn.molecule().geometry())
    metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
    with open('ETncAA727.meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    exit()
# Save properties
coords_bohr = np.array(wfn.molecule().geometry())
grad = np.array(wfn.gradient())
basis = wfn.basisset()
metadata['Electronic Energy (Eh)'] = psi4.variable('CURRENT ENERGY')
metadata['One Electron Energy (Eh)'] = psi4.variable('ONE-ELECTRON ENERGY')
metadata['Two Electron Energy (Eh)'] = psi4.variable('TWO-ELECTRON ENERGY')
metadata['Nuclear Repulsion Energy (Eh)'] = psi4.variable('NUCLEAR REPULSION ENERGY')
metadata['Gradient (Eh/Bohr)'] = grad.tolist()
metadata['Coordinates (Bohr)'] = coords_bohr.tolist()
metadata['Number of Primitive Basis Functions'] = basis.nprimitive()
# Calculate and save more properties
psi4.oeprop(
    wfn,
    'DIPOLE',
    'QUADRUPOLE',
    'MULLIKEN_CHARGES',
    'LOWDIN_CHARGES',
    'WIBERG_LOWDIN_INDICES',
    'MAYER_INDICES',
)
metadata['Dipole'] = np.array(wfn.variable('CURRENT DIPOLE')).tolist()
metadata['Quadrupole'] = np.array(wfn.variable('QUADRUPOLE')).tolist()
metadata['Mulliken Charges'] = np.array(wfn.variable('MULLIKEN CHARGES')).tolist()
metadata['Lowdin Charges'] = np.array(wfn.variable('LOWDIN CHARGES')).tolist()
metadata['Wiberg Bond Orders'] = np.array(wfn.array_variable('WIBERG LOWDIN INDICES')).tolist()
metadata['Mayer Bond Orders']  = np.array(wfn.array_variable('MAYER INDICES')).tolist()
# Save Fock matricies
F_ao = np.array(wfn.Fa_subset('AO'))
metadata['Fock Matrix File Name'] = 'ETncAA727.fock'
np.savetxt('ETncAA727.fock', F_ao, fmt='%.16e')
# Get HOMO and LUMO energies
homo_idx = wfn.nalpha() - 1
lumo_idx = wfn.nalpha()
eps_a = wfn.epsilon_a_subset("AO", "ALL").np
metadata['HOMO Energy (Eh)'] = float(eps_a[homo_idx])
metadata['LUMO Energy (Eh)'] = float(eps_a[lumo_idx])

RAM = int(get_max_rss_mb())
end = time.time()
time_taken = int(round(end - start, 0))
metadata['Time Taken (s)'] = time_taken
metadata['Maximum RAM used (MB)'] = RAM
with open('ETncAA727.meta.json', 'w') as f:
   json.dump(metadata, f, indent=2)
