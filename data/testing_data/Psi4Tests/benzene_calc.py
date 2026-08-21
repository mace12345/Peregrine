import resource
import psi4 

psi4.set_output_file('Benzene.out', False)
psi4.set_memory('1000 MB')
psi4.set_num_threads(4)

psi4MolObj = psi4.geometry(
    '''
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
''',
)

props = ['DIPOLE', 'QUADRUPOLE', 'WIBERG_LOWDIN_INDICES', 'MAYER_INDICES']
psi4.optimize('wb97m-d3bj/def2-tzvppd', properties=props)

print(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))