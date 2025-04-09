import numpy as np

from module import crystallo as cr
from module import micromech as mm
from module import microstruct as ms

'''
Input
'''
# Unit cell parameters of the reference configuration
abc_ref = np.array([7.6, 8.58, 17.23])
angle_ref = np.array([78.22, 86.71, 72.1])

# Unit cell parameters of the deformed configuration
abc_def = np.array([7.76, 7.74, 16.94])
angle_def = np.array([77.80, 88.5, 82.2])

# Millar index from experiment
milfrac = np.array([0,1,0])

'''
Calculation
'''
# Converts unit cell parameters (fractional coordinate) to lattice vectors (Cartesian coordinate)
reflat = cr.unit2vect(abc_ref, angle_ref)
deflat = cr.unit2vect(abc_def, angle_def)

# Converts Miller indices to Cartesian vectors
milvec = cr.frac2cart(milfrac, reflat)

# Calculates the deformation gradient and its stretch and rotation
F = mm.defgrad(reflat, deflat)
U, Q = mm.streten(F)

# Calculates the kinematic Compatibility
n, a, s, nu, K, n_n, a_n, s_n, nu_n, K_n = ms.kincomp(F, np.eye(3))

# Finds the minimum angle with the possible normal vector combination
theta = cr.minang(n, n_n, milvec)

'''
Output
'''
print('COMPLETE: Minimum angle in degrees is')
print(theta)