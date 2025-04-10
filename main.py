import numpy as np

from module import crystallo as cr
from module import micromech as mm
from module import microstruct as ms

'''
Input
'''
# Unit cell parameters of the reference configuration
abc_ref = np.array([6.1748, 9.9126, 19.6020])
angle_ref = np.array([84.484, 86.208, 88.252])

# Unit cell parameters of the deformed configuration
abc_def = np.array([6.1899, 9.8715, 19.7418])
angle_def = np.array([90, 96.946, 90])

# Millar index from experiment
milfrac = np.array([0,0,1])

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
theta, n_true = cr.minang(n, n_n, milvec)

# Converts the analytical normal vector with the minimum angle to fractional coordinates
n_frac = cr.cart2frac(n_true, reflat)
n_frac = n_frac/np.linalg.norm(n_frac)

'''
Output
'''
print('COMPLETE: Minimum angle in degrees: ', theta)
print('          Analytical normal''s miller index', -n_frac)