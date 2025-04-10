from module import crystallo as cr
from module import micromech as mm
from module import microstruct as ms
import numpy as np

'''
Input Parameters
'''
# Unit cell parameters of the reference configuration
#FORM II
abc_ref = np.array([7.6, 8.58, 17.23])
angle_ref = np.array([78.22, 86.71, 72.1])

#TWIN
#abc_def = np.array([7.76, 7.74, 16.95])
#angle_def = np.array([77.87, 88.54, 82.2])

#FORM I
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
theta, n_true = cr.minang(n, n_n, milvec)

# Converts the analytical normal vector with the minimum angle to fractional coordinates
n_frac = cr.cart2frac(n_true, reflat)
n_frac = n_frac/np.linalg.norm(n_frac)

'''
Output
'''
print('COMPLETE: ')
print(' Minimum angle in degrees: ', theta)
print(' Analytical normal''s miller index: ', n_frac, '\n')