import numpy as np

# Ni-49.75at.%Ti example
alp = 1.0243
gam = 0.9563
dlt = 0.058
eta = -0.0427

U1 = np.array([[gam, eta, eta],
               [eta, alp, dlt],
               [eta, dlt, alp]])
U2 = np.array([[gam, -eta, -eta],
               [-eta, alp, dlt],
               [-eta, dlt, alp]])
U3 = np.array([[gam, -eta, eta],
               [-eta, alp, -dlt],
               [eta, -dlt, alp]])
U4 = np.array([[gam, eta, -eta],
               [eta, alp, -dlt],
               [-eta, -dlt, alp]])

U5 = np.array([[alp, eta, dlt],
               [eta, gam, eta],
               [dlt, eta, alp]])
U6 = np.array([[alp, -eta, dlt],
               [-eta, gam, -eta],
               [dlt, -eta, alp]])
U7 = np.array([[alp, -eta, -dlt],
               [-eta, gam, eta],
               [-dlt, eta, alp]])
U8 = np.array([[alp, eta, -dlt],
               [eta, gam, -eta],
               [-dlt, -eta, alp]])

U9 = np.array([[alp, dlt, eta],
               [dlt, alp, eta],
               [eta, eta, gam]])
U10 = np.array([[alp, dlt, -eta],
                [dlt, alp, -eta],
                [-eta, -eta, gam]])
U11 = np.array([[alp, -dlt, eta],
                [-dlt, alp, -eta],
                [eta, -eta, gam]])
U12 = np.array([[alp, -dlt, -eta],
                [-dlt, alp, eta],
                [-eta, eta, gam]])


def kinComp(F, G):

    G_inv = np.linalg.inv(G)
    C = G_inv.T @ F.T @ F @ G_inv
    
    if np.all(np.equal(C , np.eye(3))):
        return 0
    
    lam, e = np.linalg.eig(C)
    
    #sorting eigenvector and values from low to high (ascending order)
    idx = lam.argsort()
    lam = lam[idx]
    e = e[:,idx]

    #checks if kinematic compatibility is met
    if not (lam[0] <= 1 and np.isclose(lam[1], 1.0, atol=1e-3) and lam[2] >= 1):
        print('WARNING: eigenvalues does not exactly satisfy kinematic compatibility')
        print(' lam1 = ', lam[0])
        print(' lam2 = ', lam[1])
        print(' lam3 = ', lam[2] , '\n')
    
    #calculates twinning for type I
    kap = 1

    n = (np.sqrt(lam[2]) - np.sqrt(lam[0]))/np.sqrt(lam[2]-lam[0]) * (-np.sqrt(1-lam[0]) * (G.T @ e[:,0]) + kap*np.sqrt(lam[2]-1)* (G.T @ e[:,2]))
    rho = np.linalg.norm(n)
    n = n/rho

    a = rho * (np.sqrt(lam[2]*(1-lam[0])/(lam[2]-lam[0]))*e[:,0] + 
               kap*np.sqrt(lam[0]*(lam[2]-1)/(lam[2]-lam[0]))*e[:,2])

    s = np.linalg.norm(a) * np.linalg.norm( (G_inv @ n) )
    nu = a/np.linalg.norm(a)
    K = (G_inv @ n)/ np.linalg.norm( (G_inv @ n) )
    
    #calculates twinning for type II
    kap = -1

    n_n = (np.sqrt(lam[2]) - np.sqrt(lam[0]))/np.sqrt(lam[2]-lam[0]) * (-np.sqrt(1-lam[0]) * (G.T @ e[:,0]) + kap*np.sqrt(lam[2]-1)* (G.T @ e[:,2]))
    rho_n = np.linalg.norm(n_n)
    n_n = n_n/rho_n

    a_n = rho_n * (np.sqrt(lam[2]*(1-lam[0])/(lam[2]-lam[0]))*e[:,0] + 
               kap*np.sqrt(lam[0]*(lam[2]-1)/(lam[2]-lam[0]))*e[:,2])

    s_n = np.linalg.norm(a_n) * np.linalg.norm( (G_inv @ n_n) )
    nu_n = a_n/np.linalg.norm(a_n)
    K_n = (G_inv @ n_n)/ np.linalg.norm( (G_inv @ n_n) )

    return n, a, s, nu, K, n_n, a_n, s_n, nu_n, K_n

n, a, s, nu, K, n_n, a_n, s_n, nu_n, K_n = kinComp(U1, U2)

np.set_printoptions(precision=4)

print('Type I')
print('n = ', n)
print('a = ', a)
print('s = ', s)
print('nu = ', nu)
print('K = ', K,'\n')

print('Type II')
print('n = ', n_n)
print('a = ', a_n)
print('s = ', s_n)
print('nu = ', nu_n)
print('K = ', K_n, '\n')