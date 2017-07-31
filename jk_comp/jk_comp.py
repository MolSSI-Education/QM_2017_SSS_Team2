import numpy as np
import psi4
import jk_build
import time

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
        O
        H 1 1.1
        H 1 1.1 2 104
        """)

# Build a ERg tensor
basis = psi4.core.BasisSet.build(mol, target="cc-pVDZ")
mints = psi4.core.MintsHelper(basis)
g = np.array(mints.ao_eri())

# Symmetric random density
nbf = g.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
t_jref_start = time.time()
for i in range(0,100):
    J_ref = np.einsum("pqrs,rs->pq", g, D)
t_jref_stop = time.time()
print("J_ref time = ",(t_jref_stop - t_jref_start) / 100)
t_kref_start = time.time()
for i in range(0,100):
    K_ref = np.einsum("pqrs,qs->pr", g, D)
t_kref_stop = time.time()
print("K_ref time = ",(t_kref_stop - t_kref_start) / 100)

# Your implementation
t_jref_start = time.time()
for i in range(0,100):
    J = jk_build.J_build(g, D)
t_jref_stop = time.time()
print("J_build time = ",(t_jref_stop - t_jref_start) / 100)
t_kref_start = time.time()
for i in range(0,100):
    K = jk_build.K_build(g, D)
t_kref_stop = time.time()
print("K_build time = ",(t_kref_stop - t_kref_start) / 100)


# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))
