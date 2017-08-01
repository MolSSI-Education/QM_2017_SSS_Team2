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
print(g)

# Symmetric random density
nbf = g.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
J_ref = np.einsum("pqrs,rs->pq", g, D)
K_ref = np.einsum("pqrs,qs->pr", g, D)

# Your new class implementation
JK_bldr = jk_build.JKBuilder(g)
J = JK_bldr.J_build(D)
print(J_ref)
print("\n\n\n")
print(J)
#JK_bldr.K_build(D)

# Your old implementation
#J = jk_build.J_build(g, D)
#K = jk_build.K_build(g, D)

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
#print("K is correct: %s" % np.allclose(K, K_ref))
