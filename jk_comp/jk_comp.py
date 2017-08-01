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
basis = psi4.core.BasisSet.build(mol, target="cc-pvQZ")
mints = psi4.core.MintsHelper(basis)
g = np.array(mints.ao_eri())

# Symmetric random density
nbf = g.shape[0]
print("NBF is : {} ".format(nbf))
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

print("Computing J_ref and K_ref")
J_ref = np.einsum("pqrs,rs->pq", g, D)
K_ref = np.einsum("pqrs,qs->pr", g, D)

print("building JK_builder")
my_JK_bldr = jk_build.JKBuilder(g)

print("timing my implementation")
 #Your implementation
t_start = time.time()
for i in range(0,10):
    J,K = my_JK_bldr.compute(D)
t_stop = time.time()
print("JK_build time = ",(t_stop - t_start) / 10)

# print("Testing J build")
# J = my_JK_bldr.build_J_only(D)
# print("J is correct: %s" % np.allclose(J, J_ref))
# print("Testing K build")
# K = my_JK_bldr.build_K_only(D)
# print("K is correct: %s" % np.allclose(K, K_ref))
# if not  np.allclose(K, K_ref):
#     print("Ref K ")
#     print(K_ref)
#     print("My K")
#     print(K)




