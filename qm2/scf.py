import numpy as np
import psi4

np.set_printoptions(suppress=True, precision=4)


def build_geom(molecule_string):
    mol = psi4.geometry(molecule_string)

    # Build a molecule
    mol.update_geometry()
    mol.print_out()

    return mol


# Define general function to diagonalize Fock Matrix using Orthogonalization matrix A
def diag(F, A):
    Fp = A.T @ F @ A
    eps, Cp = np.linalg.eigh(Fp)
    C = A @ Cp
    return eps, C


def JK_build(g, D):
    # F_pq = H_pq + 2 * G_pqrs D_rs - G_prqs D_rs
    J = np.einsum("pqrs,rs->pq", g, D)
    K = np.einsum("prqs,rs->pq", g, D)
    return J, K



def update_fock(g, D, H, F_old, damp=False, damp_value=0.20):
    J,K = JK_build(g, D)

    F_new = H + 2.0 * J - K

    if (damp):
        F = (damp_value) * F_old + (1.0-damp_value) * F_new
    else:
        F = F_new

    return F
        




def run_scf(molstr, nel, e_conv=1.e-6, d_conv=1.e-6, damp=False):
    mol = build_geom(molstr)

    # Build a basis
    bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
    bas.print_out()

    # Build a MintsHelper
    mints = psi4.core.MintsHelper(bas)
    nbf = mints.nbf()
    if (nbf > 100):
        raise Exception("More than 100 basis functions!")


    # Potential Energy Integrals
    V = np.array(mints.ao_potential())

    # Kinetic Energy Integrals
    T = np.array(mints.ao_kinetic())

    # Core Hamiltonian
    H = T + V
    # print(H)

    # Overlap Integrals
    S = np.array(mints.ao_overlap())
    # print(S.shape)

    # Two-electron Repulsion Integrals
    g = np.array(mints.ao_eri())
    # print(I.shape)


    # Orthogonalization Matrix
    A = mints.ao_overlap()
    A.power(-0.5, 1.e-14)
    A = np.array(A)

    # Check Orthogonalization
    # print(A @ S @ A)


    # Transform Fock Matrix to Orthogonal Basis
    F = H
    eps, C = diag(F, A)
    Cocc = C[:, :nel]
    D = Cocc @ Cocc.T

    E_old = 0.0
    E_diff = 100000
    grad_rms = 100000   # Initialize to a 
    for iteration in range(50):
    
        F_old = F
        F = update_fock(g, D, H, F_old, iteration>5)

        # Build the AO gradient
        grad = F @ D @ S - S @ D @ F

        grad_rms = np.mean(grad ** 2) ** 0.5

        E_electronic = np.sum((F + H) * D)
        E_total = E_electronic + mol.nuclear_repulsion_energy()

        E_diff = E_total - E_old
        E_old = E_total
        #F_old = F

        print("Iter= % 3d E = % 16.12f E_diff = % 8.4e D_rms = % 8.4e" % (
            iteration, E_total, E_diff, grad_rms))

        # Break if e_conv and d_conv are met
        if (abs(E_diff) < e_conv) and (abs(grad_rms) < d_conv):
            print("\nEnergy convergence and Density convergence were met!")
            break

        eps, C = diag(F, A)
        Cocc = C[:, :nel]
        D = Cocc @ Cocc.T


    # Exercise
    # np.sum(np.diag(A @ B) == np.sum(A * B)
    # print(F)


    print("\nSCF has finished!\n")
    if (abs(E_diff) > e_conv) or (abs(grad_rms) > d_conv):
        print("\nConvergence criteria were not met!\n")

    psi4.set_options({"scf_type": "pk"})
    psi4_energy = psi4.energy("SCF/aug-cc-pVDZ", molecule=mol)

    print("Energy matches Psi4 %s" % np.allclose(psi4_energy, E_total))

    return E_total
