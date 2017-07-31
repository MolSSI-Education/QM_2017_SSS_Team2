"""
Testing for the scf.py module.
"""

import qm2
import pytest
import test_data
import numpy as np

@pytest.mark.parametrize("molstr,E_nuc",test_data.test_build_geom_data)
def test_build_geom(molstr, E_nuc):
    mol = qm2.scf.build_geom(molstr)
    assert mol.nuclear_repulsion_energy() == E_nuc

#@pytest.mark.parametrize("",)
#def test_diag(F, eps, C):
#    eps, C = qm2.scf.diag(F)
#    assert qm2.scf.diag(F) == eps, C
#    assert False

@pytest.mark.parametrize("molstr,E_total",test_data.test_E_scf_data)
def test_run_scf(molstr, E_total):
    mol = qm2.scf.build_geom(molstr)
    #assert qm2.scf.run_scf(mol) == E_total
    assert np.allclose(qm2.scf.run_scf(mol), E_total)
