"""
Testing for the math.py module.
"""

import qm2  # runs a test on the qm2 in the same directory as tests/
import pytest
import test_data # test sets for mymath functions stored in test_data.py


# Test mymath.add function
@pytest.mark.parametrize("a,b,expected", test_data.test_add_data)
def test_add(a, b, expected):
    assert qm2.mymath.add(a, b) == expected
    assert qm2.mymath.add(b, a) == expected


# Test mymath.sub function
@pytest.mark.parametrize("a,b,expected", test_data.test_sub_data)
def test_sub(a, b, expected):
    assert qm2.mymath.sub(a, b) == expected
    assert qm2.mymath.sub(b, a) == -expected


# Test mymath.mult function
@pytest.mark.parametrize("a,b,expected", test_data.test_mult_data)
def test_mult(a, b, expected):
    assert qm2.mymath.mult(a, b) == expected
    assert qm2.mymath.mult(b, a) == expected


# Test mymath.div function
@pytest.mark.parametrize("a,b,expected", test_data.test_div_data)
def test_div(a, b, expected):
    assert qm2.mymath.div(a, b) == expected


# Test mymath.mod function
@pytest.mark.parametrize("a,b,expected", test_data.test_mod_data)
def test_mod(a, b, expected):
    assert qm2.mymath.mod(a, b) == expected


# Test mymath.greater function
@pytest.mark.parametrize("a,b,expected", test_data.test_greater_data)
def test_greater(a, b, expected):
    assert qm2.mymath.greater(a, b) == expected
    assert qm2.mymath.greater(b, a) == expected
