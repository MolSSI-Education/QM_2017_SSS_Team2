"""
Testing for the math.py module.
"""

import qm2  # runs a test on the qm2 in the same directory as tests/
import pytest
import test_data # test sets for math functions stored in test_data.py


# Test math.add function
@pytest.mark.parametrize("a,b,expected", test_data.test_add_data)
def test_add(a, b, expected):
    assert qm2.math.add(a, b) == expected
    assert qm2.math.add(b, a) == expected


# Test math.sub function
@pytest.mark.parametrize("a,b,expected", test_data.test_sub_data)
def test_sub(a, b, expected):
    assert qm2.math.sub(a, b) == expected
    assert qm2.math.sub(b, a) == -expected


# Test math.mult function
@pytest.mark.parametrize("a,b,expected", test_data.test_mult_data)
def test_mult(a, b, expected):
    assert qm2.math.mult(a, b) == expected
    assert qm2.math.mult(b, a) == expected


# Test math.div function
@pytest.mark.parametrize("a,b,expected", test_data.test_div_data)
def test_div(a, b, expected):
    assert qm2.math.div(a, b) == expected


# Test math.mod function
@pytest.mark.parametrize("a,b,expected", test_data.test_mod_data)
def test_mod(a, b, expected):
    assert qm2.math.mod(a, b) == expected


# Test math.greater function
@pytest.mark.parametrize("a,b,expected", test_data.test_greater_data)
def test_greater(a, b, expected):
    assert qm2.math.greater(a, b) == expected
    assert qm2.math.greater(b, a) == expected
