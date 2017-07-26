"""
Testing for the math.py module.
"""

import qm2  # runs a test on the qm2 in the same directory as tests/
import pytest
import test_data


@pytest.mark.parametrize("a,b,expected", test_data.test_add_data)
def test_add(a, b, expected):
    assert qm2.math.add(a, b) == expected
    assert qm2.math.add(b, a) == expected


@pytest.mark.parametrize("a,b,expected", test_data.test_mult_data)
def test_mult(a, b, expected):
    assert qm2.math.mult(a, b) == expected
    assert qm2.math.mult(b, a) == expected


@pytest.mark.parametrize("a,b,expected", test_data.test_greater_data)
def test_greater(a, b, expected):
    assert qm2.math.greater(a, b) == expected
    assert qm2.math.greater(b, a) == expected
