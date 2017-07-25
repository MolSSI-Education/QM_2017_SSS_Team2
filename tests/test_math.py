"""
Testing for the math.py module.
"""

import qm2  # runs a test on the qm2 in the same directory as tests/
import pytest

test_add_data = [
    (2, 5, 7),
    (1, 2, 3),
    (11, 9, 20),
    (11, 0, 11),
    (0, 0, 0),
]

test_mult_data = [
    (2, 5, 10),
    (1, 2, 2),
    (11, 9, 99),
    (11, 0, 0),
    (0, 0, 0),
]
test_greater_data = [
    (2, 5, 5),
    (1, 2, 2),
    (11, 9, 11),
    (11, 0, 11),
    (0, 0, 0),
]


@pytest.mark.parametrize("a,b,expected", test_add_data)
def test_add(a, b, expected):
    assert qm2.math.add(a, b) == expected
    assert qm2.math.add(b, a) == expected


@pytest.mark.parametrize("a,b,expected", test_mult_data)
def test_mult(a, b, expected):
    assert qm2.math.mult(a, b) == expected
    assert qm2.math.mult(b, a) == expected


@pytest.mark.parametrize("a,b,expected", test_greater_data)
def test_greater(a, b, expected):
    assert qm2.math.greater(a, b) == expected
    assert qm2.math.greater(b, a) == expected
