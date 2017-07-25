"""
Testing for the math.py module.
"""

import qm2 #runs a test on the qm2 in the same directory as tests/
import pytest

def test_add():
    assert qm2.math.add(5, 2) == 7
    assert qm2.math.add(2, 5) == 7

def test_mult():
    assert qm2.math.mult(5, 2) == 10
    assert qm2.math.mult(2, 5) == 10

def test_greater():
    assert qm2.math.greater(5, 2) == 5
    assert qm2.math.greater(2, 5) == 5
    assert qm2.math.greater(5, 5) == 5

