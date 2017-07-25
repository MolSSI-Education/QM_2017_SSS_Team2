"""
Testing for the math.py module.
"""

import lbq as lbq #runs a test on the lbq in the same directory as tests/
import pytest

def test_add():
    assert lbq.math.add(5, 2) == 7
    assert lbq.math.add(2, 5) == 7

def test_mult():
    assert lbq.math.mult(5, 2) == 10
    assert lbq.math.mult(2, 5) == 10

def test_greater():
    assert lbq.math.greater(5, 2) == 5
    assert lbq.math.greater(2, 5) == 5
    assert lbq.math.greater(5, 5) == 5

