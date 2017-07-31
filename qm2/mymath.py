"""
A small set of functions for doing math operations.
"""

# Add Function
def add(arg1, arg2):
    """
    The add() function takes two arguments and returns their sum
    """
    return arg1 + arg2


# Subtract Function
def sub(arg1, arg2):
    """
    The sub() function takes two arguments and returns their difference
    """
    return arg1 - arg2


# Multiply Function
def mult(arg1, arg2):
    """
    The mult() function takes two arguments and returns their product
    """
    return arg1 * arg2


# Divide Function
def div(arg1, arg2):
    """
    The div() function takes two arguments and returns their quotient
    """
    return float(arg1) / float(arg2)


# Modulus Function
def mod(arg1, arg2):
    """
    The mod() function takes two arguments and returns their modulo
    """
    return arg1 % arg2

# Greather Function
def greater(arg1, arg2):
    """
    The greater() function takes two arguments and returns whicever is greater
    """
    if arg1 >= arg2:
        return arg1
    else: 
        return arg2
