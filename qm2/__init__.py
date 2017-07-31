"""
This is the base file of QM2.
"""

from . import mymath
from .mymath import add  
from .mymath import sub  
from .mymath import mult 
from .mymath import div
from .mymath import mod
from .mymath import greater

from . import scf
from .scf import build_geom
#from .scf import update_fock
#from .scf import diag
from .scf import run_scf
