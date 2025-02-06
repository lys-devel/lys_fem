import builtins
from .util import *
from .space import FunctionSpace, H1, L2, NGSVariable, FiniteElementSpace
from .functions import det, inv, diag, offdiag, grad, sin, cos, tan, step, min, max, exp, sqrt, norm, prod

def eval(expr, dic={}):
    return builtins.eval(expr, globals(), dic)