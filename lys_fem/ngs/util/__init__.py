import builtins
import ngsolve

from .util import *
from .trials import TrialFunction, TestFunction, SolutionFunction, DifferentialSymbol
from .functions import det, inv, diag, offdiag, grad, sin, cos, tan, step, min, max, exp, sqrt, norm
from .space import FunctionSpace, H1, L2, NGSVariable, FiniteElementSpace

def eval(expr, dic={}):
    return builtins.eval(expr, globals(), dic)

dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
dimension = 3