import builtins
import ngsolve

from .coef import NGSFunction, DomainWiseFunction
from .trials import TrialFunction, TestFunction, SolutionFunction, DifferentialSymbol
from .fields import VolumeField, SolutionFieldFunction, RandomFieldFunction
from .functions import det, inv, diag, offdiag, grad, sin, cos, tan, step, min, max, exp, sqrt, norm, einsum
from .space import FunctionSpace, H1, L2, FiniteElementSpace
from .solver import LinearSolver

def eval(expr, dic={}):
    return builtins.eval(expr, globals(), dic)

t = NGSFunction(ngsolve.Parameter(0), name="t", tdep=True)
dti = NGSFunction(ngsolve.Parameter(-1), name="dti", tdep=True)
stepn = NGSFunction(ngsolve.Parameter(0), name = "step", tdep=True)
dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
