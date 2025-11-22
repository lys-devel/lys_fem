import builtins
import ngsolve

from .operators import NGSFunctionBase
from .coef import NGSFunction, DomainWiseFunction
from .trials import TrialFunction, TestFunction, DifferentialSymbol
from .trials import TrialFunction as trial
from .trials import TestFunction as test
from .fields import VolumeField, SolutionFieldFunction, RandomFieldFunction
from .functions import det, inv, diag, offdiag, grad, rot, sin, cos, tan, step, min, max, exp, sqrt, norm, einsum, prev
from .space import FunctionSpace, H1, L2, FiniteElementSpace
from .solver import Solver, ConvergenceError

def eval(expr, dic={}):
    return builtins.eval(expr, globals(), dic)

t = NGSFunction(ngsolve.Parameter(0), name="t", tdep=True)
dti = NGSFunction(ngsolve.Parameter(-1), name="dti", tdep=True)
stepn = NGSFunction(ngsolve.Parameter(0), name = "step", tdep=True)
dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
