import builtins
import ngsolve

from .util import NGSFunction, DomainWiseFunction
from .trials import TrialFunction, TestFunction, SolutionFunction, DifferentialSymbol
from .fields import VolumeField, SolutionFieldFunction, RandomFieldFunction
from .functions import det, inv, diag, offdiag, grad, sin, cos, tan, step, min, max, exp, sqrt, norm
from .space import FunctionSpace, H1, L2, NGSVariable, FiniteElementSpace

def eval(expr, dic={}):
    return builtins.eval(expr, globals(), dic)

t = NGSFunction(ngsolve.Parameter(0), name="t", tdep=True)
dti = NGSFunction(ngsolve.Parameter(-1), name="dti", tdep=True)
stepn = NGSFunction(ngsolve.Parameter(0), name = "step", tdep=True)
dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
dimension = 3