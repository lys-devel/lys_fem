import builtins
import ngsolve
import numpy as np

from .mesh import Mesh
from .operators import NGSFunctionBase
from .coef import NGSFunction, DomainWiseFunction
from .trials import TrialFunction, TestFunction, DifferentialSymbol
from .trials import TrialFunction as trial
from .trials import TestFunction as test
from .fields import GridField, VolumeField, SolutionFieldFunction, RandomFieldFunction
from .functions import det, inv, diag, offdiag, grad, rot, sin, cos, tan, step, min, max, exp, sqrt, norm, einsum, prev
from .space import FunctionSpace, H1, L2, FiniteElementSpace
from .solver import Solver, ConvergenceError
from .consts import x, y, z, c, e, pi, k_B, g_e, mu_0, eps_0, mu_B, Ve, t, dti,stepn

def eval(expr, dic={}, name=None, geom=None, J=None):
    """
    Evaluate the expression expr using ngsolve wrapper.

    If the expr is string or number, it will be automatically translated to scalar NGSFunction.
    If the expr is iterable, it will be translated to vector (or matrix) NGSFunction.
    If the expr is dict, it will be translated to DomainWiseFunction. In this case, geom parameter is required. The key of the dictionary should be integer.

    All constants and functions can be used in the expr.

    Args:
        expr (int, str, iterable, or dict): The expression.
        dic (dict): The dictionary of NGSFunctions used for translation.
        name(str): The name of the resulting NGSFunction
        geom("domain" or "boundary"): The geometry type.
        J(str): The expression for the jacobian, which will be applied to the result.
    """
    if expr is None:
        return None
    elif isinstance(expr, (list, tuple, np.ndarray)):
        return NGSFunction([eval(c, dic=dic) for c in expr], name=name if name is not None else str(expr), J=J)
    elif isinstance(expr, (int, float, complex)):
        return NGSFunction(expr, name=name if name is not None else str(expr))
    if isinstance(expr, dict):
        if geom is None:
            raise ValueError("Geometry type is required for dictionary input.")
        coefs = {geom+str(key): eval(value, dic=dic) for key, value in expr.items() if key != "default"}
        if "default" in expr:
            coefs["default"] = eval(expr["default"], dic=dic)
        return DomainWiseFunction(coefs, geomType=geom, name=name, J=J)
    else:
        return builtins.eval(expr, globals(), dic)

dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
