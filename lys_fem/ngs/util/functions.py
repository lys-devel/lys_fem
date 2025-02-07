import builtins
import numpy as np
import ngsolve

from .operators import NGSFunctionBase
from .coef import NGSFunction
from .trials import TrialFunction, TestFunction


def grad(f):
    """
    Calculate gradient of a function f.
    The output shape is in the form of (dimension, original shpae). 
    
    """
    return _Func(f, "grad")


def exp(x):
    return _Func(x, "exp")


def sin(x):
    return _Func(x, "sin")


def cos(x):
    return _Func(x, "cos")


def tan(x):
    return _Func(x, "tan")


def step(x):
    return _Func(x, "step")


def sqrt(x):
    return _Func(x, "sqrt")


def norm(x):
    return _Func(x, "norm")


def min(x,y):
    return _MinMax(x, y, "min")


def max(x,y):
    return _MinMax(x, y, "max")


def diag(m):
    M = np.zeros(m.shape, dtype=object)
    for i in range(builtins.min(*m.shape)):
        M[i,i] = m[i,i]
    return NGSFunction(M.tolist())


def offdiag(m):
    M = np.zeros(m.shape, dtype=object)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i != j:
                M[i,j] = m[i,j]
    return NGSFunction(M.tolist())


def det(J):
    if J.shape[0] == 3:
        return J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1]
    elif J.shape[0] == 2:
        return J[0,0]*J[1,1] - J[0,1]*J[1,0]
    elif J.shape[0] == 1:
        return J[0,0]
    
def inv(m):
    if m.shape[0] == 3:
        return NGSFunction([
            [m[1,1]*m[2,2]-m[1,2]*m[2,1], m[0,2]*m[2,1]-m[0,1]*m[2,2], m[0,1]*m[1,2]-m[0,2]*m[1,1]],
            [m[1,2]*m[2,0]-m[1,0]*m[2,2], m[0,0]*m[2,2]-m[0,2]*m[2,0], m[0,2]*m[1,0]-m[0,0]*m[1,2]],
            [m[1,0]*m[2,1]-m[1,1]*m[2,0], m[0,1]*m[2,0]-m[0,0]*m[2,1], m[0,0]*m[1,1]-m[0,1]*m[1,0]]])/m.det()
    elif m.shape[0] == 2:
        return NGSFunction([[m[1,1], -m[0,1]], [-m[1,0], m[0,0]]])/m.det()
    else:
        return 1/m


class _Func(NGSFunctionBase):
    def __init__(self, obj, type):
        if not isinstance(obj, NGSFunctionBase):
            obj = NGSFunction(obj, str(obj))
        self._obj = obj
        self._type = type

    def replace(self, d):
        if self in d:
            return d.get(self)
        replaced = self._obj.replace(d)
        if replaced == 0:
            replaced = NGSFunction()
        return self(replaced)
    
    def __contains__(self, item):
        return item in self._obj
    
    @property
    def isTimeDependent(self):
        return self._obj.isTimeDependent

    def __call__(self, obj1):
        return _Func(obj1, self._type)
    
    @property
    def valid(self):
        return True

    @property
    def rhs(self):
        if self.hasTrial:
            return NGSFunction()
        else:
            return self

    @property
    def lhs(self):
        if self.hasTrial:
            return self
        else:
            return NGSFunction()

    @property
    def hasTrial(self):
        return self._obj.hasTrial

    @property
    def isNonlinear(self):
        if self._type == "grad":
            return self._obj.isNonlinear
        else:
            return self._obj.hasTrial

    @property
    def shape(self):
        if self._type == "grad":
            return tuple([3]+list(self._obj.shape))
        else:
            return self._obj.shape

    def __hash__(self):
        if self._type == "grad" and isinstance(self._obj, (TrialFunction, TestFunction)):
            return hash(str(self._obj)+"__grad")
        return super().__hash__()
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def eval(self, fes):
        if self._type == "exp":
            return ngsolve.exp(self._obj.eval(fes))
        if self._type == "sin":
            return ngsolve.sin(self._obj.eval(fes))
        if self._type == "cos":
            return ngsolve.cos(self._obj.eval(fes))
        if self._type == "tan":
            return ngsolve.tan(self._obj.eval(fes))
        if self._type == "step":
            return ngsolve.IfPos(self._obj.eval(fes), 1, 0)
        if self._type == "sqrt":
            return ngsolve.sqrt(self._obj.eval(fes))    
        if self._type == "norm":
            v = self._obj.eval(fes)
            return ngsolve.sqrt(v*v)
        if self._type == "grad":
            if hasattr(self._obj, "grad"):
                x = self._obj.grad(fes)
                if fes.J is None:
                    return x
                else:
                    return fes.J.eval(fes)*x
                
            raise RuntimeError("grad is not implemented for " + str(type(self._obj)))
    
    def __str__(self):
        return self._type + "(" + str(self._obj) + ")"


class _MinMax(_Func):
    def __init__(self, obj1, obj2, type="min"):
        if not isinstance(obj1, NGSFunctionBase):
            obj1 = NGSFunction(obj1, str(obj1))
        if not isinstance(obj2, NGSFunctionBase):
            obj2 = NGSFunction(obj2, str(obj2))

        self._obj = [obj1, obj2]
        self._type = type

    def __call__(self, obj1, obj2):
        if isinstance(obj1, (int,float)) and isinstance(obj2, (int,float)):
            if self._type == "min":
                return builtins.min([obj1, obj2])
            else:
                return builtins.max([obj1, obj2])
        return _MinMax(obj1, obj2, self._type)

    @property
    def shape(self):
        return self._obj[0].shape

    @property
    def rhs(self):
        return self(self._obj[0].rhs, self._obj[1].rhs)

    @property
    def lhs(self):
        return self(self._obj[0].lhs, self._obj[1].lhs)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    @property
    def isNonlinear(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    @property
    def isTimeDependent(self):
        return self._obj[0].isTimeDependent or self._obj[1].isTimeDependent

    def replace(self, d):
        if self in d:
            return d.get(self)
        objs = []
        for i in range(2):
            obj = self._obj[i]
            replaced = obj.replace(d)
            if replaced == 0:
                replaced = NGSFunction()
            objs.append(replaced)
        return self(*objs)
    
    def eval(self, fes):
        e1, e2 = self._obj[0].eval(fes), self._obj[1].eval(fes)
        if self._type == "max":
            return ngsolve.IfPos(e1-e2, e1, e2)
        else:
            return ngsolve.IfPos(e1-e2, e2, e1)
    
    def __str__(self):
        return self._type + "(" + str(self._obj[0]) + ", " + str(self._obj[1]) + ")"
