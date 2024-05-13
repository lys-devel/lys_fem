import itertools
import sympy as sp
import numpy as np

import ngsolve
from ngsolve import x,y,z, CoefficientFunction
from ngsolve.fem import Einsum

from lys_fem.fem import FEMCoefficient
from ..models.common import DirichletBoundary

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res


def cross(u,v):
    return [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]


def dot(u,v):
    res = u[0] * v[0]
    if len(u) > 1:
        res += u[1]*v[1]
    if len(u) > 2:
        res += u[2]*v[2]
    return res


def generateGeometry(region):
    return  "|".join([region.geometryType.lower() + str(r) for r in region])


def generateDirichletCondition(model):
    conditions = model.boundaryConditions.get(DirichletBoundary)
    bdr_dir = {i: [] for i in range(model.variableDimension())}
    for b in conditions:
        for axis, check in enumerate(b.values):
            if check:
                bdr_dir[axis].extend(b.geometries.getSelection())
    return list(bdr_dir.values())


def generateCoefficient(coef, mesh=None, geom="domain", **kwargs):
    if isinstance(coef, FEMCoefficient):
        geom = coef.geometryType.lower()
        coefs = {geom+str(key): generateCoefficient(value) for key, value in coef.items()}
        if geom=="domain":
            return mesh.MaterialCF(coefs, **kwargs)
        else:
            return mesh.BoundaryCF(coefs, **kwargs)
    elif isinstance(coef, (list, tuple, np.ndarray)):
        return CoefficientFunction(tuple([generateCoefficient(c) for c in coef]), dims=np.shape(coef))
    elif isinstance(coef, (int, float, sp.Integer, sp.Float)):
        return CoefficientFunction(coef)
    elif isinstance(coef, CoefficientFunction):
        return coef
    else:
        res = sp.lambdify(sp.symbols("x_scaled,y_scaled,z_scaled"), coef, modules=[{"abs": _absolute}, ngsolve])(x,y,z)
        return res


def _absolute(x):
    return x.Norm()


def coef(coef, mesh=None, geom="domain", name=None, **kwargs):
    if name is None:
        name = str(coef)
    obj = generateCoefficient(coef, mesh, geom, **kwargs)
    return NGSFunction(obj, name=name)


def grad(f):
    return f.grad


class NGSFunction:
    def __init__(self, obj=None, name="Undefined"):
        if obj is None:
            self._obj = None
            name = "Zero"
        else:
            self._obj = obj
        self._name = name

    @property
    def shape(self):
        if hasattr(self._obj, "shape"):
            return self._obj.shape
        else:
            return ()
    
    def __mul__(self, other):
        if self.valid and other.valid:
            return _Mul(self, other)
        else:
            return NGSFunction()

    def __truediv__(self, other):
        if self.valid and other.valid:
            return _Mul(self, other, "/")
        else:
            return NGSFunction()
        
    def __add__(self, other):
        if not self.valid:
            return other
        elif not other.valid:
            return self
        else:
            return _Add(self, other)

    def __sub__(self, other):
        if not self.valid:
            return other
        elif not other.valid:
            return self
        else:
            return _Add(self, other, type="-")
        
    def dot(self, other):
        if self.valid and other.valid:
            return _TensorDot(self, other)
        else:
            return NGSFunction()
    
    def ddot(self, other):
        if self.valid and other.valid:
            return _TensorDot(self, other, axes=2)
        else:
            return NGSFunction()

    def __str__(self):
        return self._name
    
    @property
    def hasTrial(self):
        return False
    
    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            return NGSFunction()

    @property
    def lhs(self):
        if self.hasTrial:
            return self
        else:
            return NGSFunction()
        
    @property
    def valid(self):
        return self._obj is not None
        
    def eval(self):
        if not self.valid:
            return generateCoefficient(0)*ngsolve.dx
        res = _expand(self._obj)
        if isinstance(res, list):
            return np.array(res)
        else:
            return res


class _Oper(NGSFunction):
    def __init__(self, obj1, obj2=None):
        if not isinstance(obj1, NGSFunction):
            obj1 = NGSFunction(obj1, str(obj1))
        if not isinstance(obj2, NGSFunction) and obj2 is not None:
            obj2 = NGSFunction(obj2, str(obj2))
        super().__init__([obj1, obj2])

    def replace(self, d):
        for i in range(2):
            if isinstance(self._obj[i], _Oper):
                self._obj[i].replace(d)        
            else:
                if self._obj[i] in d:
                    self._obj[i] = d[self._obj[i]]       


class _Add(_Oper):
    def __init__(self, obj1, obj2,type="+"):
        super().__init__(obj1, obj2)
        self._type = type

    def __call__(self, v1, v2):
        if self._type == "+":
            return v1+v2
        else:
            return v1-v2

    @property
    def rhs(self):
        return self(self._obj[0].rhs, self._obj[1].rhs)

    @property
    def lhs(self):
        return self(self._obj[0].lhs, self._obj[1].lhs)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())
    
    def __str__(self):
        return "(" + str(self._obj[0]) + self._type + str(self._obj[1]) + ")"


class _Mul(_Oper):
    def __init__(self, obj1, obj2,type="*"):
        super().__init__(obj1, obj2)
        self._type = type

    def __call__(self, x, y):
        if self._type == "*":
            return x * y
        else:
            return x / y

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())

    def __str__(self):
        return "(" + str(self._obj[0]) + self._type + str(self._obj[1]) + ")"

    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            if self._obj[0].hasTrial:
                return self(self._obj[0].rhs, self._obj[1])
            else:
                return self(self._obj[0], self._obj[1].rhs)

    @property
    def lhs(self):
        if self.hasTrial:
            if self._obj[0].hasTrial:
                return self(self._obj[0].lhs, self._obj[1])
            else:
                return self(self._obj[0], self._obj[1].lhs)
        else:
            return NGSFunction()
    

class _TensorDot(_Oper):
    def __init__(self, obj1, obj2, axes=1):
        super().__init__(obj1, obj2)
        self._axes = axes

    def __call__(self, a, b):
        if self._axes == 1:
            return a.dot(b)
        else:
            return a.ddot(b)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        res = np.tensordot(self._obj[0].eval(), self._obj[1].eval(), axes=self._axes)
        if len(res.shape) == 0:
            return res.item()
        return res

    def __str__(self):
        if self._axes == 1:
            return "(" + str(self._obj[0]) + "." + str(self._obj[1]) + ")"
        else:
            return "(" + str(self._obj[0]) + ":" + str(self._obj[1]) + ")"

    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            if self._obj[0].hasTrial:
                return self(self._obj[0].rhs, self._obj[1])
            else:
                return self(self._obj[0], self._obj[1].rhs)

    @property
    def lhs(self):
        if self.hasTrial:
            if self._obj[0].hasTrial:
                return self(self._obj[0].lhs, self._obj[1])
            else:
                return self(self._obj[0], self._obj[1].lhs)
        else:
            return NGSFunction()


class TrialFunction(NGSFunction):
    def __init__(self, name, obj, dt=0, grad=False):
        super().__init__(obj, name="trial("+name+")")
        self._name = name
        self._dt = dt
        self._grad = grad

    @property
    def t(self):
        return TrialFunction(self._name, self._obj, self._dt+1, grad=self._grad)

    @property
    def tt(self):
        return TrialFunction(self._name, self._obj, self._dt+2, grad=self._grad)

    @property    
    def grad(self):
        return TrialFunction(self._name, self._obj, self._dt, grad=True)
    
    @property
    def value(self):
        return TrialFunction(self._name, self._obj, -1)
       
    def __hash__(self):
        return hash(self._name + "__" + str(self._dt) + "__" + str(self._grad))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __str__(self):
        name = self._name
        if self._dt == -1:
            return name+"0"
        for i in range(self._dt):
            name += "t" 
        if self._grad:
            name = "grad(" + name + ")"
        return name

    @property
    def hasTrial(self):
        return True
    
    def eval(self):
        if self._grad:
            res = _expand(_grad(self._obj))
            if isinstance(res, list):
                return np.array(res)
            else:
                return res
        else:
            return super().eval()


class TestFunction(NGSFunction):
    def __init__(self, obj, name, grad=False):
        super().__init__(obj, name="test("+name+")")
        self._grad = grad
        self._nam = name

    @property    
    def grad(self):
        return TestFunction(self._obj, "grad("+self._nam+")", grad=True)
    
    def eval(self):
        if self._grad:
            res = _expand(_grad(self._obj))
            if isinstance(res, list):
                return np.array(res)
            else:
                return res
        else:
            return super().eval()


def _grad(x):
    if isinstance(x, CoefficientFunction):
        return ngsolve.grad(x)
    else:
        return [_grad(y) for y in x]   

def _expand(x):
    if isinstance(x, ngsolve.comp.DifferentialSymbol):
        return x
    elif isinstance(x, CoefficientFunction):
        if len(x.shape) == 0:
            return x
        else:
            res = np.array([x[i] for i in itertools.product(*[range(s) for s in x.shape])])
            return res.reshape(*x.shape).tolist()
    else:
        return [_expand(y) for y in x]   


dx = NGSFunction(ngsolve.dx, name="dx")
ds = NGSFunction(ngsolve.ds, name="ds")
