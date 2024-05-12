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
            self._obj = 0
            self._name = "0"
            self._valid = False
        else:
            self._obj = obj
            self._name = name
            self._valid = True

    @property
    def shape(self):
        if hasattr(self._obj, "shape"):
            return self._obj.shape
        else:
            return ()
    
    def __mul__(self, other):
        if not isinstance(other, NGSFunction):
            other = NGSFunction(other, name=str(other))
        if self.valid and other.valid:
            return _Mul(self, other)
        else:
            return NGSFunction()

    def __add__(self, other):
        if not isinstance(other, NGSFunction):
            other = NGSFunction(other, name=str(other))
        if not self.valid:
            return other
        elif not other.valid:
            return self
        else:
            return _Add(self, other)

    def __sub__(self, other):
        if not isinstance(other, NGSFunction):
            other = NGSFunction(other, name=str(other))
        if not self.valid:
            return other
        elif not other.valid:
            return self
        else:
            return _Add(self, other, type="-")
        
    def dot(self, other):
        return self*other
    
    def ddot(self, other):
        if not isinstance(other, NGSFunction):
            other = NGSFunction(other, name=str(other))
        if self.valid and other.valid:
            return _DoubleDot(self, other)
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
        return self._valid
        
    def eval(self):
        return self._obj
        

class _Oper(NGSFunction):
    def __init__(self, obj1, obj2=None):
        self._valid = True
        if not isinstance(obj1, NGSFunction):
            obj1 = NGSFunction(obj1)
        if not isinstance(obj2, NGSFunction) and obj2 is not None:
            obj2 = NGSFunction(obj2)
        self._obj1 = obj1
        self._obj2 = obj2

    def replace(self, d):
        if isinstance(self._obj1, _Oper):
            self._obj1.replace(d)        
        else:
            if self._obj1 in d:
                self._obj1 = d[self._obj1]
        if isinstance(self._obj2, _Oper):
            self._obj2.replace(d)        
        else:
            if self._obj2 in d:
                self._obj2 = d[self._obj2]
        

class _Add(_Oper):
    def __init__(self, obj1, obj2,type="+"):
        super().__init__(obj1, obj2)
        self._type = type

    @property
    def rhs(self):
        if self._type == "+":
            return self._obj1.rhs + self._obj2.rhs
        else:
            return self._obj1.rhs - self._obj2.rhs

    @property
    def lhs(self):
        if self._type == "+":
            return self._obj1.lhs + self._obj2.lhs
        else:
            return self._obj1.lhs + self._obj2.lhs

    @property
    def hasTrial(self):
        return self._obj1.hasTrial or self._obj2.hasTrial

    def eval(self):
        if self._type == "+":
            return self._obj1.eval() + self._obj2.eval()
        else:
            return self._obj1.eval() - self._obj2.eval()
    
    def __str__(self):
        return "(" + str(self._obj1) + self._type + str(self._obj2) + ")"


class _Mul(_Oper):
    @property
    def hasTrial(self):
        return self._obj1.hasTrial or self._obj2.hasTrial

    def eval(self):
        return self._obj1.eval() * self._obj2.eval()

    def __str__(self):
        return "(" + str(self._obj1) + "*" + str(self._obj2) + ")"

    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            if self._obj1.hasTrial:
                return self._obj1.rhs * self._obj2
            else:
                return self._obj1 * self._obj2.rhs

    @property
    def lhs(self):
        if self.hasTrial:
            if self._obj1.hasTrial:
                return self._obj1.lhs * self._obj2
            else:
                return self._obj1 * self._obj2.lhs
        else:
            return NGSFunction()
    

class _DoubleDot(_Oper):
    @property
    def hasTrial(self):
        return self._obj1.hasTrial or self._obj2.hasTrial

    def eval(self):
        o1 = self._obj1.eval()
        o2 = self._obj2.eval()
        print(o1.shape, o2.shape)
        if len(o1.shape) == 4:
            res = Einsum("ijkl,kl->ij", o1, o2)
        elif len(o1.shape) == 2:
            res = Einsum("ij,ij", o1, o2)
        else:
            res = o1 * o2
        return res

    def __str__(self):
        return "(" + str(self._obj1) + ":" + str(self._obj2) + ")"

    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            if self._obj1.hasTrial:
                return self._obj1.rhs.ddot(self._obj2)
            else:
                return self._obj1.ddot(self._obj2.rhs)

    @property
    def lhs(self):
        if self.hasTrial:
            if self._obj1.hasTrial:
                return self._obj1.lhs.ddot(self._obj2)
            else:
                return self._obj1.ddot(self._obj2.lhs)
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
       
    def __hash__(self):
        return hash(self._name + "__" + str(self._dt) + "__" + str(self._grad))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __str__(self):
        name = self._name
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
            return ngsolve.grad(self._obj)
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
            return ngsolve.grad(self._obj)
        else:
            return super().eval()
    
dx = NGSFunction(ngsolve.dx, name="dx")
ds = NGSFunction(ngsolve.ds, name="ds")
