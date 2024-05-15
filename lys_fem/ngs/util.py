import sympy as sp
import numpy as np

import ngsolve

from lys_fem.fem import FEMCoefficient
from ..models.common import DirichletBoundary

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
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
        return ngsolve.CoefficientFunction(tuple([generateCoefficient(c) for c in coef]), dims=np.shape(coef))
    elif isinstance(coef, (int, float, sp.Integer, sp.Float)):
        return ngsolve.CoefficientFunction(coef)
    elif isinstance(coef, ngsolve.CoefficientFunction):
        return coef
    else:
        def _absolute(x):
            return x.Norm()
        res = sp.lambdify(sp.symbols("x_scaled,y_scaled,z_scaled"), coef, modules=[{"abs": _absolute}, ngsolve])(ngsolve.x,ngsolve.y,ngsolve.z)
        return res


def coef(coef, mesh=None, geom="domain", name=None, default=None, **kwargs):
    if coef == 0:
        return NGSFunction()
    if name is None:
        name = str(coef)
    if default is not None and not isinstance(default, ngsolve.CoefficientFunction):
        default = generateCoefficient(default)
    obj = generateCoefficient(coef, mesh, geom, default=default, **kwargs)
    return NGSFunction(obj, name=name)


def grad(f):
    return f.grad


class GridFunction(ngsolve.GridFunction):
    def __init__(self, fes, value=None):
        super().__init__(fes)
        if value is not None:
            self.set(value)

    def set(self, value):
        if self.isSingle:
            self.Set(*value)
        else:
            for ui, i in zip(self.components, value):
                ui.Set(i)

    @property
    def components(self):
        if self.isSingle:
            return [self]
        else:
            return super().components

    @property
    def isSingle(self):
        return not isinstance(self.space, ngsolve.ProductSpace)


class NGSFunction:
    def __init__(self, obj=None, name="Undefined"):
        if obj is None:
            self._obj = None
            self._name = "Zero"
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
        if isinstance(other, (int, float, complex)):
            other = coef(other)
        if self.valid and other.valid:
            return _Mul(self, other)
        else:
            return NGSFunction()

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            other = coef(other)
        if self.valid and other.valid:
            return _Mul(self, other, "/")
        else:
            return NGSFunction()
        
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = coef(other)
        if not self.valid:
            return other
        elif not other.valid:
            return self
        else:
            return _Add(self, other)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = coef(other)
        if not self.valid:
            return coef(-1)*other
        elif not other.valid:
            return self
        else:
            return _Add(self, other, type="-")
        
    def __neg__(self):
        if not self.valid:
            return NGSFunction()
        else:
            return self*(-1)

    def __rmul__(self, other):
        return self * other
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return (-self) + other 

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

    def cross(self, other):
        if self.valid and other.valid:
            return _Cross(self, other)
        else:
            return NGSFunction()
        
    def det(self):
        J = self.eval()
        if J.shape[0] == 3:
            return NGSFunction(J[0,0]*J[1,1]*J[2,2] + J[0,1]*J[1,2]*J[2,0] + J[0,2]*J[1,0]*J[2,1] - J[0,2]*J[1,1]*J[2,0] - J[0,1]*J[1,0]*J[2,2] - J[0,0]*J[1,2]*J[2,1], "|"+self._name+"|")
        elif J.shape[0] == 2:
            return NGSFunction(J[0,0]*J[1,1] - J[0,1]*J[1,0]*J[2,2], "|"+self._name+"|")
        elif J.shape[0] == 1:
            return NGSFunction(J[0,0], "|"+self._name+"|")

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
        if self.valid:
            return self._obj
        else:
            return generateCoefficient(0)*ngsolve.dx


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
        return self


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
            if isinstance(y, np.ndarray):
                return y * x
            return x * y
        else:
            return x / y

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())

    def __str__(self):
        if isinstance(self._obj[0], _Mul) or isinstance(self._obj[1], _Mul):
            return str(self._obj[0]) + self._type + str(self._obj[1])
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
        if self._axes == 1:
            return self._obj[0].eval() * self._obj[1].eval()
        else:
            v1, v2 = self._obj[0].eval(), self._obj[1].eval()
            sym1, sym2 = "abcdef"[0:len(v1.shape)-2]+"ij", "ij"+"mnopqr"[0:len(v2.shape)-2]
            expr = sym1+","+sym2+"->"+sym1[0:-2]+sym2[2:]
            return ngsolve.fem.Einsum(expr, v1, v2)

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



class _Cross(_Oper):
    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        # Levi Civita symbol
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        eijk = generateCoefficient(eijk.tolist())

        v1, v2 = self._obj[0].eval(), self._obj[1].eval()
        # Create expression
        sym1, sym2 = "abcdefgh"[:len(v1.shape)], "nmpqrs"[:len(v2.shape)]
        expr = "i"+sym1[-1]+sym2[0]+","+sym1+","+sym2+"->"+sym1[:-1]+"i"+sym2[1:]

        # Calculate by einsum
        return ngsolve.fem.Einsum(expr, eijk, v1, v2)

    def __str__(self):
        return "(" + str(self._obj[0]) + " x " + str(self._obj[1]) + ")"

    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            if self._obj[0].hasTrial:
                return self._obj[0].rhs.cross(self._obj[1])
            else:
                return self._obj[0].cross(self._obj[1].rhs)

    @property
    def lhs(self):
        if self.hasTrial:
            if self._obj[0].hasTrial:
                return self._obj[0].lhs.cross(self._obj[1])
            else:
                return self._obj[0].cross(self._obj[1].lhs)
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
        return NGSFunction(self.eval(), self._name)
       
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
            if isinstance(self._obj, list):
                return ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._obj]), dims=(ngsolve.grad(self._obj[0]).shape[0], len(self._obj))).TensorTranspose((1,0))
            return ngsolve.grad(self._obj)
        else:
            if isinstance(self._obj, list):
                return ngsolve.CoefficientFunction(tuple([t for t in self._obj]))
            return self._obj


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
            if isinstance(self._obj, list):
                return ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._obj]), dims=(ngsolve.grad(self._obj[0]).shape[0], len(self._obj))).TensorTranspose((1,0))
            return ngsolve.grad(self._obj)
        else:
            if isinstance(self._obj, list):
                return ngsolve.CoefficientFunction(tuple([t for t in self._obj]))
            return super().eval()


class DifferentialSymbol(NGSFunction):
    pass


dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
