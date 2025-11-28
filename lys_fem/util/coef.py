
import numpy as np
import sympy as sp
import ngsolve

from .operators import NGSFunctionBase


class NGSFunction(NGSFunctionBase):
    def __init__(self, obj=None, name="Undefined", tdep=False, J=None):
        self._J = J
        if obj is None or obj == 0:
            self._obj = None
            self._name = "0"
            self._tdep = False
            return
        if isinstance(obj, (int, float, complex, sp.Integer, sp.Float)):
            if name == "Undefined":
                name = str(obj)
            obj = ngsolve.CoefficientFunction(obj)
        if isinstance(obj, (list, tuple, np.ndarray)):
            self._obj = [value if isinstance(value, NGSFunctionBase) else NGSFunction(value) for value in obj]
            self._name = name
        elif isinstance(obj, ngsolve.CoefficientFunction):
            self._obj = obj
            self._name = name
            self._tdep = tdep

    @property
    def shape(self):
        if self._obj is None:
            return ()
        elif isinstance(self._obj, ngsolve.CoefficientFunction):
            return self._obj.shape
        elif isinstance(self._obj, list):
            return tuple([len(self._obj)] + list(self._obj[0].shape))
        else:
            raise RuntimeError("error")

    @property
    def valid(self):
        if isinstance(self._obj, list):
            return any([obj.valid for obj in self._obj])
        else:
            return self._obj is not None

    def eval(self, fes):
        if self._obj is None:
            return ngsolve.CoefficientFunction(0)
        elif isinstance(self._obj, ngsolve.CoefficientFunction):
            res = self._obj
        elif isinstance(self._obj, list):
            res = ngsolve.CoefficientFunction(tuple([obj.eval(fes) for obj in self._obj]), dims=self.shape)
        if self._J is not None and len(res.shape) > 0:
            res = applyJacobian(res, fes.jacobi(self._J))
        return res

    def grad(self, fes):
        if self._obj is None:
            return ngsolve.CoefficientFunction(tuple([0]*fes.dimension))
        if isinstance(self._obj, list):
            return ngsolve.CoefficientFunction(tuple([obj.grad(fes) for obj in self._obj]), dims=self.shape+(fes.dimension,)).TensorTranspose((1,0))
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            g = [self._obj.Diff(symbol) for symbol in [ngsolve.x, ngsolve.y, ngsolve.z]]
            return ngsolve.CoefficientFunction(tuple(g), dims=(3,))
        raise RuntimeError("grad not implemented")
    
    def replace(self, d):
        if isinstance(self._obj, list):
            return NGSFunction([obj.replace(d) for obj in self._obj], name=self._name)
        else:
            return d.get(self, self)

    def __contains__(self, item):
        if isinstance(self._obj, list):
            return any([item in obj for obj in self._obj])
        else:
            return False

    @property
    def hasTrial(self):
        if isinstance(self._obj, list):
            return any([obj.hasTrial for obj in self._obj])
        else:
            return False
    
    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        if isinstance(self._obj, list):
            return NGSFunction([obj.rhs for obj in self._obj])
        else:
            return self

    @property
    def lhs(self):
        if not self.hasTrial:
            return NGSFunction()
        if isinstance(self._obj, list):
            return NGSFunction([obj.lhs for obj in self._obj])
        else:
            return self
        
    @property
    def isNonlinear(self):
        if isinstance(self._obj, list):
            return any([obj.isNonlinear for obj in self._obj])
        else:
            return False
        
    @property
    def isTimeDependent(self):
        if isinstance(self._obj, list):
            return any([obj.isTimeDependent for obj in self._obj])
        else:
            return self._tdep
        
    @isTimeDependent.setter
    def isTimeDependent(self, b):
        self._tdep = b

    def __str__(self):
        return self._name

    def set(self, value):
        if isinstance(self._obj, ngsolve.Parameter):
            self._obj.Set(value)
        else:
            raise RuntimeError("Cannot set coefficient function except parameter.")


class DomainWiseFunction(NGSFunctionBase):
    def __init__(self, obj=None, default=None, geomType="domain", name="Undefined", J=None):
        self._obj = {key: value if isinstance(value, NGSFunctionBase) else NGSFunction(value) for key, value in obj.items()}
        self._default = default
        self._geom = geomType
        self._name = name
        self._J = J

    @property
    def shape(self):
        return list(self._obj.values())[0].shape

    @property
    def valid(self):
        return any([obj.valid for obj in self._obj.values()])

    def eval(self, fes):
        coefs = {key: obj.eval(fes) for key, obj in self._obj.items()}
        if self._default is None:
            default = None
        else:
            default = self._default.eval(fes)

        if self._geom=="domain":
            res = fes.mesh.MaterialCF(coefs, default=default)
        else:
            res = fes.mesh.BoundaryCF(coefs, default=default)

        if self._J is not None:
            res = applyJacobian(res, fes.jacobi(self._J))
        return res

    def grad(self, fes):
        coefs = {key: obj.grad(fes) for key, obj in self._obj.items()}
        if self._default is None:
            default = None
        else:
            default = self._default.grad(fes)
        if self._geom=="domain":
            return fes.mesh.MaterialCF(coefs, default=default)
        else:
            return fes.mesh.BoundaryCF(coefs, default=default)
    
    def replace(self, d):
        if self._default is None:
            default = None
        else:
            default = self._default.replace(d)
        return DomainWiseFunction({key: value.replace(d) for key, value in self._obj.items()}, name=self._name, default=default, geomType=self._geom, J=self._J)

    def __contains__(self, item):
        default = False if self._default is None else item in self._default
        return any([item in obj for obj in self._obj.values()]+[default])

    @property
    def hasTrial(self):
        return any([obj.hasTrial for obj in self._obj.values()])
    
    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            return DomainWiseFunction({key: obj.rhs for key, obj in self._obj.items()})

    @property
    def lhs(self):
        if self.hasTrial:
            return DomainWiseFunction({key: obj.lhs for key, obj in self._obj.items()})
        else:
            return NGSFunction()
        
    @property
    def isNonlinear(self):
        return any([obj.isNonlinear for obj in self._obj.values()])
        
    @property
    def isTimeDependent(self):
        return any([obj.isTimeDependent for obj in self._obj.values()])
        
    def __str__(self):
        return self._name
 
def applyJacobian(A, J):
    if len(A.shape) == 0 or J is None:
        return A
    st1, st2 = "ijklmnopqrt"[:len(A.shape)], "IJKLMNOPQRT"[:len(A.shape)]
    st3 = ",".join(s1+s2 for s1, s2 in zip(st1, st2))
    Js = [J] * len(A.shape)
    return ngsolve.fem.Einsum(st1+","+st3+"->"+st2, A, *Js)
