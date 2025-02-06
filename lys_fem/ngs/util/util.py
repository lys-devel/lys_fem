
import numpy as np
import sympy as sp
import ngsolve

from .operators import NGSFunctionBase


class NGSFunction(NGSFunctionBase):
    def __init__(self, obj=None, default=None, geomType="domain", name="Undefined", tdep=False):
        if obj is None or obj == 0:
            self._obj = None
            self._name = "0"
        elif isinstance(obj, (int, float, complex, sp.Integer, sp.Float)):
            self._obj = ngsolve.CoefficientFunction(obj)
            if name == "Undefined":
                self._name = str(obj)
            else:
                self._name = name
            self._tdep = False
        elif isinstance(obj, (list, tuple, np.ndarray)):
            self._obj = [value if isinstance(value, NGSFunctionBase) else NGSFunction(value) for value in obj]
            self._name = name
        elif isinstance(obj, dict):
            self._obj = {key: value if isinstance(value, NGSFunctionBase) else NGSFunction(value) for key, value in obj.items()}
            self._default = default
            self._geom = geomType
            self._name = name
        else:
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
        elif isinstance(self._obj, dict):
            return list(self._obj.values())[0].shape
        else:
            raise RuntimeError("error")

    @property
    def valid(self):
        if self._obj is None:
            return False
        elif isinstance(self._obj, ngsolve.CoefficientFunction):
            return True
        elif isinstance(self._obj, list):
            return any([obj.valid for obj in self._obj])
        elif isinstance(self._obj, dict):
            return any([obj.valid for obj in self._obj.values()])
        else:
            raise RuntimeError("error")

    def eval(self, fes):
        if self._obj is None:
            return ngsolve.CoefficientFunction(0)
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            return self._obj
        elif isinstance(self._obj, list):
            return ngsolve.CoefficientFunction(tuple([obj.eval(fes) for obj in self._obj]), dims=self.shape)
        elif isinstance(self._obj, dict):
            coefs = {key: obj.eval(fes) for key, obj in self._obj.items()}
            if self._default is None:
                default = None
            else:
                default = self._default.eval(fes)
            if self._geom=="domain":
                return fes.mesh.MaterialCF(coefs, default=default)
            else:
                return fes.mesh.BoundaryCF(coefs, default=default)

            
    def grad(self, fes):
        if self._obj is None:
            return ngsolve.CoefficientFunction(tuple([0]*fes.dimension))
        if isinstance(self._obj, list):
            return ngsolve.CoefficientFunction(tuple([obj.grad(fes) for obj in self._obj]), dims=self.shape+(fes.dimension,)).TensorTranspose((1,0))
        if isinstance(self._obj, dict):
            coefs = {key: obj.grad(fes) for key, obj in self._obj.items()}
            if self._default is None:
                default = None
            else:
                default = self._default.grad(fes)
            if self._geom=="domain":
                return fes.mesh.MaterialCF(coefs, default=default)
            else:
                return fes.mesh.BoundaryCF(coefs, default=default)
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            g = [self._obj.Diff(symbol) for symbol in [ngsolve.x, ngsolve.y, ngsolve.z][:fes.dimension]]
            return ngsolve.CoefficientFunction(tuple(g), dims=(fes.dimension,))
        raise RuntimeError("grad not implemented")
    
    def replace(self, d):
        if self._obj is None:
            return self
        elif isinstance(self._obj, list):
            return NGSFunction([obj.replace(d) for obj in self._obj], name=self._name)
        elif isinstance(self._obj, dict):
            if self._default is None:
                default = None
            else:
                default = self._default.replace(d)
            return NGSFunction({key: value.replace(d) for key, value in self._obj.items()}, name=self._name, default=default, geomType=self._geom)
        else:
            return d.get(self, self)

    def __contains__(self, item):
        if self._obj is None:
            return False
        elif isinstance(self._obj, list):
            return any([item in obj for obj in self._obj])
        elif isinstance(self._obj, dict):
            default = False if self._default is None else item in self._default
            return any([item in obj for obj in self._obj.values()]+[default])
        else:
            return False

    @property
    def hasTrial(self):
        if self._obj is None:
            return False
        elif isinstance(self._obj, ngsolve.CoefficientFunction):
            return False
        elif isinstance(self._obj, list):
            return any([obj.hasTrial for obj in self._obj])
        elif isinstance(self._obj, dict):
            return any([obj.hasTrial for obj in self._obj.values()])
        else:
            raise RuntimeError("error")
    
    @property
    def rhs(self):
        if not self.hasTrial:
            return self
        else:
            if isinstance(self._obj, ngsolve.CoefficientFunction):
                return self
            elif isinstance(self._obj, list):
                return NGSFunction([obj.rhs for obj in self._obj])
            elif isinstance(self._obj, dict):
                return NGSFunction({key: obj.rhs for key, obj in self._obj.items()})
            else:
                raise RuntimeError("error")

    @property
    def lhs(self):
        if self.hasTrial:
            if isinstance(self._obj, ngsolve.CoefficientFunction):
                return self
            elif isinstance(self._obj, list):
                return NGSFunction([obj.lhs for obj in self._obj])
            elif isinstance(self._obj, dict):
                return NGSFunction({key: obj.lhs for key, obj in self._obj.items()})
            else:
                raise RuntimeError("error")
        else:
            return NGSFunction()
        
    @property
    def isNonlinear(self):
        if self._obj is None:
            return False
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            return False
        elif isinstance(self._obj, list):
            return any([obj.isNonlinear for obj in self._obj])
        elif isinstance(self._obj, dict):
            return any([obj.isNonlinear for obj in self._obj.values()])
        else:
            raise RuntimeError("error")
        
    @property
    def isTimeDependent(self):
        if self._obj is None:
            return False
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            return self._tdep
        elif isinstance(self._obj, list):
            return any([obj.isTimeDependent for obj in self._obj])
        elif isinstance(self._obj, dict):
            return any([obj.isTimeDependent for obj in self._obj.values()])
        else:
            raise RuntimeError("error")

    def __str__(self):
        return self._name


class RandomFieldFunction(NGSFunction):
    """
    NGSFunction that provide the access to the present solution.
    Args:
        name(str): The symbol name
        type('L2' or 'H1'): The finite element space type.
        tdep(bool): Whether the field is time dependent
    """
    def __init__(self, type, shape, tdep, name=None):
        self._name = name
        self._shape = shape
        self._type = type
        self._tdep = tdep
        self._init = False

    def _initialize(self, fes):
        if self._type=="L2":
            sp = ngsolve.L2(fes.mesh, order=0)
        if self._type=="H1":
            sp = ngsolve.H1(fes.mesh, order=0)
        if self._shape == ():
            self._func = ngsolve.GridFunction(sp)
        else:
            size = 1
            for s in self._shape:
                size *= s
            self._func = [ngsolve.GridFunction(sp) for _ in range(size)]
        self._init = True
        self.update()

    @property
    def shape(self):
        return self._shape

    @property
    def valid(self):
        return True
    
    def update(self):
        if not self._init:
            raise RuntimeWarning("The random field is not initialized. Please delete unused field.")
        if self._shape == ():
            self._func.vec.data = np.random.normal(size=len(self._func.vec))
        else:
            for f in self._func:
                f.vec.data = np.random.normal(size=len(f.vec))

    def eval(self, fes):
        if not self._init:
            self._initialize(fes)
        if self._shape == ():
            return self._func
        else:
            return ngsolve.CoefficientFunction(tuple(self._func), dims=self._shape)
            
    def grad(self, fes):
        if not self._init:
            self._initialize(fes)
        if self._shape == ():
            return ngsolve.grad(self._func)
        else:
            raise RuntimeError("grad for multi-dim random is not defined.")
            
    def replace(self, d):
        return d.get(self, self)

    def __contains__(self, item):
        return self == item

    @property
    def hasTrial(self):
        return False
    
    @property
    def rhs(self):
        return self

    @property
    def lhs(self):
        return NGSFunction()
        
    @property
    def isNonlinear(self):
        return False
        
    @property
    def isTimeDependent(self):
        return self._tdep

    def __str__(self):
        return self._name


class VolumeField(NGSFunction):
    """
    NGSFunction that provide the access to the element wise volume.
    Args:
        name(str): The symbol name
    """
    def __init__(self, name=None):
        self._name = name
        self._init = False

    def _initialize(self, fes):
        sp = ngsolve.L2(fes.mesh)
        self._func = ngsolve.GridFunction(sp)
        data = ngsolve.Integrate(ngsolve.CoefficientFunction(1), fes.mesh, element_wise=True)
        self._func.vec.data = np.array(data)
        self._init = True

    @property
    def shape(self):
        return ()

    @property
    def valid(self):
        return True
    
    def eval(self, fes):
        if not self._init:
            self._initialize(fes)
        return self._func
            
    def grad(self, fes):
        raise RuntimeError("Gradient of the element-wise volume is not defined.")
            
    def replace(self, d):
        return d.get(self, self)

    def __contains__(self, item):
        return self == item

    @property
    def hasTrial(self):
        return False
    
    @property
    def rhs(self):
        return self

    @property
    def lhs(self):
        return NGSFunction()
        
    @property
    def isNonlinear(self):
        return False
        
    @property
    def isTimeDependent(self):
        return False

    def __str__(self):
        return self._name
     

class Parameter(NGSFunction):
    def __init__(self, name, value, tdep=False):
        super().__init__(ngsolve.Parameter(value), name=name, tdep=tdep)

    def set(self, value):
        self._obj.Set(value)

    def get(self):
        return self._obj.Get()

    @property
    def tdep(self):
        return self._tdep

    @tdep.setter
    def tdep(self, b):
        self._tdep = b

    @property
    def isNonlinear(self):
        return False


class SolutionFieldFunction(NGSFunction):
    pass


t = Parameter("t", 0, tdep=True)
stepn = Parameter("step", 0, tdep=True)
