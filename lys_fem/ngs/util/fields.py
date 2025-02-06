import numpy as np
import ngsolve

from .operators import NGSFunctionBase
from .util import NGSFunction


class RandomFieldFunction(NGSFunctionBase):
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


class VolumeField(NGSFunctionBase):
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
    

class SolutionFieldFunction(NGSFunction):
    pass