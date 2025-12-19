import numpy as np
import ngsolve

from .operators import NGSFunctionBase
from .coef import NGSFunction
from .functions import grad
from .space import FiniteElementSpace, L2


class GridField(NGSFunctionBase):
    """
    NGSFunction that provide the access to a component of the grid function.
    Args:
        grid(GridFunction): The original grid function
        var(FunctionSpace): The symbol to be used.
    """
    def __init__(self, grid, var):
        self._grid = grid
        self._var = var
        n = 0
        for v in grid.finiteElementSpace.variables:
            if v == var:
                self._n = n
            n += v.size

    def replace(self, d):
        return d.get(self, self)
    
    def eval(self, fes):
        if self._var.isScalar:
            return self._value()
        else:
            value = np.array(self._value(), dtype=object)
            pad_width = [(0, t - s) for s, t in zip(value.shape, self.shape)]
            value = np.pad(value, pad_width, mode="constant").flatten().tolist()
            return ngsolve.CoefficientFunction(tuple(value), dims=self.shape)
        
    def grad(self, fes):
        if self._var.isScalar:
            return self._grad(fes, self._value())
        else:
            value = np.array(self._value(), dtype=object)
            pad_width = [(0, t - s) for s, t in zip(value.shape, self.shape)]
            value = np.pad(value, pad_width, mode="constant").flatten().tolist()
            shape = tuple([len(self.shape)] + [i for i in range(len(self.shape))])
            return ngsolve.CoefficientFunction(tuple([self._grad(fes, t) for t in value]),dims=tuple([3] + list(self.shape))).TensorTranspose(shape)
        
    def error(self):
        val = grad(self)
        mesh = self._grid.finiteElementSpace.mesh
        fes = FiniteElementSpace([self._var], mesh)

        grids = [GridField(fes.gridFunction([val[d]]), self._var) for d in range(3)]
        grids = NGSFunction(grids)
        err = ((grids-val)**2).integrate(fes, element_wise=True)

        fes_L2 =  L2("err", size = 1, order = 0, isScalar=True)
        fes = FiniteElementSpace(fes_L2, mesh)
        g = fes.gridFunction()
        g.vec.data = err.NumPy()

        return GridField(g, fes_L2)

    def _grad(self, fes, x):
        if x == 0:
            return ngsolve.CoefficientFunction((0,0,0))
        else:
            g = ngsolve.grad(x)
            g = tuple([g[i] if i < fes.dimension else ngsolve.CoefficientFunction(0) for i in range(3)])
            return ngsolve.CoefficientFunction(g)

    def __contains__(self, item):
        return self == item

    @property
    def shape(self):
        if self._var.isScalar:
            return ()
        else:
            return tuple([3] * len(self._var.shape))
        
    @property
    def isNonlinear(self):
        return False

    @property
    def valid(self):
        return True    
    
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
    def isTimeDependent(self):
        return True

    def _value(self):
        if self._var.isScalar:
            return self._grid.components[self._n]
        else:
            return list(np.reshape(self._grid.components[self._n:self._n+self._var.size], self._var.shape))

    def __str__(self):
        return self._var.name + "_grid"
    

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

    def to_dict(self):
        return {"name": self._name, "type": self._type, "tdep": self._tdep, "shape": self._shape}
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


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
    

class SolutionFieldFunction(NGSFunctionBase):
    def __init__(self, obj, tdep):
        self._obj = obj
        self._tdep = tdep

    @property
    def shape(self):
        return self._obj.shape

    @property
    def valid(self):
        return self._obj.valid

    def eval(self, fes):
        return self._obj.eval(fes)

    def integrate(self, fes, **kwargs):
        return self._obj.integrate(fes, **kwargs)
            
    def grad(self, fes):
        return self._obj.grad(fes)
    
    def replace(self, d):
        return SolutionFieldFunction(self._obj.replace(d), self._tdep)

    def __contains__(self, item):
        return item in self._obj

    @property
    def hasTrial(self):
        return self._obj.hasTrial
    
    @property
    def rhs(self):
        return self._obj.rhs

    @property
    def lhs(self):
        return self._obj.lhs
        
    @property
    def isNonlinear(self):
        return self._obj.isNonlinear
        
    @property
    def isTimeDependent(self):
        return self._tdep

    def __str__(self):
        return str(self._obj)
