import ngsolve
from .operators import NGSFunctionBase
from .coef import NGSFunction


class TrialFunction(NGSFunctionBase):
    def __init__(self, var, dt=0):
        self._var = var
        self._dt = dt

    @property
    def t(self):
        return TrialFunction(self._var, self._dt+1)

    @property
    def tt(self):
        return TrialFunction(self._var, self._dt+2)
    
    @property
    def rhs(self):
        return NGSFunction()
    
    @property
    def lhs(self):
        return self
    
    @property
    def shape(self):
        if self._var.isScalar:
            return ()
        else:
            return (self._var.size,)
       
    def __hash__(self):
        return hash(self._var.name + "__" + str(self._dt))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __str__(self):
        name = self._var.name
        if self._dt == -1:
            return name+"0"
        for i in range(self._dt):
            name += "t" 
        return name

    @property
    def hasTrial(self):
        return True
    
    def eval(self, fes):
        trial = fes.trial(self._var)
        if isinstance(trial, list):
            return ngsolve.CoefficientFunction(tuple(trial), dims=self.shape)
        return trial
        
    def grad(self, fes):
        trial = fes.trial(self._var)
        if isinstance(trial, list):
            return ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in trial]), dims=(len(trial), fes.dimension)).TensorTranspose((1,0))
        return ngsolve.grad(trial)
        
    @property
    def isNonlinear(self):
        return False

    @property
    def isTimeDependent(self):
        return False

    @property
    def valid(self):
        return True

    def replace(self, d):
        return d.get(self, self)
    
    def __contains__(self, item):
        return self == item
        

class TestFunction(NGSFunctionBase):
    def __init__(self, var):
        self._var = var

    def eval(self, fes):
        test = fes.test(self._var)
        if isinstance(test, list):
            return ngsolve.CoefficientFunction(tuple(test), dims=self.shape)
        return test
        
    def grad(self, fes):
        test = fes.test(self._var)
        if isinstance(test, list):
            return ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in test]), dims=(len(test), fes.dimension)).TensorTranspose((1,0))
        return ngsolve.grad(test)
        
    def __hash__(self):
        return hash(self._var.name)
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def shape(self):
        if self._var.isScalar:
            return ()
        else:
            return (self._var.size,)

    @property
    def lhs(self):
        return NGSFunction()

    @property
    def rhs(self):
        return self

    @property
    def isNonlinear(self):
        return False

    @property
    def isTimeDependent(self):
        return False

    @property
    def valid(self):
        return True
    
    @property
    def hasTrial(self):
        return False

    def replace(self, d):
        return d.get(self, self)

    def __contains__(self, item):
        return self == item
    
    def __str__(self):
        return "test(" + self._var.name + ")"


class SolutionFunction(NGSFunctionBase):
    """
    NGSFunction that provide the access to the present solution.
    Args:
        name(str): The symbol name
        sol(Solution): The solution object
        type(int): The type of the solution. 0:x, 1:x.t, 2:x.tt
    """
    def __init__(self, var, sol, type):
        self._sol = sol
        self._var = var
        self._type = type
        n = 0
        for v in sol.finiteElementSpace.variables:
            if v == var:
                self._n = n
            n += v.size

    @property
    def shape(self):
        if self._var.isScalar:
            return ()
        else:
            return (self._var.size,)

    @property
    def valid(self):
        return True

    def eval(self, fes):
        v = self._var
        g = self._sol[self._type]
        if v.isScalar:
            return ngsolve.CoefficientFunction(g.components[self._n])
        else:
            return ngsolve.CoefficientFunction(tuple(g.components[self._n:self._n+v.size]), dims=(v.size,))
            
    def grad(self, fes):
        v = self._var
        g = self._sol[self._type]

        if v.isScalar:
            return ngsolve.grad(g.components[self._n])
        else:
            g = [ngsolve.grad(g.components[i]) for i in range(self._n,self._n+v.size)]
            print(v.size, fes.dimension, len(g), g[0].shape)
            return ngsolve.CoefficientFunction(tuple(g), dims=(v.size, fes.dimension)).TensorTranspose((1,0))
    
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
        return True

    def __str__(self):
        return self._var.name + "_n"


class DifferentialSymbol(NGSFunctionBase):
    def __init__(self, obj, geom=None, name=""):
        self._obj = obj
        self._geom = geom
        self._name = name

    def eval(self, fes):
        from .functions import det
        if fes.J is None:
            J = None
        else:
            J = det(fes.J).eval(fes)
        if self._geom is None:
            return _MultDiffSimbol(self._obj, J)
        else:
            if self._obj == ngsolve.dx:
                g = fes.mesh.Materials(self._geom)
            else:
                g = fes.mesh.Boundaries(self._geom)
            return _MultDiffSimbol(self._obj(definedon=g), J)

    @property
    def hasTrial(self):
        return False
        
    @property
    def rhs(self):
        return self

    @property
    def lhs(self):
        return NGSFunction()

    def replace(self, d):
        return d.get(self, self)
    
    def __contains__(self, item):
        return self == item
            
    @property
    def shape(self):
        return ()
    
    @property
    def valid(self):
        return True

    @property
    def hasTrial(self):
        return False
        
    @property
    def isNonlinear(self):
        return False
    
    @property
    def isTimeDependent(self):
        return False

    def __call__(self, region):
        geom = "|".join([region.geometryType.lower() + str(r) for r in region])
        return DifferentialSymbol(self._obj, geom, name=str(self))
    
    def __str__(self):
        return self._name


class _MultDiffSimbol:
    def __init__(self, obj, J):
        self._obj = obj
        self._J = J

    def __mul__(self, other):
        if self._J is None:
            return other * self._obj
        return (other / self._J) * self._obj
