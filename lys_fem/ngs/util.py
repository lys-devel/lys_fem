import numpy as np
import ngsolve

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res


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
    def __init__(self, obj=None, name="Undefined", grad=False):
        self._hasTrial = False
        if obj is None or obj == 0:
            self._obj = None
            self._name = "0"
        elif isinstance(obj, (int, float, complex)):
            self._obj = ngsolve.CoefficientFunction(obj)
            self._name = str(obj)
        else:
            self._obj = obj
            self._name = name
        self._grad = grad

    @property
    def shape(self):
        if hasattr(self._obj, "shape"):
            return self._obj.shape
        else:
            return ()
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            other = NGSFunction(other)
        if self.valid and other.valid:
            return _Mul(self, other)
        else:
            return NGSFunction()

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            other = NGSFunction(other)
        if self.valid and other.valid:
            return _Mul(self, other, "/")
        else:
            return NGSFunction()
        
    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = NGSFunction(other)
        if not self.valid:
            return other
        elif not other.valid:
            return self
        else:
            return _Add(self, other)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = NGSFunction(other)
        if not self.valid:
            return (-1)*other
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

    def __pow__(self, other):
        if not self.valid:
            return self
        return _Pow(self, other)

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
            return NGSFunction(J[0,0]*J[1,1] - J[0,1]*J[1,0], "|"+self._name+"|")
        elif J.shape[0] == 1:
            return NGSFunction(J[0,0], "|"+self._name+"|")
        
    @property
    def grad(self):
        return NGSFunction(self._obj, "grad("+self._name+")", grad=True)

    def __str__(self):
        return self._name
    
    @property
    def hasTrial(self):
        return self._hasTrial
    
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
            if self._grad:
                raise RuntimeError("grad for general NGSFUnction is not implemented.")
            else:
                return self._obj
        else:
            return ngsolve.CoefficientFunction(0)*ngsolve.dx
        
    @property
    def isNonlinear(self):
        return False


class NGSFunctionVector(NGSFunction):
    def __init__(self, objs, name="Undefined"):
        self._objs = [obj if isinstance(obj, NGSFunction) else NGSFunction(obj) for obj in objs]
        self._name = name

    @property
    def shape(self):
        return tuple([len(self._objs)] + list(self._objs[0].shape))
            
    @property
    def grad(self):
        return NGSFunctionVector([grad(obj) for obj in self._objs])
    
    @property
    def hasTrial(self):
        return any([obj.hasTrial for obj in self._objs])
    
    @property
    def rhs(self):
        return NGSFunctionVector([obj.rhs for obj in self._objs])

    @property
    def lhs(self):
        return NGSFunctionVector([obj.lhs for obj in self._objs])
        
    @property
    def valid(self):
        return any([obj.valid for obj in self._objs])
        
    def eval(self):
        if self.valid:
            return ngsolve.CoefficientFunction(tuple([obj.eval() if obj.valid else 0 for obj in self._objs]), dims=self.shape)
        else:
            return ngsolve.CoefficientFunction(0)*ngsolve.dx
        
    @property
    def isNonlinear(self):
        return any([obj.isNonlinear for obj in self._objs])
    

class NGSDomainWiseFunction(NGSFunction):
    def __init__(self, objs, mesh, geomType="domain", default=None, name="Undefined"):
        self._objs = {key: obj if isinstance(obj, NGSFunction) else NGSFunction(obj) for key, obj in objs.items()}
        self._mesh = mesh
        self._default = default
        self._geom = geomType
        self._name = name

    @property
    def _first(self):
        return list(self._objs.values())[0]

    @property
    def shape(self):
        return self._first.shape
            
    @property
    def grad(self):
        return NGSDomainWiseFunction({key: grad(obj) for key, obj in self._objs.items()})

    def __str__(self):
        return self._name
    
    @property
    def hasTrial(self):
        return any([obj.hasTrial for obj in self._objs.values()])
    
    @property
    def rhs(self):
        return NGSDomainWiseFunction({key: obj.rhs for key, obj in self._objs.items()}, self._name)

    @property
    def lhs(self):
        return NGSDomainWiseFunction({key: obj.lhs for key, obj in self._objs.items()}, self._name)
        
    @property
    def valid(self):
        return any([obj.valid for obj in self._objs.values()])
        
    def eval(self):
        if self.valid:
            coefs = {key: obj.eval() if obj.valid else 0 for key, obj in self._objs.items()}
            if self._geom=="domain":
                return self._mesh.MaterialCF(coefs, default=self._default)
            else:
                return self._mesh.BoundaryCF(coefs, default=self._default)
        else:
            return ngsolve.CoefficientFunction(0)*ngsolve.dx
        
    @property
    def isNonlinear(self):
        return any([obj.isNonlinear for obj in self._objs.values()])


def printError(f):
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            print("Error while evaluating", args[0])
            return None
    return wrapper


class _Oper(NGSFunction):
    def __init__(self, obj1, obj2=None):
        if not isinstance(obj1, NGSFunction):
            obj1 = NGSFunction(obj1, str(obj1))
        if not isinstance(obj2, NGSFunction) and obj2 is not None:
            obj2 = NGSFunction(obj2, str(obj2))
        super().__init__([obj1, obj2])

    def replace(self, d):
        objs = []
        for i in range(2):
            obj = self._obj[i]
            if isinstance(obj, _Oper):
                replaced = obj.replace(d)
            else:
                replaced = d.get(obj, obj)
            if replaced == 0:
                replaced = NGSFunction()
            objs.append(replaced)
        return self(*objs)
    
    @property
    def isNonlinear(self):
        return self._obj[0].isNonlinear or self._obj[1].isNonlinear


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
            if isinstance(x, ngsolve.CoefficientFunction) and isinstance(y, ngsolve.CoefficientFunction):
                if len(x.shape)!=0 and len(x.shape) == len(y.shape):
                    return ngsolve.CoefficientFunction(tuple([xi*yi for xi, yi in zip(x,y)]), dims=x.shape)
            if isinstance(y, _DMul):
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

    @printError
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
    def __call__(self, v1, v2):
        return v1.cross(v2)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        # Levi Civita symbol
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        eijk = tuple([tuple([tuple(ek) for ek in ejk]) for ejk in eijk])
        eijk = ngsolve.CoefficientFunction(eijk, dims=(3,3,3))

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


class _Pow(_Oper):
    def __call__(self, v1, v2):
        return v1**v2

    @property
    def rhs(self):
        return self(self._obj[0].rhs, self._obj[1])

    @property
    def lhs(self):
        return self(self._obj[0].lhs, self._obj[1])

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())
    
    def __str__(self):
        return str(self._obj[0]) + "**" + str(self._obj[1])


class TrialFunction(NGSFunction):
    def __init__(self, name, obj, dt=0, grad=False, xscale=1, scale=1):
        super().__init__(obj, name="trial("+name+")")
        self._name = name
        self._dt = dt
        self._grad = grad
        self._xscale = xscale
        self._scale = scale

    @property
    def t(self):
        return TrialFunction(self._name, self._obj, self._dt+1, grad=self._grad, xscale=self._xscale, scale=self._scale)

    @property
    def tt(self):
        return TrialFunction(self._name, self._obj, self._dt+2, grad=self._grad, xscale=self._xscale, scale=self._scale)

    @property    
    def grad(self):
        return TrialFunction(self._name, self._obj, self._dt, grad=True, xscale=self._xscale, scale=self._scale)
    
    @property
    def value(self):
        return TrialFunctionValue(self.eval(), self._name)
       
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
                return self._scale/self._xscale * ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._obj]), dims=(ngsolve.grad(self._obj[0]).shape[0], len(self._obj))).TensorTranspose((1,0))
            return self._scale/self._xscale * ngsolve.grad(self._obj)
        else:
            if isinstance(self._obj, list):
                return self._scale * ngsolve.CoefficientFunction(tuple([t for t in self._obj]))
            return self._scale * self._obj


class TestFunction(NGSFunction):
    def __init__(self, obj, name, grad=False, xscale=1, scale=1):
        super().__init__(obj, name="test("+name+")")
        self._grad = grad
        self._nam = name
        self._xscale = xscale
        self._scale = scale

    @property    
    def grad(self):
        return TestFunction(self._obj, "grad("+self._nam+")", grad=True, xscale=self._xscale, scale=self._scale)
    
    def eval(self):
        if self._grad:
            if isinstance(self._obj, list):
                return self._scale/self._xscale * ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._obj]), dims=(ngsolve.grad(self._obj[0]).shape[0], len(self._obj))).TensorTranspose((1,0))
            return self._scale/self._xscale * ngsolve.grad(self._obj)
        else:
            if isinstance(self._obj, list):
                return self._scale*ngsolve.CoefficientFunction(tuple([t for t in self._obj]))
            return self._scale*self._obj
        
    def __hash__(self):
        return hash(self._name + "__" + str(self._grad))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    

class TrialFunctionValue(NGSFunction):
    def __hash__(self):
        return hash("Value__" + self._name)
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def isNonlinear(self):
        return True


class DifferentialSymbol(NGSFunction):
    def __init__(self, obj, scale=1, **kwargs):
        super().__init__(obj, **kwargs)
        self._scale = scale

    def setScale(self, scale):
        self._scale = scale

    def setMesh(self, mesh):
        self._mesh = mesh

    def eval(self):
        return _DMul(self._obj, self._scale)

    def __call__(self, region):
        geom = "|".join([region.geometryType.lower() + str(r) for r in region])
        if self == dx:
            d = self._mesh.Materials(geom)
        else:
            d = self._mesh.Boundaries(geom)
        return DifferentialSymbol(self._obj(definedon=d), self._scale)

class _DMul:
    def __init__(self, obj, scale):
        self._obj = obj
        self._scale = scale

    def __mul__(self, other):
        return self._scale * other * self._obj


dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
