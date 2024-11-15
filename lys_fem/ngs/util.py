import builtins
import numpy as np
import sympy as sp
import ngsolve

def eval(expr, dic={}):
    return builtins.eval(expr, globals(), dic)

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res


def grad(f):
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

    def setComponent(self, var, value, model):
        if self.isSingle:
            self.Set(value)
        else:
            n = 0
            for v in model.variables:
                if v.name == var.name:
                    break
                n += v.size
            for i in range(var.size):
                self.components[n+i].Set(value[i])

    @property
    def components(self):
        if self.isSingle:
            return [self]
        else:
            return super().components

    @property
    def isSingle(self):
        return not isinstance(self.space, ngsolve.ProductSpace)

    def toNGSFunctions(self, model, pre=""):
        res = {}
        n = 0
        for v in model.variables:
            if v.size == 1 and v.isScalar:
                res[v.name] = NGSFunction(v.scale*self.components[n], name=v.name+pre, tdep=True)
            else:
                res[v.name] = NGSFunction(v.scale*ngsolve.CoefficientFunction(tuple(self.components[n:n+v.size])), name=v.name+pre, tdep=True)
            n+=v.size
        return res

    def toGradFunctions(self, model, pre=""):
        res = {}
        n = 0
        for v in model.variables:
            if v.size == 1 and v.isScalar:
                res[v.name] = NGSFunction(v.scale*ngsolve.grad(self.components[n]), name="grad("+v.name+pre+")", tdep=True)
            else:
                g = [ngsolve.grad(self.components[i]) for i in range(n,n+v.size)]
                res[v.name] = NGSFunction(v.scale*ngsolve.CoefficientFunction(tuple(g), dims=(g[0].shape[0], v.size)).TensorTranspose((1,0)), name="grad("+v.name+pre+")", tdep=True)
            n+=v.size           
        return res


class NGSFunction:
    def __init__(self, obj=None, mesh=None, default=None, geomType="domain", name="Undefined", tdep=False):
        if obj is None or obj == 0:
            self._obj = None
            self._name = "0"
        elif isinstance(obj, (int, float, complex, sp.Integer, sp.Float)):
            self._obj = ngsolve.CoefficientFunction(obj)
            self._name = str(obj)
            self._tdep = False
        elif isinstance(obj, (list, tuple, np.ndarray)):
            self._obj = [value if isinstance(value, NGSFunction) else NGSFunction(value) for value in obj]
            self._name = name
        elif isinstance(obj, dict):
            self._obj = {key: value if isinstance(value, NGSFunction) else NGSFunction(value) for key, value in obj.items()}
            self._mesh = mesh
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

    def eval(self):
        if self._obj is None:
            return ngsolve.CoefficientFunction(0)
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            return self._obj
        elif isinstance(self._obj, list):
            return ngsolve.CoefficientFunction(tuple([obj.eval() for obj in self._obj]), dims=self.shape)
        elif isinstance(self._obj, dict):
            coefs = {key: obj.eval() for key, obj in self._obj.items()}
            if self._default is None:
                default = None
            else:
                default = self._default.eval()
            if self._geom=="domain":
                return self._mesh.MaterialCF(coefs, default=default)
            else:
                return self._mesh.BoundaryCF(coefs, default=default)
            
    def grad(self):
        if self._obj is None:
            return ngsolve.CoefficientFunction([0]*dimension)
        if isinstance(self._obj, ngsolve.CoefficientFunction):
            g = [self._obj.Diff(symbol) for symbol in [ngsolve.x, ngsolve.y, ngsolve.z][:dimension]]
            return ngsolve.CoefficientFunction(tuple(g), dims=(dimension,))
        raise RuntimeError("grad not implemented")
    
    def replace(self, d):
        return (1*self).replace(d)

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
        
    @property
    def value(self):
        return self

    def __str__(self):
        return self._name

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
            return _Div(self, other)
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
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float, complex)):
            other = NGSFunction(other)
        return other/self

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
        
    def __getitem__(self, index):
        if self._obj is None:
            return NGSFunction()
        return _Index(self, index)
        
    @property
    def T(self):
        return _Transpose(self)

def printError(f):
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
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

    def replace(self, d, type="NGS"):
        if self in d:
            return d.get(self)
        objs = []
        for i in range(2):
            obj = self._obj[i]
            if isinstance(obj, _Oper):
                replaced = obj.replace(d, type)
            else:
                replaced = d.get(obj, obj)
            if replaced == 0 and type=="NGS":
                replaced = NGSFunction()
            if type=="value" and isinstance(replaced, NGSFunction):
                replaced = eval(str(replaced))
            objs.append(replaced)
        #print(self, objs)
        return self(*objs)
    
    @property
    def isNonlinear(self):
        return self._obj[0].isNonlinear or self._obj[1].isNonlinear
    
    @property
    def isTimeDependent(self):
        return self._obj[0].isTimeDependent or self._obj[1].isTimeDependent


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

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())
    
    def __str__(self):
        return "(" + str(self._obj[0]) + self._type + str(self._obj[1]) + ")"


class _Mul(_Oper):
    def __init__(self, obj1, obj2):
        super().__init__(obj1, obj2)

    def __call__(self, x, y):
        if isinstance(x, ngsolve.CoefficientFunction) and isinstance(y, ngsolve.CoefficientFunction):
            if len(x.shape)!=0 and len(x.shape) == len(y.shape):
                return ngsolve.CoefficientFunction(tuple([xi*yi for xi, yi in zip(x,y)]), dims=x.shape)
        if isinstance(x, (ngsolve.la.DynamicVectorExpression, ngsolve.la.BaseVector)) and isinstance(y, (int, float, complex)):
            return y * x
        return x * y

    @property
    def shape(self):
        if len(self._obj[0].shape)==0:
            return self._obj[1].shape
        else:
            return self._obj[0].shape

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())

    def grad(self):
        return self._obj[0].eval()*self._obj[1].grad()+self._obj[1].eval()*self._obj[0].grad()

    def __str__(self):
        if isinstance(self._obj[0], _Mul) or isinstance(self._obj[1], _Mul):
            return str(self._obj[0]) + "*" + str(self._obj[1])
        return "(" + str(self._obj[0]) + "*" + str(self._obj[1]) + ")"

    @property
    def rhs(self):
        return self._obj[0].rhs * self._obj[1].rhs

    @property
    def lhs(self):
        return self._obj[0].lhs * self._obj[1].lhs + self._obj[0].lhs * self._obj[1].rhs + self._obj[0].rhs * self._obj[1].lhs

    @property
    def isNonlinear(self):
        if self._obj[0].hasTrial and self._obj[1].hasTrial:
            return True
        else:
            return super().isNonlinear


class _Div(_Oper):
    def __call__(self, x, y):
        if isinstance(x, ngsolve.la.DynamicVectorExpression) and isinstance(y, (int, float, complex)):
            return 1 / y * x
        return x / y

    @property
    def shape(self):
        if len(self._obj[0].shape)==0:
            return self._obj[1].shape
        else:
            return self._obj[0].shape

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._obj[1].eval())

    def grad(self):
        raise RuntimeError("grad not implenented")

    def __str__(self):
        if isinstance(self._obj[0], _Mul) or isinstance(self._obj[1], _Mul):
            return str(self._obj[0]) + "/" + str(self._obj[1])
        return "(" + str(self._obj[0]) + "/" + str(self._obj[1]) + ")"

    @property
    def rhs(self):
        if self._obj[1].hasTrial:
            return NGSFunction()
        else:
            return self._obj[0].rhs/self._obj[1]

    @property
    def lhs(self):
        if self._obj[1].hasTrial:
            return self
        else:
            return self._obj[0].lhs/self._obj[1]

    @property
    def isNonlinear(self):
        if self._obj[1].hasTrial:
            return True
        return self._obj[0].isNonlinear


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
        return self(self._obj[0].rhs, self._obj[1].rhs)

    @property
    def lhs(self):
        return self(self._obj[0].lhs, self._obj[1].lhs) + self(self._obj[0].lhs, self._obj[1].rhs) + self(self._obj[0].rhs, self._obj[1].lhs)

    @property
    def isNonlinear(self):
        if self._obj[0].hasTrial and self._obj[1].hasTrial:
            return True
        else:
            return super().isNonlinear

class _Cross(_Oper):
    def __call__(self, v1, v2):
        return v1.cross(v2)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self):
        v1, v2 = self._obj[0].eval(), self._obj[1].eval()
        if len(v1.shape) == len(v2.shape) == 1:
            return ngsolve.Cross(v1, v2)

        # Levi Civita symbol
        eijk = np.zeros((3, 3, 3))
        eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
        eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
        eijk = tuple([tuple([tuple(ek) for ek in ejk]) for ejk in eijk])
        eijk = ngsolve.CoefficientFunction(eijk, dims=(3,3,3))

        # Create expression
        sym1, sym2 = "abcdefgh"[:len(v1.shape)], "nmpqrs"[:len(v2.shape)]
        expr = "i"+sym1[-1]+sym2[0]+","+sym1+","+sym2+"->"+sym1[:-1]+"i"+sym2[1:]

        # Calculate by einsum
        return ngsolve.fem.Einsum(expr, eijk, v1, v2)

    def __str__(self):
        return "(" + str(self._obj[0]) + " x " + str(self._obj[1]) + ")"

    @property
    def rhs(self):
        return self._obj[0].rhs.cross(self._obj[1].rhs)

    @property
    def lhs(self):
        return self._obj[0].lhs.cross(self._obj[1].lhs)+self._obj[0].rhs.cross(self._obj[1].lhs)+self._obj[0].lhs.cross(self._obj[1].rhs)

    @property
    def isNonlinear(self):
        if self._obj[0].hasTrial and self._obj[1].hasTrial:
            return True
        else:
            return super().isNonlinear


class _Pow(_Oper):
    def __init__(self, v1, v2):
        super().__init__(v1)
        self._pow = v2

    def __call__(self, v1, v2):
        return v1 ** self._pow
    
    @property
    def shape(self):
        return self._obj[0].shape

    @property
    def rhs(self):
        return self(self._obj[0].rhs, self._pow)

    @property
    def lhs(self):
        return self(self._obj[0].lhs, self._pow)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial

    def eval(self):
        return self(self._obj[0].eval(), self._pow)
    
    def __str__(self):
        return str(self._obj[0]) + "**" + str(self._pow)

    @property
    def isNonlinear(self):
        return self._obj[0].hasTrial


class _Index(_Oper):
    def __init__(self, obj1, index):
        super().__init__(obj1)
        self._index = index

    def __call__(self, obj1, obj2):
        return _Index(obj1, self._index)

    @property
    def lhs(self):
        if self.hasTrial:
            return self
        else:
            return NGSFunction()
        
    @property
    def rhs(self):
        if self.hasTrial:
            return NGSFunction()
        else:
            return self
        
    @property
    def hasTrial(self):
        return self._obj[0].hasTrial
    
    @property
    def shape(self):
        return self._obj[0].shape[1:]
    
    def eval(self):
        sl = [int(self._index)] + [slice(None)]*(len(self.shape))
        return self._obj[0].eval()[tuple(sl)]
    
    def __str__(self):
        return str(self._obj[0]) + "[" + str(self._index) + "]"

    @property
    def isNonlinear(self):
        return self._obj[0].hasTrial


class _Transpose(_Oper):
    def __init__(self, obj):
        super().__init__(obj)

    def __call__(self, obj1, obj2):
        return _Transpose(obj1)

    @property
    def lhs(self):
        if self.hasTrial:
            return self
        else:
            return NGSFunction()
        
    @property
    def rhs(self):
        if self.hasTrial:
            return NGSFunction()
        else:
            return self
        
    @property
    def hasTrial(self):
        return self._obj[0].hasTrial
    
    @property
    def shape(self):
        return tuple(reversed(self._obj[0].shape))
    
    def eval(self):
        return self._obj[0].eval().TensorTranspose((1,0))
    
    def __str__(self):
        return str(self._obj[0]) + ".T"

    @property
    def isNonlinear(self):
        return self._obj[0].hasTrial


class _Func(_Oper):
    def __init__(self, obj1, type):
        super().__init__(obj1)
        self._type = type

    def __call__(self, obj1, obj2=None):
        return _Func(obj1, self._type)

    @property
    def rhs(self):
        return self(self._obj[0].rhs)

    @property
    def lhs(self):
        return self(self._obj[0].lhs)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial

    @property
    def isNonlinear(self):
        if self._type == "grad":
            return self._obj[0].isNonlinear
        else:
            return self._obj[0].hasTrial

    @property
    def shape(self):
        if self._type == "grad":
            return tuple(list(self._obj[0].shape)+[dimension])
        else:
            return self._obj[0].shape

    def __hash__(self):
        if self._type == "grad" and isinstance(self._obj[0], (TrialFunction, TestFunction)):
            return hash(str(self._obj[0])+"__grad")
        return super().__hash__()
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def eval(self):
        if self._type == "exp":
            return ngsolve.exp(self._obj[0].eval())
        if self._type == "sin":
            return ngsolve.sin(self._obj[0].eval())
        if self._type == "cos":
            return ngsolve.cos(self._obj[0].eval())
        if self._type == "tan":
            return ngsolve.tan(self._obj[0].eval())
        if self._type == "step":
            return ngsolve.IfPos(self._obj[0].eval(), 1, 0)
        if self._type == "sqrt":
            return ngsolve.sqrt(self._obj[0].eval())    
        if self._type == "norm":
            v = self._obj[0].eval()
            return ngsolve.sqrt(v*v)
        if self._type == "grad":
            if hasattr(self._obj[0], "grad"):
                return self._obj[0].grad()
            raise RuntimeError("grad is not implemented for " + str(type(self._obj[0])))
    
    def __str__(self):
        return self._type + "(" + str(self._obj[0]) + ")"


class _MinMax(_Oper):
    def __init__(self, obj1, obj2, type="min"):
        super().__init__(obj1, obj2)
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
        return self._obj[0].hasTrial

    def eval(self):
        e1, e2 = self._obj[0].eval(), self._obj[1].eval()
        if self._type == "max":
            return ngsolve.IfPos(e1-e2, e1, e2)
        else:
            return ngsolve.IfPos(e1-e2, e2, e1)
    
    def __str__(self):
        return self._type + "(" + str(self._obj[0]) + ", " + str(self._obj[1]) + ")"


class TrialFunction(NGSFunction):
    def __init__(self, name, obj, dt=0, scale=1):
        super().__init__(obj, name="trial("+name+")")
        self._name = name
        self._tris = obj
        self._dt = dt
        self._scale = scale

    @property
    def t(self):
        return TrialFunction(self._name, self._tris, self._dt+1, scale=self._scale)

    @property
    def tt(self):
        return TrialFunction(self._name, self._tris, self._dt+2, scale=self._scale)

    @property
    def value(self):
        return TrialFunctionValue(self._tris, name = self._name)
    
    @property
    def rhs(self):
        return NGSFunction()
    
    @property
    def lhs(self):
        return self
       
    def __hash__(self):
        return hash(self._name + "__" + str(self._dt))
    
    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __str__(self):
        name = self._name
        if self._dt == -1:
            return name+"0"
        for i in range(self._dt):
            name += "t" 
        return name

    @property
    def hasTrial(self):
        return True
    
    def eval(self):
        return self._scale * super().eval()
        
    def grad(self):
        if isinstance(self._tris, list):
            return self._scale * ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._tris]), dims=(ngsolve.grad(self._tris[0]).shape[0], len(self._tris))).TensorTranspose((1,0))
        return self._scale * ngsolve.grad(super().eval())
        
    @property
    def isNonlinear(self):
        return False
        

class TestFunction(NGSFunction):
    def __init__(self, obj, name, scale=1):
        super().__init__(obj, name="test("+name+")")
        self._tests = obj
        self._nam = name
        self._scale = scale

    def eval(self):
        return self._scale*super().eval()
        
    def grad(self):
        if isinstance(self._tests, list):
            return self._scale * ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._tests]), dims=(ngsolve.grad(self._tests[0]).shape[0], len(self._obj))).TensorTranspose((1,0))
        return self._scale * ngsolve.grad(self._tests)
        
    def __hash__(self):
        return hash(self._name)
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def lhs(self):
        return NGSFunction()

    @property
    def rhs(self):
        return self

    @property
    def isNonlinear(self):
        return False
     

class TrialFunctionValue(NGSFunction):
    def __init__(self, obj, name):
        super().__init__(obj, name=name)
        self._nam = name
        self._tris = obj

    def __hash__(self):
        return hash("Value__" + self._name)
    
    def __eq__(self, other):
        return hash(self) == hash(other)

    def __str__(self):
        return self._nam

    def grad(self):
        if isinstance(self._tris, list):
            return ngsolve.CoefficientFunction(tuple([ngsolve.grad(t) for t in self._tris]), dims=(ngsolve.grad(self._tris[0]).shape[0], len(self._tris))).TensorTranspose((1,0))
        return ngsolve.grad(super().eval())

    @property
    def hasTrial(self):
        return True

    @property
    def isNonlinear(self):
        return False

    @property
    def lhs(self):
        return self

    @property
    def rhs(self):
        return NGSFunction()


class DifferentialSymbol(NGSFunction):
    def __init__(self, obj, **kwargs):
        super().__init__(obj, **kwargs)

    def setMesh(self, mesh):
        self._mesh = mesh

    def eval(self):
        return self._obj
    
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
        if self == dx:
            d = self._mesh.Materials(geom)
        else:
            d = self._mesh.Boundaries(geom)
        return DifferentialSymbol(self._obj(definedon=d), name=str(self))


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


dx = DifferentialSymbol(ngsolve.dx, name="dx")
ds = DifferentialSymbol(ngsolve.ds, name="ds")
t = Parameter("t", 0, tdep=True)
stepn = Parameter("step", 0, tdep=True)
dimension = 3