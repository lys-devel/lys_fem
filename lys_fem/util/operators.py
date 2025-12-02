import numpy as np
import ngsolve


def NGSFunction(*args, **kwargs):
    from .coef import NGSFunction as func
    return func(*args, **kwargs)


class Base:
    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def valid(self):
        raise NotImplementedError()

    def eval(self, fes):
        raise NotImplementedError()

    def integrate(self, fes, **kwargs):
        return ngsolve.Integrate(self.eval(fes), fes.mesh, **kwargs)
            
    def grad(self, fes):
        raise NotImplementedError()
    
    def replace(self, d):
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    @property
    def hasTrial(self):
        raise NotImplementedError()
    
    @property
    def rhs(self):
        raise NotImplementedError()

    @property
    def lhs(self):
        raise NotImplementedError()
        
    @property
    def isNonlinear(self):
        raise NotImplementedError()
        
    @property
    def isTimeDependent(self):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()


class NGSFunctionBase(Base):
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

    def __getitem__(self, index):
        if not self.valid:
            return NGSFunction()
        return _Index(self, index)

    @property
    def T(self):
        return _Transpose(self)
    
    def __call__(self, fes, coords):
        f = self.eval(fes)
        g = lambda x: f(x) if x is not None else np.nan
        mip = self.__coordsToMIP(fes.dimension-1, fes.mesh, np.array(coords))
        return np.array(np.vectorize(g)(mip)).squeeze()

    def __coordsToMIP(self, dim, mesh, coords):
        if len(coords.shape) > dim:
            return [self.__coordsToMIP(dim, mesh, c) for c in coords]
        else:
            if dim == 0:
                if mesh.Contains(coords):
                    return mesh(coords)
            else:
                if mesh.Contains(*coords):
                    return mesh(*coords)
            return None


class _BinaryOper(NGSFunctionBase):
    def __init__(self, obj1, obj2):
        if not isinstance(obj1, NGSFunctionBase):
            obj1 = NGSFunction(obj1, str(obj1))
        if not isinstance(obj2, NGSFunctionBase):
            obj2 = NGSFunction(obj2, str(obj2))
        self._obj = [obj1, obj2]

    @property
    def valid(self):
        return True

    def replace(self, d):
        if self in d:
            return d.get(self)
        objs = []
        for i in range(2):
            obj = self._obj[i]
            replaced = obj.replace(d)
            if replaced == 0:
                replaced = NGSFunction()
            objs.append(replaced)
        return self(*objs)
    
    def __contains__(self, item):
        return any([item in self._obj[i] for i in range(2)])

    @property
    def isNonlinear(self):
        return self._obj[0].isNonlinear or self._obj[1].isNonlinear
    
    @property
    def isTimeDependent(self):
        return self._obj[0].isTimeDependent or self._obj[1].isTimeDependent


class _Add(_BinaryOper):
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

    def eval(self, fes):
        return self(self._obj[0].eval(fes), self._obj[1].eval(fes))
    
    def grad(self, fes):
        return self._obj[0].grad(fes) + self._obj[1].grad(fes)
    
    def __str__(self):
        return "(" + str(self._obj[0]) + self._type + str(self._obj[1]) + ")"


class _Mul(_BinaryOper):
    def __init__(self, obj1, obj2):
        super().__init__(obj1, obj2)

    def __call__(self, x, y):
        from .trials import _MultDiffSimbol
        if isinstance(y, _MultDiffSimbol):
            return y * x
        if isinstance(x, ngsolve.CoefficientFunction) and isinstance(y, ngsolve.CoefficientFunction):
            if len(x.shape)!=0 and len(x.shape) == len(y.shape):
                return ngsolve.CoefficientFunction(tuple([xi*yi for xi, yi in zip(x,y)]), dims=x.shape)
            if (len(x.shape) == 0 and y.shape == (1,)) or (x.shape==(1,) and len(y.shape)==0):
                return ngsolve.CoefficientFunction(x*y, dims=(1,))
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

    def eval(self, fes):
        return self(self._obj[0].eval(fes), self._obj[1].eval(fes))

    def grad(self, fes):
        return self._obj[0].eval(fes)*self._obj[1].grad(fes)+self._obj[1].eval(fes)*self._obj[0].grad(fes)

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


class _Div(_BinaryOper):
    def __call__(self, x, y):
        if isinstance(x, ngsolve.CoefficientFunction) and isinstance(y, ngsolve.CoefficientFunction):
            if (len(x.shape) == 0 and y.shape == (1,)) or (x.shape==(1,) and len(y.shape)==0):
                return ngsolve.CoefficientFunction(x/y, dims=(1,))

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

    def eval(self, fes):
        return self(self._obj[0].eval(fes), self._obj[1].eval(fes))

    def grad(self, fes):
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


class _Pow(_BinaryOper):
    def __init__(self, v1, v2):
        super().__init__(v1, v2)
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

    def eval(self, fes):
        return self(self._obj[0].eval(fes), self._pow)
    
    def __str__(self):
        return str(self._obj[0]) + "**" + str(self._pow)

    @property
    def isNonlinear(self):
        return self._obj[0].hasTrial


class _TensorDot(_BinaryOper):
    def __init__(self, obj1, obj2, axes=1):
        super().__init__(obj1, obj2)
        self._axes = axes

    def __call__(self, a, b):
        if self._axes == 1:
            return a.dot(b)
        else:
            return a.ddot(b)
        
    @property
    def shape(self):
        if self._axes == 1:
            return tuple(list(self._obj[0].shape[:-1]) + list(self._obj[1].shape[1:]))
        elif self._axes == 2:
            return tuple(list(self._obj[0].shape[:-2]) + list(self._obj[1].shape[2:]))

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial

    def eval(self, fes):
        if self._obj[0].shape == self._obj[1].shape == ():
            return self._obj[0].eval(fes) * self._obj[1].eval(fes)
        s = self._axes
        v1, v2 = self._obj[0].eval(fes), self._obj[1].eval(fes)
        sym1, sym2 = "abcdef"[0:len(v1.shape)-s]+"ijklmn"[:s], "ijklmn"[:s]+"opqrst"[0:len(v2.shape)-s]
        expr = sym1+","+sym2+"->"+sym1[0:-s]+sym2[s:]
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


class _Cross(_BinaryOper):
    def __call__(self, v1, v2):
        return v1.cross(v2)

    @property
    def hasTrial(self):
        return self._obj[0].hasTrial or self._obj[1].hasTrial
    
    @property
    def shape(self):
        return tuple(list(self._obj[0].shape[:-1]) + [3] + list(self._obj[1].shape[1:]))

    def eval(self, fes):
        v1, v2 = self._obj[0].eval(fes), self._obj[1].eval(fes)
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


class _Index(NGSFunctionBase):
    def __init__(self, obj, index):
        self._obj = obj
        self._index = index

    def __call__(self, obj1):
        return _Index(obj1, self._index)
    
    def __contains__(self, item):
        return False
    
    @property
    def valid(self):
        return self._obj.valid

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
        return self._obj.hasTrial
    
    @property
    def shape(self):
        if isinstance(self._index, int):
            return self._obj.shape[1:]
        else:
            return ()
        
    @property
    def isTimeDependent(self):
        return self._obj.isTimeDependent
    
    def eval(self, fes):
        if isinstance(self._index, int):
            sl = [int(self._index)] + [slice(None)]*(len(self.shape))
            return self._obj.eval(fes)[tuple(sl)]
        else:
            return self._obj.eval(fes)[self._index]

    def replace(self, d):
        return self._obj.replace(d)[self._index]
    
    def __str__(self):
        return str(self._obj) + "[" + str(self._index) + "]"

    @property
    def isNonlinear(self):
        return self._obj.hasTrial


class _Transpose(NGSFunctionBase):
    def __init__(self, obj):
        self._obj = obj

    def __call__(self, obj):
        return _Transpose(obj)
    
    def __contains__(self, item):
        return item in self._obj

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
    def valid(self):
        return self._obj.valid

    @property
    def hasTrial(self):
        return self._obj.hasTrial
    
    @property
    def shape(self):
        return tuple(reversed(self._obj.shape))
    
    def eval(self, fes):
        return self._obj.eval(fes).TensorTranspose((1,0))
    
    def __str__(self):
        return str(self._obj) + ".T"

    def replace(self, d):
        return self._obj.replace(d).T
    
    @property
    def isNonlinear(self):
        return self._obj.hasTrial
    
    @property
    def isTimeDependent(self):
        return self._obj.isTimeDependent