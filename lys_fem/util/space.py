import ngsolve
from .trials import TestFunction, TrialFunction
from .operators import NGSFunctionBase

def prod(args):
    res = args[0]
    for arg in args[1:]:
        res = res * arg
    return res


class FunctionSpace:
    def __init__(self, name, fetype="H1", size=1, dirichlet=None, isScalar=True, valtype="x", geometries=None, order=1):
        self._name = name
        self._type = fetype
        self._size = size
        self._scalar = isScalar
        if dirichlet is None:
            dirichlet = [None] * size
        self._dirichlet = dirichlet
        self._valtype = valtype
        self._geom = geometries
        self._order = order

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size
    
    @property
    def isScalar(self):
        return self._scalar and self._size==1

    @property
    def type(self):
        return self._valtype

    def eval(self, mesh):
        if self._type == "H1":
            space = ngsolve.H1
        elif self._type == "L2":
            space = ngsolve.L2

        kwargs = {"order": self._order}
        if self._geom is not None:
            kwargs["definedon"] = "|".join(["domain" + str(r) for r in self._geom])

        fess = []
        for i in range(self.size):
            if self._dirichlet[i] is None:
                fess.append(space(mesh, **kwargs))
            else:
                dirichlet = "|".join(["boundary" + str(r) for r in self._dirichlet[i]])
                fess.append(space(mesh, dirichlet=dirichlet, **kwargs))
        return prod(fess)
    
    @property
    def trial(self):
        return TrialFunction(self)
    
    @property
    def test(self):
        return TestFunction(self)
    
    def __str__(self):
        return "symbol = " + self._name + ", space = " + self._type + ", size = " + str(self._size) + ", order = " + str(self._order) + ", valtype = " + str(self._valtype)

    def __eq__(self, other):
        if not isinstance(other, FunctionSpace):
            return False
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self._name)


class H1(FunctionSpace):
    def __init__(self, name, **kwargs):
        super().__init__(name, "H1", **kwargs)


class L2(FunctionSpace):
    def __init__(self, name, **kwargs):
        super().__init__(name, "L2", **kwargs)


class FiniteElementSpace:
    """
    Finite-element space composed of multiple :class:`FunctionSpace` and mesh.

    Args:
        vars(FunctionSpace or list of FunctionSpace): The functions spaces.
        mesh(aaa): The mesh object.

    """
    def __init__(self, vars, mesh):
        self._mesh = mesh
        self._vars = [vars] if isinstance(vars, FunctionSpace) else vars
        self._fes = prod([v.eval(mesh) for v in self._vars])
        self._tnt = self._TnT_dict(self._vars, self._fes)

    def _TnT_dict(self, vars, fes):
        if isinstance(fes, (ngsolve.ProductSpace, ngsolve.comp.Compress)):
            trials, tests = fes.TnT()
        else:
            trials, tests = [[t] for t in fes.TnT()]

        n = 0
        res = {}
        for var in vars:
            if var.isScalar:
                trial, test = trials[n], tests[n]
            else:
                trial, test = trials[n:n+var.size], tests[n:n+var.size]
            res[var] = (trial, test)
            n+=var.size
        return res

    @property
    def mesh(self):
        return self._mesh
    
    @property
    def variables(self):
        return self._vars
        
    def compress(self, symbols):
        if symbols is None:
            return self
        return CompressedFESpace(self, symbols)
    
    @property
    def dimension(self):
        """
        Return spatial dimension of the given mesh.

        Returns:
            int: The dimension of the mesh.
        """
        return self._mesh.dim
    
    def gridFunction(self, value=None):
        return GridFunction(self, value)
    
    @property
    def ndof(self):
        return self._fes.ndofglobal
    
    def trial(self, var):
        return self._tnt[var][0]

    def test(self, var):
        return self._tnt[var][1]
    
    def BilinearForm(self, blf, condense=False, symmetric=False):
        return _BilinearForm(self, blf, condense=condense, symmetric=symmetric)

    def LinearForm(self, lf):
        return _LinearForm(self, lf)
    
    def FreeDofs(self, condense=False):
        return self._fes.FreeDofs(condense)
    
    def __str__(self):
        res = "Class: FiniteElementSpace\n"
        res += "Mesh: " + str(self.mesh.nodes) + " nodes, " + str(self.mesh.elements) + "elements\n"
        res += "FunctionSpaces:\n"
        for v in self._vars:
            res += str(v)
        return res


class CompressedFESpace(FiniteElementSpace):
    def __init__(self, parent, symbols):
        self._symbols = symbols
        self._mesh = parent.mesh
        self._vars = parent.variables

        self._fes = prod([v.eval(self._mesh) for v in self._vars if v.name in symbols])
        self._tnt = self._TnT_dict([v for v in self._vars if v.name in symbols], self._fes)
        self._mask = self.__mask(parent._fes, symbols)

    def __mask(self, fes, symbols):
        dofs = ngsolve.BitArray(fes.FreeDofs())
        n = 0
        for v in self.variables:
            for j in range(n, n+v.size):
                dofs[fes.Range(j)]=v.name in symbols
            n += v.size
        return dofs

    @property
    def mask(self):
        return self._mask
    
    @property
    def symbols(self):
        return self._symbols
    

class _LinearForm:
    def __init__(self, fes, wf):
        self._obj = ngsolve.LinearForm(fes._fes)
        if wf.valid:
            self._obj += wf.eval(fes)
        self._init = False
        self._tdep = wf.isTimeDependent

    def update(self):
        if not self._init or self._tdep:
            self._obj.Assemble()
            self._init = True

    def __add__(self, other):
        return other + self._obj.vec
    
    def __radd__(self, other):
        return self+other

    @property
    def isTimeDependent(self):
        return self._tdep


class _BilinearForm:
    def __init__(self, fes, wf, condense=False, symmetric=False):
        self._obj = ngsolve.BilinearForm(fes._fes, condense=condense, symmetric=symmetric)
        if wf.valid:
            self._obj += wf.eval(fes)
        self._init = False
        self._tdep = wf.isTimeDependent
        self._nl = wf.isNonlinear

    def update(self):
        if (self._tdep or not self._init) and not self._nl:
            self._obj.Assemble()
            self._init = True
            return True
        return False
    
    def linearize(self, x):
        if self._nl:
            self._obj.AssembleLinearization(x)
            return True
        return False

    def __mul__(self, x):
        if self._nl or self._obj.condense:
            return self._obj.Apply(x)
        else:
            return self._obj.mat * x

    @property
    def condense(self):
        return self._obj.condense
    
    @property
    def mat(self):
        return self._obj.mat
    
    @property
    def harmonic_extension(self):
        return self._obj.harmonic_extension

    @property
    def harmonic_extension_trans(self):
        return self._obj.harmonic_extension_trans

    @property
    def inner_solve(self):
        return self._obj.inner_solve
    
    @property
    def isTimeDependent(self):
        return self._tdep
    
    @property
    def isNonlinear(self):
        return self._nl


class GridFunction(ngsolve.GridFunction):
    def __init__(self, fes, value=None):
        super().__init__(fes._fes)
        self._fes = fes
        if isinstance(value, NGSFunctionBase):
            value = [value]
        if value is not None:
            self._set(value)

    def _set(self, value):
        if self.isSingle:
            self.Set(*[v.eval(self._fes) for v in value])
        else:
            value = [val if v.isScalar else val[i] for v, val in zip(self._fes.variables, value) for i in range(v.size)]
            for ui, i in zip(self.components, value):
                ui.Set(i.eval(self._fes))

    def setComponent(self, var, value):
        value = value.eval(self._fes)
        if self.isSingle:
            if var.isScalar:
                self.Set(value)
            else:
                self.Set(value[:var.size])
        else:
            n = 0
            for v in self._fes.variables:
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

    @property
    def finiteElementSpace(self):
        return self._fes