from . import mfem
from .coef import generateCoefficient
from ..fem import DirichletBoundary, NeumannBoundary, Source

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, mesh, mat):
    return [modelList[m.name](m, mesh, mat) for m in fem.models]


class MFEMModel:
    def __init__(self, fec, model, mesh, vDim=None):
        if vDim is None:
            vDim = model.variableDimension()
        self._fec = fec
        self._fespace = mfem.FiniteElementSpace(mesh, self._fec, vDim, mfem.Ordering.byVDIM)
        self._model = model

        # dirichlet boundary
        self.generateDirichletBoundaryCondition(self._fespace, [b for b in model.boundaryConditions if isinstance(b, DirichletBoundary)])

        # initial value
        c = self.generateDomainCoefficient(self._fespace, model.initialConditions)
        self.setInitialValue(c)

        # neumann boundary
        neumann = [b for b in model.boundaryConditions if isinstance(b, NeumannBoundary)]
        if len(neumann) != 0:
            c = self.generateSurfaceCoefficient(self._fespace, neumann)
            self._neumann = c
        else:
            self._neumann = None

        # source term
        source = [d for d in model.domainConditions if isinstance(d, Source)]
        if len(source) != 0:
            c = self.generateDomainCoefficient(self._fespace, source)
            self._source = c
        else:
            self._source = None


    def generateDirichletBoundaryCondition(self, space, conditions):
        bdr_dir = {i: [] for i in range(space.GetMesh().Dimension())}
        for b in conditions:
            for axis, check in enumerate(b.components):
                if check:
                    bdr_dir[axis].extend(b.boundaries.getSelection())
        self._dirichlet = bdr_dir

    def generateDomainFunction(self, space, conditions):
        c = self.generateDomainCoefficient(space, conditions)
        gf = mfem.GridFunction(space)
        gf.ProjectCoefficient(c)
        return gf

    def generateDomainCoefficient(self, space, conditions):
        coefs = {}
        for c in conditions:
            for d in space.GetMesh().attributes:
                if c.domains.check(d):
                    coefs[d] = c.values
        return generateCoefficient(coefs, space.GetMesh().Dimension())
    
    def generateSurfaceCoefficient(self, space, conditions):
        bdr_stress = {}
        for b in conditions:
            for d in space.GetMesh().bdr_attributes:
                if b.boundaries.check(d):
                    bdr_stress[d] = b.values
        return generateCoefficient(bdr_stress, space.GetMesh().Dimension())

    def setInitialValue(self, x=None, xt=None):
        self._x_gf = mfem.GridFunction(self._fespace)
        self._xt_gf = mfem.GridFunction(self._fespace)
        if x is not None:
            self._x_gf.ProjectCoefficient(x)
        else:
            self._x_gf.Assign(0.0)
        if xt is not None:
            self._xt_gf.ProjectCoefficient(xt)
        else:
            self._xt_gf.Assign(0.0)

    def getInitialValue(self):
        return self._x_gf, self._xt_gf

    def essential_tdof_list(self):
        res = []
        for axis, b in self._dirichlet.items():
            if len(b) == 0:
                continue
            ess_bdr = mfem.intArray(self.space.GetMesh().bdr_attributes.Max())
            ess_bdr.Assign(0)
            for i in b:
                ess_bdr[i - 1] = 1
            ess_tdof_list = mfem.intArray()
            self.space.GetEssentialTrueDofs(ess_bdr, ess_tdof_list, axis)
            res.extend([i for i in ess_tdof_list])
        return mfem.intArray(res)

    def assemble_b(self):
        b = mfem.LinearForm(self.space)
        if self._neumann is not None:
            b.AddBoundaryIntegrator(mfem.VectorBoundaryLFIntegrator(self._neumann))
        if self._source is not None:
            b.AddDomainIntegrator(mfem.VectorDomainLFIntegrator(self._source))
        b.Assemble()
        return b

    @property
    def space(self):
        return self._fespace

    @property
    def variableName(self):
        return self._model.variableName

    @property
    def preconditioner(self):
        return None
    
    @property
    def timeUnit(self):
        return 1

class MFEMLinearModel(MFEMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize()

    def initialize(self):
        self.ess_tdof_list = self.essential_tdof_list()

        self._X0 = mfem.Vector()
        self._Xt0 = mfem.Vector()
        self._x, self._xt0 = self.getInitialValue()
        self._x.GetTrueDofs(self._X0)
        self._xt0.GetTrueDofs(self._Xt0)

        self._m = self.assemble_m()
        if self._m is not None:
            self._M = mfem.SparseMatrix()
            self._m.FormSystemMatrix(self.ess_tdof_list, self._M)
        else:
            self._M = None

        self._k = self.assemble_a()
        if self._k is not None:
            self._K = mfem.SparseMatrix()
            self._k.FormSystemMatrix(self.ess_tdof_list, self._K)
        else:
            self._K = None

        self._b = self.assemble_b()
        if self._b is not None:
            self._B = mfem.Vector()
            mfem.GridFunction(self.space, self._b).GetTrueDofs(self._B)
            self._k.EliminateVDofsInRHS(self.ess_tdof_list, self.x0, self._B)
        else:
            self._B = None

    def RecoverFEMSolution(self, X):
        self._X0 = X
        self._x.SetFromTrueDofs(X)
        return self._x
    
    def assemble_a(self):
        return None

    def assemble_m(self):
        return None

    def update(self, x):
        pass

    @property
    def M(self):
        return self._M

    @property
    def grad_Mx(self):
        return self.M

    @property
    def K(self):
        return self._K

    @property
    def grad_Kx(self):
        return self.K

    @property
    def b(self):
        return self._B

    @property
    def grad_b(self):
        return None

    @property
    def x0(self):
        return self._X0

    @property
    def xt0(self):
        return self._Xt0


class MFEMNonlinearModel(MFEMModel):
    def update(self, x):
        self.ess_tdof_list = self.essential_tdof_list()
        self._k = self.assemble_a()
        self._k.assemble(x, self.ess_tdof_list)
        self._b = self.assemble_b()

        self._K = self._k.A
        self._DK = self._k.DA
        self._B = self._k.EliminateVDofsInRHS(self.ess_tdof_list, x, self._b)

    def RecoverFEMSolution(self, X):
        self._x.SetFromTrueDofs(X)
        return self._x

    @property
    def K(self):
        return self._K

    @property
    def grad_Kx(self):
        return self._DK

    @property
    def x0(self):
        self._x, _ = self.getInitialValue()
        self._X0 = mfem.Vector()
        self._x.GetTrueDofs(self._X0)
        return self._X0

    @property
    def b(self):
        return self._B

    def grad_b(self):
        return None
