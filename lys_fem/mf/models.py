from . import mfem
from .coef import generateCoefficient
from ..fem import DirichletBoundary, NeumannBoundary

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fem, geom, mesh, mat):
    models = []
    for m in fem.models:
        model = modelList[m.name](m, mesh, mat)

        # Initial conditions
        attrs = [tag for dim, tag in geom.getEntities(fem.dimension)]
        coefs = {}
        for init in m.initialConditions:
            for d in attrs:
                if init.domains.check(d):
                    coefs[d] = init.values
        c = generateCoefficient(coefs, fem.dimension)
        model.setInitialValue(c)

        # Dirichlet Boundary conditions
        bdr_attrs = [tag for dim, tag in geom.getEntities(fem.dimension - 1)]
        bdr_dir = {i: [] for i in range(fem.dimension)}
        for b in m.boundaryConditions:
            if isinstance(b, DirichletBoundary):
                for axis, check in enumerate(b.components):
                    if check:
                        bdr_dir[axis].extend(b.boundaries.getSelection())
        model.setDirichletBoundary(bdr_dir)

        # Neumann Boundary conditions
        bdr_stress = {}
        for b in m.boundaryConditions:
            if isinstance(b, NeumannBoundary):
                for d in bdr_attrs:
                    if b.boundaries.check(d):
                        bdr_stress[d] = b.values
        if len(bdr_stress) != 0:
            c = generateCoefficient(bdr_stress, fem.dimension)
            model.setNeumannBoundary(c)
        models.append(model)
    return models


class MFEMModel:
    def __init__(self, fec, model, mesh, vDim=None):
        if vDim is None:
            vDim = model.variableDimension()
        self._fec = fec
        self._fespace = mfem.FiniteElementSpace(mesh, self._fec, vDim, mfem.Ordering.byVDIM)
        self._model = model
        # initial values
        self._x_gf = mfem.GridFunction(self._fespace)
        self._xt_gf = mfem.GridFunction(self._fespace)
        self._x_gf.Assign(0.0)
        self._xt_gf.Assign(0.0)

        # boundary conditions
        self._dirichlet = None
        self._neumann = None

    def setInitialValue(self, x=None, xt=None):
        if x is not None:
            self._x_gf.ProjectCoefficient(x)
        if xt is not None:
            self._xt_gf.ProjectCoefficient(xt)

    def getInitialValue(self):
        return self._x_gf, self._xt_gf

    def setDirichletBoundary(self, dirichlet_dict):
        self._dirichlet = dirichlet_dict

    def setNeumannBoundary(self, condition):
        self._neumann = condition

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
        self._initM = False
        self._initK = False
        self._initB = False
        self._initX0 = False
        self._initXt0 = False
        self._initTi = False
        self._initMi = False

    def update(self, x):
        pass

    @property
    def M(self):
        if not self._initM:
            self._m = self.assemble_m()
            self._M = mfem.SparseMatrix()
            self.ess_tdof_list = self.essential_tdof_list()
            self._m.FormSystemMatrix(self.ess_tdof_list, self._M)
            self._initM = True
        return self._M

    @property
    def grad_Mx(self):
        return self.M

    @property
    def K(self):
        if not self._initK:
            self._k = self.assemble_a()
            self._K = mfem.SparseMatrix()
            self.ess_tdof_list = self.essential_tdof_list()
            self._k.FormSystemMatrix(self.ess_tdof_list, self._K)
            self._initK = True
        return self._K

    @property
    def grad_Kx(self):
        return self.K

    @property
    def b(self):
        if not self._initB:
            self._b = self.assemble_b()
            self.ess_tdof_list = self.essential_tdof_list()
            tmp = mfem.GridFunction(self.space, self._b)
            self._B = mfem.Vector()
            tmp.GetTrueDofs(self._B)
            K = self.K
            self._k.EliminateVDofsInRHS(self.ess_tdof_list, self.x0, self._B)
            self._initB = True
        return self._B

    def grad_b(self):
        return None

    @property
    def x0(self):
        if not self._initX0:
            self._x, _ = self.getInitialValue()
            self.ess_tdof_list = self.essential_tdof_list()
            self._X0 = mfem.Vector()
            self._x.GetTrueDofs(self._X0)
            self._initX0 = True
        return self._X0

    @property
    def xt0(self):
        if not self._initXt0:
            _, self._xt0 = self.getInitialValue()
            self.ess_tdof_list = self.essential_tdof_list()
            self._Xt0 = mfem.Vector()
            self._xt0.GetTrueDofs(self._Xt0)
            self._initXt0 = True
        return self._Xt0

    def RecoverFEMSolution(self, X):
        self._X0 = X
        if self._initX0:
            self._x.SetFromTrueDofs(X)
        else:
            self._a.RecoverFEMSolution(X, self._b, self._x)
        return self._x


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
