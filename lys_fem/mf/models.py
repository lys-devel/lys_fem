from . import mfem
from .coef import generateCoefficient
from ..fem import DirichletBoundary, NeumannBoundary

modelList = {}


def addMFEMModel(name, model):
    model.name = name
    modelList[name] = model


def generateModel(fec, fem, geom, mesh, mat):
    models = []
    for m in fem.models:
        model = modelList[m.name](fec, m, mesh, mat)

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
    def __init__(self, fec, model, mesh):
        self._fespace = mfem.FiniteElementSpace(mesh, fec, model.variableDimension(), mfem.Ordering.byVDIM)
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
        if self._dirichlet is None:
            return mfem.intArray()
        res = []
        for axis, b in self._dirichlet.items():
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

    def assemble(self):
        self._a = self.assemble_a()
        self._b = self.assemble_b()
        self._x, _ = self.getInitialValue()
        self.ess_tdof_list = self.essential_tdof_list()

        self._A = mfem.SparseMatrix()
        self._B = mfem.Vector()
        self._X = mfem.Vector()
        self._a.FormLinearSystem(self.ess_tdof_list, self._x, self._b, self._A, self._X, self._B)
        return self._A, self._B, self._X

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
    def K(self):
        if not self._initK:
            self._k = self.assemble_a()
            self._K = mfem.SparseMatrix()
            self.ess_tdof_list = self.essential_tdof_list()
            self._k.FormSystemMatrix(self.ess_tdof_list, self._K)
            self._initK = True
        return self._K

    @property
    def b(self):
        if not self._initB:
            self._b = self.assemble_b()
            self.ess_tdof_list = self.essential_tdof_list()
            self._B = mfem.Vector()
            # self.b.GetTrueDofs(self.B)
            self._initB = True
        return self._B

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

    def set_Mi(self, sol_M):
        if not self._initMi:
            sol_M.SetOperator(self.M)
        return sol_M

    def set_Ti(self, sol_T, c):
        if not self._initTi:
            self._T = mfem.SparseMatrix()
            self._T = mfem.Add(1.0, self.M, c, self.K)
            sol_T.SetOperator(self._T)
        return sol_T

    def RecoverFEMSolution(self, X):
        if self._initX0:
            self._x.SetFromTrueDofs(X)
        else:
            self._a.RecoverFEMSolution(X, self._b, self._x)
        return self._x


class MFEMNonlinearModel(MFEMModel):
    def assemble(self):
        self.x, _ = self.getInitialValue()
        self.a = self.assemble_a()
        self.b = self.assemble_b()
        self.ess_tdof_list = self.essential_tdof_list()
        B, X = self.a.initializeRHS(self.ess_tdof_list, self.x, self.b)
        return self.a, B, X

    def RecoverFEMSolution(self, X):
        self.a.RecoverFEMSolution(X, self.b, self.x)
        return self.x
