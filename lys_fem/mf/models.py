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
