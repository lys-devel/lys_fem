

from .model import models as modelList
from .coef import generateCoefficient
from ..models import DirichletBoundary, NeumannBoundary


def generateModel(fem, geom, mesh, mat):
    _modelDict = {m.name: m for m in modelList}
    models = []
    for m in fem.models:
        model = _modelDict[m.name](m, mesh, mat)

        # Initial conditions
        attrs = [tag for dim, tag in geom.getEntities(fem.dimension)]
        coefs = {}
        for init in m.initialConditions:
            for d in attrs:
                if init.domains.check(d):
                    coefs[d] = init.values
        c = generateCoefficient(coefs, geom)
        model.setInitialValue(c)

        # Dirichlet Boundary conditions
        bdr_attrs = [tag for dim, tag in geom.getEntities(fem.dimension - 1)]
        bdr_dir = {i: [] for i in range(fem.dimension - 1)}
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
            c = generateCoefficient(bdr_stress, geom, type="surface")
            model.setBoundaryStress(c)
        models.append(model)
    return models
