from lys_fem import FEMFixedModel
from lys_fem.util import grad, dx, ds
from . import DirichletBoundary, NeumannBoundary, InitialCondition

class HeatConductionModel(FEMFixedModel):
    className = "Heat Conduction"
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]
    initialConditionTypes = [InitialCondition]

    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, varName="T", isScalar=True, **kwargs)

    def weakform(self, vars, mat):
        Cv, k = mat["C_v"], mat["k"]
        wf = 0
        u, v = vars[self.variableName]

        wf += Cv * u.t * v * dx + grad(u).dot(k.dot(grad(v))) * dx

        for n in self.boundaryConditions.get(NeumannBoundary):
            f = mat[n.value]
            wf -= f * v * ds(n.geometries)

        return wf