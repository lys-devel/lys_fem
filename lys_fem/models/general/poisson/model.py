from lys_fem import FEMFixedModel
from lys_fem.util import dx, grad, det
from lys_fem.models.common import Source, DivSource, DirichletBoundary, NeumannBoundary


class PoissonModel(FEMFixedModel):
    className = "Poisson"
    domainConditionTypes = [Source, DivSource]
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]

    def __init__(self, *args, coords="cartesian", J=None, coef=None, **kwargs):
        super().__init__(1, *args, varName="phi", isScalar=True, **kwargs)
        self._coords = coords
        self._jac = J
        self._coef = coef

    @property
    def coords(self):
        return self._coords
    
    @property
    def jacobian(self):
        return self._jac

    def weakform(self, vars, mat):
        wf = 0
        u, v = vars[self.variableName]

        if self.coords=="cartesian":
            J = mat[self.jacobian]
            if J is not None:
                gu = J.dot(grad(u))
                gv = J.dot(grad(v))
                d = det(J)
            else:
                gu = grad(u)
                gv = grad(v)
                d = 1

            if self._coef is not None:
                gu = mat[self._coef].dot(gu)
            wf += gu.dot(gv)/d * dx

            for s in self.domainConditions.get(Source):
                f = mat[s.value]
                wf += f*v*dx(s.geometries)
            
            for s in self.domainConditions.get(DivSource):
                f = mat[s.value]
                wf += -f.dot(grad(v))*dx(s.geometries)
        else:
            gu = grad(u)
            gv = grad(v)
            wf += gu.dot(gv) * mat["x"] * dx
            return wf
        return wf
    
    def widget(self, fem, canvas):
        from .widgets import PoissonEquationWidget
        return PoissonEquationWidget(self, fem, canvas)
