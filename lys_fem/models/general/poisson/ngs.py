from lys_fem.ngs import NGSModel, dx, grad, det
from .. import Source, DivSource


class NGSPoissonModel(NGSModel):
    def __init__(self, model, coef=None):
        super().__init__(model)
        self._model = model
        self._coef = coef

    def weakform(self, vars, mat):
        wf = 0
        u, v = vars[self._model.variableName]

        if self._model.coords=="cartesian":
            J = mat[self._model.jacobian]
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

            for s in self._model.domainConditions.get(Source):
                f = mat[s.values]
                wf += f*v*dx(s.geometries)
            
            for s in self._model.domainConditions.get(DivSource):
                f = mat[s.values]
                wf += -f.dot(grad(v))*dx(s.geometries)
        else:
            gu = grad(u)
            gv = grad(v)
            wf += gu.dot(gv) * mat["x"] * dx
            return wf
        return wf
    