from lys_fem import FEMFixedModel
from lys_fem.models.common import Source, DivSource, DirichletBoundary, NeumannBoundary



class PoissonModel(FEMFixedModel):
    className = "Poisson"
    domainConditionTypes = [Source, DivSource]
    boundaryConditionTypes = [DirichletBoundary, NeumannBoundary]

    def __init__(self, *args, coords="cartesian", J=None, **kwargs):
        super().__init__(1, *args, varName="phi", isScalar=True, **kwargs)
        self._coords = coords
        self._jac = J

    @property
    def coords(self):
        return self._coords
    
    @property
    def jacobian(self):
        return self._jac

    def widget(self, fem, canvas):
        raise RuntimeError("Widget for poisson should be reimplemented.")
        from .widgets import PoissonEquationWidget
        return PoissonEquationWidget(self, fem, canvas)

