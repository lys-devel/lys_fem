import numpy as np
from lys_fem.fem import BoundaryCondition, Coef

class NeumannBoundary(BoundaryCondition):
    className = "Neumann Boundary"
    def __init__(self, value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self["value"] = Coef(value, shape=np.shape(value), description = "Derivative of the variable")

    @classmethod
    def default(cls, fem, model):
        return NeumannBoundary([0]*model.variableDimension)


class DirichletBoundary(BoundaryCondition):
    className = "Dirichlet Boundary"

    @classmethod
    def default(cls, fem, model):
        return DirichletBoundary([True]*model.variableDimension)

    def widget(self, fem, canvas):
        from .boundaryWidgets import DirichletBoundaryWidget
        return DirichletBoundaryWidget(self, fem, canvas)
