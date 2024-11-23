from lys_fem.fem import BoundaryCondition

class NeumannBoundary(BoundaryCondition):
    className = "Neumann Boundary"

    @classmethod
    def default(cls, fem, model):
        return NeumannBoundary([0]*model.variableDimension())


class DirichletBoundary(BoundaryCondition):
    className = "Dirichlet Boundary"

    @classmethod
    def default(cls, fem, model):
        return DirichletBoundary([True]*model.variableDimension())

    def widget(self, fem, canvas):
        from .boundaryWidgets import DirichletBoundaryWidget
        return DirichletBoundaryWidget(self, fem, canvas)
