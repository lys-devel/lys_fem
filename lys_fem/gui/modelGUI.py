
from ..widgets import TreeItem


class ModelTree(TreeItem):
    def __init__(self, models):
        super().__init__()
        self._models = models
        self._children = [_ElasticGUI(m, self) for m in models]

    @property
    def children(self):
        return self._children


class _ElasticGUI(TreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = [_ElasticDomain(model, self), _ElasticBoundary(model, self), _InitialCondition(model, self)]

    @property
    def name(self):
        return "Elasticity"

    @property
    def children(self):
        return self._children


class _ElasticDomain(TreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = []

    @property
    def name(self):
        return "Domains"

    @property
    def children(self):
        return self._children


class _ElasticBoundary(TreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = []

    @property
    def name(self):
        return "Boundary Conditions"

    @property
    def children(self):
        return self._children


class _InitialCondition(TreeItem):
    def __init__(self, model, parent):
        super().__init__(parent)
        self._children = []

    @property
    def name(self):
        return "Initial Conditions"

    @property
    def children(self):
        return self._children
