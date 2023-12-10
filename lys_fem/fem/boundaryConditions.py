from .base import FEMObject, FEMObjectList
from .geometry import GeometrySelection


class BoundaryConditions(FEMObjectList):
    def get(self, cls):
        return [condition for condition in self if isinstance(condition, cls)]

    def have(self, cls):
        return len(self.get(cls)) > 0

class BoundaryCondition(FEMObject):
    def __init__(self, name, boundaries=None):
        self._name = name
        if isinstance(boundaries, GeometrySelection):
            self._bdrs = boundaries
        else:
            self._bdrs = GeometrySelection("Surface", boundaries)
        self._bdrs.setParent(self)

    @property
    def objName(self):
        return self._name

    @property
    def boundaries(self):
        return self._bdrs

    @boundaries.setter
    def boundaries(self, value):
        self._bdrs = value


class DirichletBoundary(BoundaryCondition):
    def __init__(self, name, components, boundaries=None):
        super().__init__(name, boundaries)
        self._comps = components

    @classmethod
    @property
    def name(cls):
        return "Dirichlet Boundary"

    def widget(self, fem, canvas):
        from ..gui import DirichletBoundaryWidget
        return DirichletBoundaryWidget(self, fem, canvas)

    @property
    def components(self):
        return self._comps

    @components.setter
    def components(self, value):
        self._comps = value

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "bdrs": self.boundaries.saveAsDictionary(), "comps": self._comps}

    @staticmethod
    def loadFromDictionary(d):
        bdrs = GeometrySelection.loadFromDictionary(d["bdrs"])
        return DirichletBoundary(d["name"], boundaries=bdrs, components=d.get("comps"))


class NeumannBoundary(BoundaryCondition):
    def __init__(self, name, values, boundaries=None):
        super().__init__(name, boundaries)
        self._value = values

    @classmethod
    @property
    def name(cls):
        return "Neumann Boundary"

    def widget(self, fem, canvas):
        from ..gui import NeumannBoundaryWidget
        return NeumannBoundaryWidget(self, fem, canvas)

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "values": self._value, "bdrs": self.boundaries.saveAsDictionary()}

    @staticmethod
    def loadFromDictionary(d):
        bdrs = GeometrySelection.loadFromDictionary(d["bdrs"])
        return NeumannBoundary(d["name"], d["values"], boundaries=bdrs)

    @property
    def values(self):
        return self._value

    @values.setter
    def values(self, value):
        self._value = value
