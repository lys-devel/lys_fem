from lys.Qt import QtWidgets, QtCore
from lys_fem.widgets import GeometrySelector


class DirichletBoundary:
    def __init__(self, name, boundaries=[]):
        self._name = name
        self._bdrs = boundaries

    @property
    def objName(self):
        return self._name

    @classmethod
    @property
    def name(cls):
        return "Dirichlet Boundary"

    def widget(self, fem, canvas):
        return _DirichletBoundaryWidget(self, fem, canvas)

    @property
    def boundaries(self):
        return self._bdrs

    @boundaries.setter
    def boundaries(self, value):
        self._bdrs = value

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "bdrs": self.boundaries}

    @staticmethod
    def loadFromDictionary(d):
        return DirichletBoundary(d["name"], boundaries=d["bdrs"])


class NeumannBoundary:
    def __init__(self, nvar):
        self._value = ["0"] * nvar

    @classmethod
    @property
    def name(cls):
        return "Neumann Boundary"


class _DirichletBoundaryWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector("Target Boundaries", fem.dimension - 1, canvas, fem, self._cond.boundaries)
        self._selector.selectionChanged.connect(self.__selectionChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        self.setLayout(layout)

    def __selectionChanged(self, bdrs):
        self._cond.boundaries = bdrs
