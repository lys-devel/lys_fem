from lys.Qt import QtWidgets
from .geometry import GeometrySelection
from lys_fem.widgets import GeometrySelector


class BoundaryCondition:
    def __init__(self, name, boundaries=None):
        self._name = name
        if isinstance(boundaries, GeometrySelection):
            self._bdrs = boundaries
        else:
            self._bdrs = GeometrySelection("Surface", boundaries)

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
        return _DirichletBoundaryWidget(self, fem, canvas)

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


class NeumannBoundary:
    def __init__(self, name, values, boundaries=None):
        super().__init__(name, boundaries)
        self._value = values

    @classmethod
    @property
    def name(cls):
        return "Neumann Boundary"

    def widget(self, fem, canvas):
        return _NeumannBoundaryWidget(self, fem, canvas)

    @property
    def values(self):
        return self._value

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "values": self._value, "bdrs": self.boundaries.saveAsDictionary()}

    @staticmethod
    def loadFromDictionary(d):
        bdrs = GeometrySelection.loadFromDictionary(d["bdrs"])
        return NeumannBoundary(d["name"], d["values"], boundaries=bdrs)


class _DirichletBoundaryWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.boundaries)
        self._fix = [QtWidgets.QCheckBox(axis, toggled=self.__toggled) for axis in ["x", "y", "z"][:len(self._cond.components)]]

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Constrain"))
        for w, b in zip(self._fix, self._cond.components):
            w.setChecked(b)
            h.addWidget(w)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(h)
        self.setLayout(layout)

    def __toggled(self):
        self._cond.components = [w.isChecked() for w in self._fix]


class _NeumannBoundaryWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.boundaries)

        grid = QtWidgets.QGridLayout()
        for i, val in enumerate(self._cond.values):
            w = QtWidgets.QLineEdit()
            w.setText(val)
            w.textChanged.connect(lambda text, index=i: self.__setValue(text, index))
            grid.addWidget(QtWidgets.QLabel("val" + str(i)), i, 0)
            grid.addWidget(w, i, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(grid)
        self.setLayout(layout)

    def __setValue(self, text, index):
        self._cond.values[index] = text
