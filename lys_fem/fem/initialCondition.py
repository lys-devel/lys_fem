from lys.Qt import QtWidgets, QtCore
from .geometry import GeometrySelection
from lys_fem.widgets import GeometrySelector


class InitialCondition:
    def __init__(self, name, nvar, value=None, domains=None):
        if value is None:
            value = ["0"] * nvar
        if domains is None:
            domains = GeometrySelection("Domain")
        self._name = name
        self._value = value
        self._domains = domains

    def setDimension(self, dim):
        if len(self._value) > dim:
            self._value = self._value[:2]
        while len(self._value) < dim:
            self._value.append("0")

    @classmethod
    @property
    def name(cls):
        return "Initial Condition"

    @property
    def objName(self):
        return self._name

    @property
    def domains(self):
        return self._domains

    @property
    def values(self):
        return self._value

    @values.setter
    def values(self, value):
        self._value = value

    def widget(self, fem, canvas):
        w = _InitialConditionWidget(self, fem, canvas)
        w.valueChanged.connect(self.__valueChanged)
        return w

    def __valueChanged(self, value):
        self.values = value

    def saveAsDictionary(self):
        return {"type": self.name, "name": self.objName, "domains": self._domains.saveAsDictionary(), "values": self._value}

    @staticmethod
    def loadFromDictionary(d):
        values = d["values"]
        domain = GeometrySelection.loadFromDictionary(d["domains"])
        return InitialCondition(d["name"], len(values), value=values, domains=domain)


class _InitialConditionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, init, fem, canvas):
        super().__init__()
        self._init = init
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        dim = len(self._init.values)
        self._selector = GeometrySelector(canvas, fem, self._init.domains)
        self._widgets = [QtWidgets.QLineEdit() for _ in range(dim)]
        for wid, val in zip(self._widgets, self._init.values):
            wid.setText(val)
            wid.textChanged.connect(self.__valueChanged)
        grid = QtWidgets.QGridLayout()
        for i, wid in enumerate(self._widgets):
            grid.addWidget(QtWidgets.QLabel("Dim " + str(i)), i, 0, 1, 1)
            grid.addWidget(wid, i, 1, 1, 3)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(grid)
        self.setLayout(layout)

    def __valueChanged(self):
        self.valueChanged.emit([w.text() for w in self._widgets])
