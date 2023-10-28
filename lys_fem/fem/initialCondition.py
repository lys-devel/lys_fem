from lys.Qt import QtWidgets, QtCore
from .geometry import GeometrySelection
from lys_fem.widgets import GeometrySelector, VectorFunctionWidget


class InitialCondition:
    def __init__(self, name, value, domains=None):
        if isinstance(domains, GeometrySelection):
            self._domains = domains
        else:
            self._domains = GeometrySelection("Domain", domains)
        self._name = name
        self._value = [str(v) for v in value]

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
        return InitialCondition(d["name"], value=values, domains=domain)


class _InitialConditionWidget(QtWidgets.QWidget):
    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, init, fem, canvas):
        super().__init__()
        self._init = init
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._init.domains)
        self._value = VectorFunctionWidget("Initial values", self._init.values, valueChanged=self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self.valueChanged.emit(vector)
