from lys.Qt import QtWidgets
from lys_fem import FEMModel

from ..Common import InitialCondition


class ElasticModel(FEMModel):
    def __init__(self, nvar=3, init=None):
        self._nvar = nvar
        if init is None:
            init = [InitialCondition("Default", self._nvar, domains="all")]
        self._init = init

    def setVariableDimension(self, dim):
        self._nvar = dim

    def variableDimension(self):
        return self._nvar

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    @classmethod
    @property
    def initialConditionTypes(self):
        return [InitialCondition]

    @property
    def initialConditions(self):
        return self._init

    def widget(self, fem, canvas):
        return _ElasticityWidget(self)

    def addInitialCondition(self, init):
        if init:
            i = 1
            while "Initial value" + str(i) in [c.objName for c in self._init]:
                i += 1
            name = "Initial value" + str(i)
            init = init(name, self._nvar)
            self._init.append(init)
        return init

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["nvar"] = self._nvar
        d["init"] = [i.saveAsDictionary() for i in self._init]
        return d

    @staticmethod
    def loadFromDictionary(d):
        nvar = d["nvar"]
        init = [InitialCondition.loadFromDictionary(dic) for dic in d["init"]]
        return ElasticModel(nvar, init)


class _ElasticityWidget(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self._model = model
        self.__initlayout()

    def __initlayout(self):
        self._dim = QtWidgets.QSpinBox()
        self._dim.setRange(1, 3)
        self._dim.setValue(self._model.variableDimension())
        self._dim.valueChanged.connect(self.__changeDim)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("Variable dimension"))
        layout.addWidget(self._dim)
        self.setLayout(layout)

    def __changeDim(self, value):
        self._model.setVariableDimension(self._dim.value())
