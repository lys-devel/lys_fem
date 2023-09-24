from lys.Qt import QtWidgets

from ..Common import BaseModel


class ElasticModel(BaseModel):
    def __init__(self, nvar=3, *args, **kwargs):
        super().__init__(nvar, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    def widget(self, fem, canvas):
        return _ElasticityWidget(self)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["nvar"] = self._nvar
        return d

    @classmethod
    def loadFromDictionary(cls, d):
        nvar = d["nvar"]
        init, bdr, domain = cls._loadConditions(d)
        return ElasticModel(nvar, init, bdr, domain)


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
