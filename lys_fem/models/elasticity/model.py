import numpy as np
from lys import Wave
from lys.Qt import QtWidgets

from lys_fem import FEMModel


class ElasticModel(FEMModel):
    def __init__(self, nvar=3, *args, **kwargs):
        super().__init__(nvar, *args, **kwargs)

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    @property
    def variableName(self):
        return "u"

    def widget(self, fem, canvas):
        return _ElasticityWidget(self)

    def resultWidget(self, fem, canvas, path):
        return _ElasticSolutionWidget(fem, canvas, path)

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


class _ElasticSolutionWidget(QtWidgets.QWidget):
    def __init__(self, fem, canvas, path):
        super().__init__()
        self._fem = fem
        self._canvas = canvas
        self._path = path
        self.__initlayout()

    def __initlayout(self):
        self._list = QtWidgets.QComboBox()
        self._list.addItems(["ux", "uy", "uz"])

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("Show", clicked=self.__show))

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def __show(self):
        data = self.__loadData()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            for w in data:
                self._canvas.append(w)

    def __loadData(self):
        var = self._list.currentText()
        data = np.load(self._path + "/stationary.npz")
        mesh = np.load(self._path + "/stationary_mesh.npz")
        if var == "ux":
            val = data["u"][:, 0]
        if var == "uy":
            val = data["u"][:, 1]
        if var == "uz":
            val = data["u"][:, 2]
        res = []
        i = 0
        while 'coords' + str(i) in mesh:
            values = val[mesh["nodes" + str(i)]]
            coords = mesh["coords" + str(i)]
            keys_valid = [key for key in ["point", "line", "triangle", "quad", "tetra", "hexa", "prism", "pyramid"] if key + str(i) in mesh]
            elems = {key: mesh[key + str(i)] for key in keys_valid}
            res.append(Wave(values, coords, elements=elems))
            i += 1
        return res
