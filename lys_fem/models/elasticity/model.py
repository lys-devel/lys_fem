import os
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
        self._time = QtWidgets.QSpinBox()
        self._time.setRange(0, 10000000)
        self._time.valueChanged.connect(self.__show)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(QtWidgets.QPushButton("Show", clicked=self.__show))

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        layout.addWidget(self._time)
        layout.addLayout(buttons)
        self.setLayout(layout)

    def __show(self):
        data = self.__loadData()
        with self._canvas.delayUpdate():
            self._canvas.clear()
            for w in data:
                o = self._canvas.append(w)
                o.showEdges(True)

    def __loadData(self):
        var = self._list.currentText()
        if os.path.exists(self._path + "/stationary.npz") and False:
            data = np.load(self._path + "/stationary.npz")
            mesh = np.load(self._path + "/stationary_mesh.npz")
        else:
            data = np.load(self._path + "/tdep" + str(self._time.value()) + ".npz")
            i = 0
            meshes = []
            while os.path.exists(self._path + "/tdep_mesh" + str(i) + ".npz"):
                meshes.append(np.load(self._path + "/tdep_mesh" + str(i) + ".npz"))
                i += 1
            mesh = np.load(self._path + "/tdep_mesh.npz")
        if var == "ux":
            val = data["u"][:, 0]
        if var == "uy":
            val = data["u"][:, 1]
        if var == "uz":
            val = data["u"][:, 2]
        res = []
        for mesh in meshes:
            keys = ["point", "line", "triangle", "quad", "tetra", "hexa", "prism", "pyramid"]
            elems = {key: mesh[key] for key in keys if key in mesh}
            res.append(Wave(val[mesh["nodes"]], mesh["coords"], elements=elems))
        return res
