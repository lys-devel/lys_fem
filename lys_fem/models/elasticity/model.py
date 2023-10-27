from lys.Qt import QtWidgets
from lys_fem import FEMModel, FEMSolution


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

    def eval(self, data, fem, var):
        if var == "ux":
            return data["u"][:, 0]
        if var == "uy":
            return data["u"][:, 1]
        if var == "uz":
            return data["u"][:, 2]

    def widget(self, fem, canvas):
        return _ElasticityWidget(self)

    def resultWidget(self, *args, **kwargs):
        return _ElasticSolutionWidget(*args, **kwargs)


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
    def __init__(self, fem, canvas, path, solver, model):
        super().__init__()
        self._fem = fem
        self._canvas = canvas
        self._path = path
        self._solver = solver
        self._model = model
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
        sol = FEMSolution(self._path)
        return sol.eval(var, model=self._model, data_number=self._time.value(), solver=self._solver)
