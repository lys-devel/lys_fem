import numpy as np
from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, ScalarFunctionWidget, VectorFunctionWidget, MatrixFunctionWidget


class ConditionWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas, title):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas, title)

    def __initlayout(self, fem, canvas, title):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        shape = np.array(self._cond.values).shape
        if len(shape) == 0:
            self._value = ScalarFunctionWidget(title, self._cond.values, valueChanged=self.__valueChanged)
        if len(shape) == 1:
            self._value = VectorFunctionWidget(title, self._cond.values, valueChanged=self.__valueChanged)
        if len(shape) == 2:
            self._value = MatrixFunctionWidget(title, self._cond.values, valueChanged=self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self._cond.values = vector


class EquationWidget(GeometrySelector):
    def __init__(self, eq, fem, canvas):
        super().__init__(canvas, fem, eq.geometries)
