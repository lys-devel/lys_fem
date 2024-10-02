import numpy as np
from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, ScalarFunctionWidget, VectorFunctionWidget, MatrixFunctionWidget


class ConditionWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas, title, shape=None):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas, title, shape)

    def __initlayout(self, fem, canvas, title, shape):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Value"))

        if self._cond.values is not None:
            if shape is None:
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
        if self._cond.values is not None:
            layout.addWidget(self._value)
        
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self._cond.values = vector

    def __comboChanged(self, text):
        if text == "Expression":
            self._value.show()
            self._calc.hide()
            self._cond.values = self._value.value()
        else:
            self._value.hide()
            self._calc.show()
            self._cond.values = self._calc.value



class EquationWidget(GeometrySelector):
    def __init__(self, eq, fem, canvas):
        super().__init__(canvas, fem, eq.geometries)
