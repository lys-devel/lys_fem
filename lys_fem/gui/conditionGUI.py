import numpy as np
from lys.Qt import QtWidgets
from lys_fem.fem import CalculatedResult
from lys_fem.widgets import GeometrySelector, ScalarFunctionWidget, VectorFunctionWidget, MatrixFunctionWidget, CalculatedResultWidget


class ConditionWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas, title, computed=False, shape=None):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas, title, computed, shape)
        if isinstance(self._cond.values, CalculatedResult):
            self._combo.setCurrentText("Calculated")

    def __initlayout(self, fem, canvas, title, computed, shape):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._combo = QtWidgets.QComboBox()
        self._combo.addItems(["Expression", "Calculated"])
        self._combo.currentTextChanged.connect(self.__comboChanged)
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("Value"))
        h1.addWidget(self._combo)

        if isinstance(self._cond.values, CalculatedResult):
            value = np.zeros(shape)
            calc = self._cond.values
        else:
            value = self._cond.values
            calc = CalculatedResult()

        if shape is None:
            shape = np.array(self._cond.values).shape
        if len(shape) == 0:
            self._value = ScalarFunctionWidget(title, value, valueChanged=self.__valueChanged)
        if len(shape) == 1:
            self._value = VectorFunctionWidget(title, value, valueChanged=self.__valueChanged)
        if len(shape) == 2:
            self._value = MatrixFunctionWidget(title, value, valueChanged=self.__valueChanged)

        self._calc = CalculatedResultWidget(calc, valueChanged=self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)

        if computed:
            layout.insertLayout(1, h1)
            layout.addWidget(self._calc)
            self._calc.hide()
        
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
