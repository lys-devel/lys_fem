from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, VectorFunctionWidget


class InitialConditionWidget(QtWidgets.QWidget):
    def __init__(self, init, fem, canvas):
        super().__init__()
        self._init = init
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._init.geometries)
        self._value = VectorFunctionWidget("Initial values", self._init.values, valueChanged=self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self._init.values = vector


class EquationWidget(GeometrySelector):
    def __init__(self, eq, fem, canvas):
        super().__init__(canvas, fem, eq.geometries)
