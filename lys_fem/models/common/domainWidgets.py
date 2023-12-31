from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, VectorFunctionWidget


class SourceWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._value = VectorFunctionWidget("Initial values", self._cond.values, valueChanged=self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self._cond.values = vector