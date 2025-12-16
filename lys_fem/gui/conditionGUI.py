from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, ParameterEditor


class ConditionWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._params = ParameterEditor(self._cond)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        if len(self._cond) > 0:
            layout.addWidget(self._params)

