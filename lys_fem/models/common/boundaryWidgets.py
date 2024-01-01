from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, VectorFunctionWidget


class DirichletBoundaryWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._fix = [QtWidgets.QCheckBox(axis, toggled=self.__toggled) for axis in ["x", "y", "z"][:len(self._cond.values)]]

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Constrain"))
        for w, b in zip(self._fix, self._cond.values):
            w.setChecked(b)
            h.addWidget(w)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(h)
        self.setLayout(layout)

    def __toggled(self):
        self._cond.values = [w.isChecked() for w in self._fix]
