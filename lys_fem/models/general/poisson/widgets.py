from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector


class PoissonEquationWidget(QtWidgets.QWidget):
    def __init__(self, eq, fem, canvas):
        super().__init__()
        self._eq = eq
        self.__initLayout(eq, fem, canvas)

    def __initLayout(self, eq, fem, canvas):
        self._geom = GeometrySelector(canvas, fem, eq.geometries)
        self._J = QtWidgets.QLineEdit()
        if eq.J is not None:
            self._J.setText(str(eq.J))
        self._J.textChanged.connect(self.__change)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(QtWidgets.QLabel("Jacobian"))
        h.addWidget(self._J)

        v = QtWidgets.QVBoxLayout()
        v.addWidget(self._geom)
        v.addLayout(h)
        self.setLayout(v)

    def __change(self):
        if len(self._J.text())==0:
            self._eq.set("J", None)
        else:
            self._eq.set("J", self._J.text())
