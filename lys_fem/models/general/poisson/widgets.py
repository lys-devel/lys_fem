from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector


class PoissonEquationWidget(QtWidgets.QWidget):
    def __init__(self, model, fem, canvas):
        super().__init__()
        self._model = model
        self.__initLayout(model, fem, canvas)

    def __initLayout(self, model, fem, canvas):
        self._geom = GeometrySelector(canvas, fem, model.geometries)
        self._J = QtWidgets.QLineEdit()
        if model.equation.J is not None:
            self._J.setText(str(model.equation.J))
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
            self._model.equation.set("J", None)
        else:
            self._model.equation.set("J", self._J.text())
