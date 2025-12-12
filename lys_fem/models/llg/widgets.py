from lys.Qt import QtWidgets
from lys_fem.widgets import ScalarFunctionWidget, VectorFunctionWidget


class ThermalFluctuationWidget(QtWidgets.QWidget):
    def __init__(self, cond):
        super().__init__()
        self._cond = cond
        self.__initLayout(cond)

    def __initLayout(self, cond):
        self._T = ScalarFunctionWidget(cond.T, label="Temperature (K)")
        self._T.valueChanged.connect(self.__change)
        self._R = VectorFunctionWidget(cond.R, label="Random Field")
        self._R.valueChanged.connect(self.__change)
        g = QtWidgets.QVBoxLayout()
        g.addWidget(self._T)
        g.addWidget(self._R)
        self.setLayout(g)

    def __change(self):
        self._cond.T = self._T.value()
        self._cond.R = self._R.value()
