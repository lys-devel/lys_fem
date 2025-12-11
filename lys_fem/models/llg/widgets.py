from lys.Qt import QtWidgets
from lys_fem.widgets import ScalarFunctionWidget, VectorFunctionWidget


class ThermalFluctuationWidget(QtWidgets.QWidget):
    def __init__(self, cond):
        super().__init__()
        self._cond = cond
        self.__initLayout(cond)

    def __initLayout(self, cond):
        self._T = ScalarFunctionWidget("Temperature (K)", cond.T)
        self._T.valueChanged.connect(self.__change)
        self._R = VectorFunctionWidget("Random Field", cond.R)
        self._R.valueChanged.connect(self.__change)
        g = QtWidgets.QVBoxLayout()
        g.addWidget(self._T)
        g.addWidget(self._R)
        self.setLayout(g)

    def __change(self):
        self._cond.T = self._T.value()
        self._cond.R = self._R.value()
