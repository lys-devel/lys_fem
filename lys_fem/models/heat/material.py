import numpy as np

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget, MatrixFunctionWidget


class HeatConductionParameters(FEMParameter):
    def __init__(self, C_v=1.0, k=np.eye(3).tolist()):
        self.C_v = C_v
        self.k = k

    @classmethod
    @property
    def name(cls):
        return "Heat Conduction"

    @classmethod
    @property
    def units(cls):
        return {"C_v": "J/K m^3", "k": "W/m K"}

    def getParameters(self, dim):
        return {"C_v": self.C_v, "k": np.array(self.k)[:dim,:dim].tolist()}

    def widget(self):
        return _HeatConductionWidget(self)


class _HeatConductionWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self._C_v = ScalarFunctionWidget("Heat capacity C_v (J/K m^3)", self._param.C_v, valueChanged=self.__set)
        self._k = MatrixFunctionWidget("Heat conduction k (W/mK)", self._param.k, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._C_v)
        layout.addWidget(self._k)
        self.setLayout(layout)

    def __set(self):
        self._param.C_v = self._C_v.value()
        self._param.k = self._k.value()
