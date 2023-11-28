import sympy as sp

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget, MatrixFunctionWidget


class HeatConductionParameters(FEMParameter):
    def __init__(self, C_v=1, k=sp.eye(3)):
        self.C_v = sp.Float(C_v)
        self.k = k

    @classmethod
    @property
    def name(cls):
        return "Heat Conduction"

    def getParameters(self, dim):
        if dim == 1:
            return {"C_v": self.C_v, "k": self.k[0, 0]}
        else:
            return {"C_v": self.C_v, "k": self.k[:dim, :dim]}

    def widget(self):
        return _HeatConductionWidget(self)


class _HeatConductionWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self._C_v = ScalarFunctionWidget("Heat capacity C_v", self._param.C_v, valueChanged=self.__set)
        self._k = MatrixFunctionWidget("Heat conduction k", self._param.k, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._C_v)
        layout.addWidget(self._k)
        self.setLayout(layout)

    def __set(self):
        self._param.C_v = self._C_v.value()
        self._param.k = self._k.value()
