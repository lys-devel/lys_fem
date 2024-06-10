import numpy as np

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget


class ElectrostaticParameters(FEMParameter):
    name = "Electrostatics"
    units = {"eps": "F/m"}
    def __init__(self, eps_r=np.eye(3)):
        self.eps_r = eps_r

    def getParameters(self, dim):
        res = {"eps": (np.array(self.eps_r)*8.8541878128e-12)[:dim,:dim].tolist()}
        return res

    def widget(self):
        self._eps = ScalarFunctionWidget("Relative permittivity", self._param.eps_r, valueChanged=self.__set)
        return self._eps

    def __set(self):
        self._param.eps_r = self._eps.value()
