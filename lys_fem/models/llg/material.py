from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget


class LLGParameters(FEMParameter):
    def __init__(self, alpha=0, Ms=8e5, Aex=1e-11, Ku=[0,0,1]):
        self.alpha = alpha
        self.Ms = Ms
        self.Aex = Aex
        self.Ku = Ku

    @classmethod
    @property
    def name(cls):
        return "LLG"

    @classmethod
    @property
    def units(cls):
        return {"alpha": 1, "M_s": "A/m", "A_ex": "J/m", "Ku": "J/m^3"}

    def getParameters(self, dim):
        return {"alpha": self.alpha, "M_s": self.Ms, "A_ex":self.Aex, "Ku": self.Ku}

    def widget(self):
        return _LLGParamsWidget(self)


class _LLGParamsWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._alpha = ScalarFunctionWidget("Gilbert damping const.", self._param.alpha, valueChanged=self.__set)
        self._Ms = ScalarFunctionWidget("Saturation Magnetization (A/m)", self._param.Ms, valueChanged=self.__set)
        self._Aex = ScalarFunctionWidget("Exchange constant (J/m)", self._param.Aex, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._Ms)
        layout.addWidget(self._Aex)
        layout.addWidget(self._alpha)
        self.setLayout(layout)

    def __set(self):
        self._param.alpha = self._alpha.value()
        self._param.Ms = self._Ms.value()
        self._param.Aex = self._Aex.value()
