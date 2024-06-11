from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget


class SemiconductorParameters(FEMParameter):
    name = "Semiconductor Drift Diffusion"
    units = {"mu_e": "m^2/V s", "mu_h": "m^2/V s", "N_d": "1/m^3", "N_a": "1/m^3", "q": "C", "k_B": "J/K", "T": "K"}
    def __init__(self, mu_n, mu_p, N_d=0.0, N_a=0.0, T=None):
        self.mu_n = mu_n
        self.mu_p = mu_p
        self.N_d = N_d
        self.N_a = N_a
        self.T = T

    def getParameters(self, dim):
        res = {"mu_e": self.mu_n, "mu_h": self.mu_p, "N_d": self.N_d, "N_a": self.N_a, "q": 1.602176634e-19, "k_B": 1.3806488e-23}
        if self.T is not None:
            res["T"] = self.T
        return res

    def widget(self):
        return _SemiconductorWidget(self)


class _SemiconductorWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self._mu_n = ScalarFunctionWidget("Electron mobility mu_n (m^2/V s)", self._param.mu_n, valueChanged=self.__set)
        self._mu_p = ScalarFunctionWidget("Hole mobility mu_n (m^2/V s)", self._param.mu_p, valueChanged=self.__set)
        self._N_d = ScalarFunctionWidget("Doner density (1/m^3)", self._param.N_d, valueChanged=self.__set)
        self._N_a = ScalarFunctionWidget("Acceptor density (1/m^3)", self._param.N_a, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._mu_n)
        layout.addWidget(self._mu_p)
        layout.addWidget(self._N_d)
        layout.addWidget(self._N_a)
        self.setLayout(layout)

    def __set(self):
        self._param.mu_n = self._mu_n.value()
        self._param.mu_p = self._mu_p.value()
        self._param.N_d = self._N_d.value()
        self._param.N_a = self._N_a.value()
