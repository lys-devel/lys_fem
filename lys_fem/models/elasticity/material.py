from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget


class ElasticParameters(FEMParameter):
    def __init__(self, rho=1, C=[1, 1], type="lame"):
        self.rho = rho
        self.C = C
        self.type = type

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    def getParameters(self, dim):
        return {"rho": self.rho, "C": self._constructC()}

    def _constructC(self):
        if self.type == "lame":
            return [[self._lame(i, j) for j in range(6)] for i in range(6)]

    def _lame(self, i, j):
        res = 0
        if i < 3 and j < 3:
            res += self.C[0]
        if i == j:
            if i < 3:
                res += 2 * self.C[1]
            else:
                res += self.C[1]
        return res

    def widget(self):
        return _ElasticParamsWidget(self)


class _ElasticParamsWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._rho = ScalarFunctionWidget("Density rho", self._param.rho, valueChanged=self.__set)
        self._lamb = ScalarFunctionWidget("Lame const. lambda", self._param.C[0], valueChanged=self.__set)
        self._mu = ScalarFunctionWidget("Lame const. mu", self._param.C[1], valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._rho)
        layout.addWidget(self._lamb)
        layout.addWidget(self._mu)
        self.setLayout(layout)

    def __set(self):
        self._param.rho = self._rho.text()
        self._param.C = [self._lamb.text(), self._mu.text()]
