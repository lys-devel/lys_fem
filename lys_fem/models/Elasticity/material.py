from lys.Qt import QtWidgets
from lys_fem import FEMParameter


class ElasticParameters(FEMParameter):
    def __init__(self, rho, C, type="lame"):
        self.rho = rho
        self.C = C
        self.type = type

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    def getParameters(self):
        return {"rho": self.rho, "C": self._constructC()}

    @classmethod
    @property
    def default(self):
        return ElasticParameters(1, [1, 1])

    def widget(self):
        return _ElasticParamsWidget(self)

    def _constructC(self):
        if self.type == "lame":
            return [[self._lame(i, j) for j in range(6)] for i in range(6)]

    def _lame(self, i, j):
        res = "0"
        if i < 3 and j < 3:
            res += "+" + self.C[0]
        if i == j:
            if i < 3:
                res += "+ 2 *" + self.C[1]
            else:
                res += "+" + self.C[1]
        return res


class _ElasticParamsWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._rho = QtWidgets.QLineEdit()
        self._rho.setText(str(self._param.rho))
        self._rho.textChanged.connect(self.__set)

        self._lamb = QtWidgets.QLineEdit()
        self._lamb.setText(str(self._param.C[0]))
        self._lamb.textChanged.connect(self.__set)

        self._mu = QtWidgets.QLineEdit()
        self._mu.setText(str(self._param.C[1]))
        self._mu.textChanged.connect(self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("rho"))
        layout.addWidget(self._rho)
        layout.addWidget(QtWidgets.QLabel("lambda"))
        layout.addWidget(self._lamb)
        layout.addWidget(QtWidgets.QLabel("mu"))
        layout.addWidget(self._mu)
        self.setLayout(layout)

    def __set(self):
        self._param.rho = self._rho.text()
        self._param.C = [self._lamb.text(), self._mu.text()]
