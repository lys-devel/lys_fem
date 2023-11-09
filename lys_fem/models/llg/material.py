from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget


class LLGParameters(FEMParameter):
    def __init__(self, alpha=0):
        self.alpha = str(alpha)

    @classmethod
    @property
    def name(cls):
        return "LLG"

    def getParameters(self, dim):
        return {"alpha": self.alpha}

    def widget(self):
        return _LLGParamsWidget(self)


class _LLGParamsWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        self._alpha = ScalarFunctionWidget("Gilbert damping const.", self._param.alpha, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._alpha)
        self.setLayout(layout)

    def __set(self):
        self._param.alpha = self._alpha.text()
