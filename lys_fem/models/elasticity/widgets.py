import numpy as np
from lys.Qt import QtWidgets
from lys_fem.widgets import ScalarFunctionWidget, MatrixFunctionWidget


class ElasticConstWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        d = {"lame": "Lamé", "young": "Young", "isotropic": "Isotropic", "monoclinic": "Monoclinic", "triclinic": "Triclinic", "general": "Triclinic"}
        self._type = QtWidgets.QComboBox()
        self._type.addItems(d.values())
        self._type.setCurrentText(d[self._param.type])
        self._type.currentTextChanged.connect(self.__changeMode)

        self._lamb = ScalarFunctionWidget(self._param.C[0], valueChanged=self.__set, label="Lamé const. lambda (Pa)")
        self._mu = ScalarFunctionWidget(self._param.C[1], valueChanged=self.__set, label="Lamé const. mu (Pa)")

        self._E = ScalarFunctionWidget(self._param.C[0], valueChanged=self.__set, label="Young modulus E (Pa)")
        self._v = ScalarFunctionWidget(self._param.C[1], valueChanged=self.__set, label="Poisson ratio v (Pa)")

        self._C1 = ScalarFunctionWidget(self._param.C[0], valueChanged=self.__set, label="Elastic const. C11 (Pa)")
        self._C2 = ScalarFunctionWidget(self._param.C[1], valueChanged=self.__set, label="Elastic const. C12 (Pa)")

        self._mon = _ElasticConstMatrix("Elastic const. (Pa)", "Monoclinic", self._param.C, valueChanged=self.__set)
        self._p = _ElasticConstMatrix("Elastic const. (Pa)", "Triclinic", self._param.C, valueChanged=self.__set)

        self._widgets = [self._lamb, self._mu, self._E, self._v, self._C1, self._C2, self._mon, self._p]

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._type)
        layout.addWidget(self._lamb)
        layout.addWidget(self._mu)
        layout.addWidget(self._E)
        layout.addWidget(self._v)
        layout.addWidget(self._C1)
        layout.addWidget(self._C2)
        layout.addWidget(self._mon)
        layout.addWidget(self._p)
        self.setLayout(layout)
        self.__changeMode()

    def __changeMode(self):
        self.__set()
        for w in self._widgets:
            w.hide()
        if self._type.currentText() == "Lamé":
            self._lamb.show()
            self._mu.show()
        if self._type.currentText() == "Young":
            self._E.show()
            self._v.show()
        if self._type.currentText() == "Isotropic":
            self._C1.show()
            self._C2.show()
        if self._type.currentText() == "Monoclinic":
            self._mon.show()
        if self._type.currentText() == "Triclinic":
            self._p.show()

    def __set(self):
        if self._type.currentText() == "Lamé":
            self._param.type = "lame"
            self._param.C = [self._lamb.value(), self._mu.value()]
        if self._type.currentText() == "Young":
            self._param.type = "young"
            self._param.C = [self._E.value(), self._v.value()]
        if self._type.currentText() == "Isotropic":
            self._param.type = "isotropic"
            self._param.C = [self._C1.value(), self._C2.value()]
        if self._type.currentText() == "Monoclinic":
            self._param.type = "monoclinic"
            self._param.C = self._mon.value()
        if self._type.currentText() == "Triclinic":
            self._param.type = "triclinic"
            self._param.C = self._p.value()


class _ElasticConstMatrix(MatrixFunctionWidget):
    def __init__(self, label, type, value, **kwargs):
        self._type = type
        super().__init__(value if np.array(value).shape[0]==6 else np.ones((6,6)), label=label, **kwargs)
        self._combo.hide()

    def _changeType(self):
        super()._changeType()
        if self._type == "Monoclinic":
            for i, j in [(0,3), (0,4), (1,3), (1,4), (2,3), (2,4), (3,5), (4,5)]:
                self._value[i][j].setEnabled(False)
                self._value[i][j].setText("0")
                self._value[j][i].setText("0")

