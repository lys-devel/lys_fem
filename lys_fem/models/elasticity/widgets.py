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


class PiezoConstWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        d = ["isotropic", "cubic", "tetragonal", "orthorhombic", "hexagonal", "trigonal","monoclinic", "triclinic"]
        self._type = QtWidgets.QComboBox()
        self._type.addItems(d)
        self._type.setCurrentText(self._param.type)
        self._type.currentTextChanged.connect(self.__changeMode)

        if self._param.type in ["isotropic", "cubic"]:
            value = [self._param.e] * 3
        elif self._param.type == ["monoclinic", "triclinic"]:
            value = self._param.e[0,0], self._param.e[1,1], self._param.e[2,2]
        else:
            value = self._param.e
        self._e1 = ScalarFunctionWidget(value[0], valueChanged=self.__set, label="Piezoelectric stress const. e11 (C/m2)")
        self._e2 = ScalarFunctionWidget(value[1], valueChanged=self.__set, label="Piezoelectric stress const. e22 (C/m2)")
        self._e3 = ScalarFunctionWidget(value[2], valueChanged=self.__set, label="Piezoelectric stress const. e33 (C/m2)")

        self._e14 = ScalarFunctionWidget(value[0], valueChanged=self.__set, label="Piezoelectric stress const. e14 (C/m2)")

        self._e31 = ScalarFunctionWidget(value[0], valueChanged=self.__set, label="Piezoelectric stress const. e31 (C/m2)")
        self._e33 = ScalarFunctionWidget(value[1], valueChanged=self.__set, label="Piezoelectric stress const. e33 (C/m2)")
        self._e15 = ScalarFunctionWidget(value[2], valueChanged=self.__set, label="Piezoelectric stress const. e15 (C/m2)")

        self._mon = _PiezoConstMatrix("Piezoelectric stress const. (C/m2)", "monoclinic", self._param.e, valueChanged=self.__set)
        self._p = _PiezoConstMatrix("Piezoelectric stress const. (C/m2)", "triclinic", self._param.e, valueChanged=self.__set)

        self._widgets = [self._e1, self._e2, self._e3, self._e14, self._e31, self._e33, self._e15, self._mon, self._p]

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._type)
        layout.addWidget(self._e1)
        layout.addWidget(self._e2)
        layout.addWidget(self._e3)
        layout.addWidget(self._e14)
        layout.addWidget(self._e31)
        layout.addWidget(self._e33)
        layout.addWidget(self._e15)
        layout.addWidget(self._mon)
        layout.addWidget(self._p)
        self.setLayout(layout)
        self.__changeMode()

    def __changeMode(self):
        self.__set()
        for w in self._widgets:
            w.hide()
        if self._type.currentText() == "isotropic":
            self._e1.show()
        elif self._type.currentText() == "cubic":
            self._e14.show()
        elif self._type.currentText() == "orthorhombic":
            self._e1.show()
            self._e2.show()
            self._e3.show()
        elif self._type.currentText() == "monoclinic":
            self._mon.show()
        elif self._type.currentText() == "triclinic":
            self._p.show()
        else:
            self._e31.show()
            self._e33.show()
            self._e15.show()

    def __set(self):
        self._param.type = self._type.currentText()
        if self._type.currentText() == "isotropic":
            self._param.e = self._e1.value()
        elif self._type.currentText() == "cubic":
            self._param.e = self._e14.value()
        elif self._type.currentText() == "orthorhombic":
            self._param.e = [self._e1.value(), self._e2.value(), self._e3.value()]
        elif self._type.currentText() == "monoclinic":
            self._param.e = self._mon.value()
        elif self._type.currentText() == "triclinic":
            self._param.e = self._p.value()
        else:
            self._param.e = [self._e31.value(), self._e33.value(), self._e15.value()]


class _PiezoConstMatrix(MatrixFunctionWidget):
    def __init__(self, label, type, value, valueChanged=None, **kwargs):
        self._type = type
        super().__init__(value if np.array(value).shape==(3,6) else np.ones((3,6)), label=label, valueChanged=None, **kwargs)
        self._combo.setCurrentText("Full")
        self._combo.hide()
        self._combo.currentTextChanged.connect(self._changeType)
        if valueChanged is not None:
            super().valueChanged.connect(valueChanged)

    def _changeType(self):
        super()._changeType()
        if self._type == "monoclinic":
            for i in range(3):
                self._value[i][3].setEnabled(False)
                self._value[i][5].setEnabled(False)
                self._value[i][3].setText("0")
                self._value[i][5].setText("0")