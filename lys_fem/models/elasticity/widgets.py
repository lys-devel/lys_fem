from lys.Qt import QtWidgets
from lys_fem.widgets import GeometrySelector, ScalarFunctionWidget


class ElasticConstWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self.__initlayout()

    def __initlayout(self):
        d = {"lame": "Lamé", "yound": "Young", "isotropic": "Isotropic"}
        self._type = QtWidgets.QComboBox()
        self._type.addItems(d.values())
        self._type.setCurrentText(d[self._param.type])
        self._type.currentTextChanged.connect(self.__changeMode)

        self._lamb = ScalarFunctionWidget("Lamé const. lambda (Pa)", self._param.C[0], valueChanged=self.__set)
        self._mu = ScalarFunctionWidget("Lamé const. mu (Pa)", self._param.C[1], valueChanged=self.__set)

        self._E = ScalarFunctionWidget("Young modulus E (Pa)", self._param.C[0], valueChanged=self.__set)
        self._v = ScalarFunctionWidget("Poisson ratio v (Pa)", self._param.C[1], valueChanged=self.__set)

        self._C1 = ScalarFunctionWidget("Elastic const. C11 (Pa)", self._param.C[0], valueChanged=self.__set)
        self._C2 = ScalarFunctionWidget("Elastic const. C12 (Pa)", self._param.C[1], valueChanged=self.__set)

        self._widgets = [self._lamb, self._mu, self._E, self._v, self._C1, self._C2]

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._type)
        layout.addWidget(self._lamb)
        layout.addWidget(self._mu)
        layout.addWidget(self._E)
        layout.addWidget(self._v)
        layout.addWidget(self._C1)
        layout.addWidget(self._C2)
        self.setLayout(layout)
        self.__changeMode()

    def __changeMode(self):
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


class ThermoelasticWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas, title):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas, title)

    def __initlayout(self, fem, canvas, title):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._value = ScalarFunctionWidget(title, self._cond.values, valueChanged=self.__valueChanged)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addWidget(self._value)
        self.setLayout(layout)

    def __valueChanged(self, vector):
        self._cond.values = vector


class DeformationPotentialWidget(QtWidgets.QWidget):
    def __init__(self, cond, fem, canvas):
        super().__init__()
        self._cond = cond
        self.__initlayout(fem, canvas)

    def __initlayout(self, fem, canvas):
        self._selector = GeometrySelector(canvas, fem, self._cond.geometries)
        self._varName1 = QtWidgets.QLineEdit()
        self._varName1.setText(self._cond.varNames[0])
        self._varName1.textChanged.connect(self.__textChanged)
        self._varName2 = QtWidgets.QLineEdit()
        self._varName2.setText(self._cond.varNames[1])
        self._varName2.textChanged.connect(self.__textChanged)

        h1 = QtWidgets.QGridLayout()
        h1.addWidget(QtWidgets.QLabel("Electron Name"), 0, 0)
        h1.addWidget(QtWidgets.QLabel("Hole Name"), 1, 0)
        h1.addWidget(self._varName1, 0, 1)
        h1.addWidget(self._varName2, 1, 1)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._selector)
        layout.addLayout(h1)
        self.setLayout(layout)

    def __textChanged(self, txt):
        self._cond.varNames = [self._varName1.text(), self._varName2.text()]
