import itertools
import numpy as np

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget, MatrixFunctionWidget


class ElasticParameters(FEMParameter):
    def __init__(self, rho=1, C=[1, 1], type="lame"):
        self.rho = rho
        self.C = C
        self.type = type

    @classmethod
    @property
    def name(cls):
        return "Elasticity"

    @classmethod
    @property
    def units(self):
        return {"rho": "kg/m^3", "C": "Pa"}

    def getParameters(self, dim):
        return {"rho": self.rho, "C": self.__getC(dim, self._constructC())}

    def _constructC(self):
        if self.type in ["lame", "young", "isotropic"]:
            if self.type == "lame":
                lam, mu = self.C
            elif self.type == "young":
                E, v = self.C
                lam, mu = E*v/(1+v)/(1-2*v), E/((1+v)*2)
            elif self.type == "isotropic":
                C1, C2 = self.C
                lam, mu = C2, (C1-C2)/2
            return [[self._lame(i, j, lam, mu) for j in range(6)] for i in range(6)]

    def __getC(self, dim, C):
        res = np.zeros((dim,dim,dim,dim)).tolist()
        for i,j,k,l in itertools.product(range(dim),range(dim),range(dim),range(dim)):
            res[i][j][k][l] = C[self.__map(i,j)][self.__map(k,l)]
        return res
    
    def __map(self, i, j):
        if i == j:
            return i
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return 3
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return 4
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return 5

    def _lame(self, i, j, lam, mu):
        res = 0
        if i < 3 and j < 3:
            res += lam
        if i == j:
            if i < 3:
                res += 2 * mu
            else:
                res += mu
        return res

    def widget(self):
        return _ElasticParamsWidget(self)


class _ElasticParamsWidget(QtWidgets.QWidget):
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
        self._rho = ScalarFunctionWidget("Density rho (kg/m^3)", self._param.rho, valueChanged=self.__set)

        self._lamb = ScalarFunctionWidget("Lamé const. lambda (GPa)", self._param.C[0]*1e-9, valueChanged=self.__set)
        self._mu = ScalarFunctionWidget("Lamé const. mu (GPa)", self._param.C[1]*1e-9, valueChanged=self.__set)

        self._E = ScalarFunctionWidget("Young modulus E (GPa)", self._param.C[0]*1e-9, valueChanged=self.__set)
        self._v = ScalarFunctionWidget("Poisson ratio v (GPa)", self._param.C[1]*1e-9, valueChanged=self.__set)

        self._C1 = ScalarFunctionWidget("Elastic const. C11 (GPa)", self._param.C[0]*1e-9, valueChanged=self.__set)
        self._C2 = ScalarFunctionWidget("Elastic const. C12 (GPa)", self._param.C[1]*1e-9, valueChanged=self.__set)

        self._widgets = [self._lamb, self._mu, self._E, self._v, self._C1, self._C2]

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._type)
        layout.addWidget(self._rho)
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
        self._param.rho = self._rho.value()
        if self._type.currentText() == "Lamé":
            self._param.type = "lame"
            self._param.C = [self._lamb.value()*1e9, self._mu.value()*1e9]
        if self._type.currentText() == "Young":
            self._param.type = "young"
            self._param.C = [self._E.value()*1e9, self._v.value()*1e9]
        if self._type.currentText() == "Isotropic":
            self._param.type = "isotropic"
            self._param.C = [self._C1.value()*1e9, self._C2.value()*1e9]


class ThermalExpansionParameters(FEMParameter):
    def __init__(self, alpha=np.eye(3).tolist()):
        self.alpha = alpha

    @classmethod
    @property
    def name(cls):
        return "Thermal Expansion"

    @classmethod
    @property
    def units(cls):
        return {"alpha": "1"}

    def getParameters(self, dim):
        if dim == 1:
            return {"alpha": self.alpha[0][0]}
        else:
            return {"alpha": np.array(self.alpha)[:dim,:dim].tolist()}

    def widget(self):
        return _ThermalExpansionWidget(self)


class _ThermalExpansionWidget(QtWidgets.QWidget):
    def __init__(self, param):
        super().__init__()
        self._param = param
        self._alpha = MatrixFunctionWidget("Thermal expansion coef. alpha", self._param.alpha, valueChanged=self.__set)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._alpha)
        self.setLayout(layout)

    def __set(self):
        self._param.alpha = self._alpha.value()
