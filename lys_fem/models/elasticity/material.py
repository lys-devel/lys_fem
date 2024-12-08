import itertools
import numpy as np

from lys.Qt import QtWidgets
from lys_fem import FEMParameter
from lys_fem.widgets import ScalarFunctionWidget


class ElasticParameters(FEMParameter):
    name = "Elasticity"

    def __init__(self, rho=1, C=[1, 1], type="lame", alpha=None, d_e=None, d_h=None):
        self.rho = rho
        self.C = C
        self.alpha = alpha
        self.d_e = d_e
        self.d_h = d_h
        self.type = type

    def getParameters(self, dim):
        super().getParameters(dim)
        res = {}
        if self.rho is not None:
            res["rho"] = self.rho
        if self.C is not None:
            res["C"] = self.__getC(dim, self._constructC())
        if self.alpha is not None:
            res["alpha"] = np.array(self.alpha)[:dim,:dim].tolist()
        if self.d_e is not None:
            res["d_e"] = self.d_e*1.60218e-19
        if self.d_h is not None:
            res["d_h"] = self.d_h*1.60218e-19
        return res

    @property
    def description(self):
        return {
            "rho": "Density (kg/m^3)",
            "C": "Elastic constant (GPa)",
            "alpha": "Thermal expansion coef. (1/K)",
            "d_e": "DP coef. for electron (eV)",
            "d_h": "DP coef. for hole (eV)"
        }

    @property
    def default(self):
        return {
            "rho": 1,
            "C": [1e9, 1e9],
            "alpha": np.eye(3).tolist(),
            "d_e": 10,
            "d_h": 10
        }

    def widget(self, name):
        if name=="C":
            return _ElasticConstWidget(self)
        else:
            return super().widget(name)


    def _constructC(self):
        if self.type in ["lame", "young", "isotropic"]:
            if self.type == "lame":
                lam, mu = self.C
            elif self.type == "young":
                E, v = self.C
                lam, mu = E*v/(1+v)/(1-2*v), E/((1+v)*2)
            elif self.type == "isotropic":
                C1, C2 = float(self.C[0]), float(self.C[1])
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


class _ElasticConstWidget(QtWidgets.QWidget):
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
