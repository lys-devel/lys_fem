import itertools
import numpy as np

from lys_fem import FEMParameter, Coef


class ElasticParameters(FEMParameter):
    name = "Elasticity"

    def __init__(self, rho=1, C=[1, 1], type="lame", alpha=None, d_e=None, d_h=None):
        self["rho"] = Coef(rho, description="Density (kg/m^3)")
        self["C"] = _ElasticConstantCoef(C, type)
        self["alpha"] = Coef(None if alpha is None else np.array(alpha), shape=(3,3), description="Thermal expansion coef. (1/K)", default=np.eye(3).tolist())
        self["d_e"] = Coef(d_e, description="DP coef. for electron (eV)", default=10)
        self["d_h"] = Coef(d_h, description="DP coef. for hole (eV)", default=10)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["C"] = self["C"].C
        d["type"] = self["C"].type
        return d


class _ElasticConstantCoef(Coef):
    def __init__(self, expr, type, shape=(3,3,3,3), description="Elastic constant (Pa)", default=None):
        super().__init__(None, shape=shape, description=description, default=default)
        self.C = np.array(expr).tolist()
        self.type = type

    @property
    def expression(self):
        return self.__getC(self._constructC())

    def _constructC(self):
        if self.type in ["lame", "young", "isotropic"]:
            if self.type == "lame":
                lam, mu = float(self.C[0]), float(self.C[1])
            elif self.type == "young":
                E, v = float(self.C[0]), float(self.C[1])
                lam, mu = E*v/(1+v)/(1-2*v), E/((1+v)*2)
            elif self.type == "isotropic":
                C1, C2 = float(self.C[0]), float(self.C[1])
                lam, mu = C2, (C1-C2)/2
            return [[self._lame(i, j, lam, mu) for j in range(6)] for i in range(6)]
        elif self.type == "cubic":
            C1, C2, C4 = float(self.C[0]), float(self.C[1]), float(self.C[2])
            return [[self._cubic(i, j, C1, C2, C4) for j in range(6)] for i in range(6)]
        elif self.type in ["monoclinic", "triclinic", "general"]:
            return self.C

    def __getC(self, C):
        res = np.zeros((3,3,3,3)).tolist()
        for i,j,k,l in itertools.product(range(3),range(3),range(3),range(3)):
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
    
    def _cubic(self, i, j , C1, C2, C4):
        if i < 3 and j < 3:
            if i == j:
                return C1
            else:
                return C2
        elif i==j:
            return C4
        return 0
    
    def widget(self):
        from .widgets import ElasticConstWidget
        return ElasticConstWidget(self)
