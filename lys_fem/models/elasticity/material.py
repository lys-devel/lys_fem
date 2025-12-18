import itertools
import numpy as np

from lys_fem import FEMParameter, Coef


class ElasticParameters(FEMParameter):
    name = "Elasticity"

    def __init__(self, rho=1, C=[1, 1], type="lame", alpha=None, d_e=None, d_h=None, e_piezo=None, piezo_type="isotropic"):
        self["rho"] = Coef(rho, description="Density (kg/m^3)")
        self["C"] = _ElasticConstantCoef(C, type)
        self["alpha"] = Coef(None if alpha is None else np.array(alpha), shape=(3,3), description="Thermal expansion coef. (1/K)", default=np.eye(3).tolist())
        self["d_e"] = Coef(d_e, description="DP coef. for electron (eV)", default=10)
        self["d_h"] = Coef(d_h, description="DP coef. for hole (eV)", default=10)
        self["e_piezo"] = _PiezoelectricStressCoef(e_piezo, piezo_type)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["C"] = self["C"].C
        d["type"] = self["C"].type
        d["e_piezo"] = self["e_piezo"].e
        d["piezo_type"] = self["e_piezo"].type
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
    

class _PiezoelectricStressCoef(Coef):
    def __init__(self, expr, type="isotropic", shape=(3,3,3), description="Piezoelectric stress coef. (C/m2)", default=1):
        super().__init__(None, shape=shape, description=description, default=default)
        self.e = np.array(expr).tolist() if expr is not None else None
        self.type = type

    @property
    def expression(self):
        if self.e is None:
            return None
        return self.__get_e(self._construct_e())

    def setValid(self, b=True):
        if b:
            self.e = self.default
        else:
            self.e = None

    def _construct_e(self):
        if self.type == "isotropic":
            return [[self.e if i==j else 0 for j in range(6)] for i in range(3)]
        elif self.type == "cubic":
            res = np.zeros((3,6))
            res[0,4] = res[1,3] = res[2,5] = self.e
            return res
        elif self.type == "orthorhombic":
            e = float(self.e[0]), float(self.e[1]), float(self.e[2])
            return [[e[i] if i==j else 0 for j in range(6)] for i in range(3)]
        elif self.type in ["monoclinic", "triclinic"]:
            return self.e
        else:
            e31, e33, e15 = float(self.e[0]), float(self.e[1]), float(self.e[2])
            res = np.zeros((3,6))
            res[2,0] = res[2,1] = e31
            res[2,2] = e33
            res[1,3] = res[0,4] = e15
            return res.tolist()

    def __get_e(self, e):
        res, e = np.zeros((3,3,3)), np.array(e)
        for i,j,k in itertools.product(range(3),range(3),range(3)):
            res[k,i,j] = e[k][self.__map(i,j)]
        return res.tolist()
    
    def __map(self, i, j):
        if i == j:
            return i
        if (i == 0 and j == 1) or (i == 1 and j == 0):
            return 3
        if (i == 1 and j == 2) or (i == 2 and j == 1):
            return 4
        if (i == 0 and j == 2) or (i == 2 and j == 0):
            return 5

    def widget(self):
        from .widgets import PiezoConstWidget
        return PiezoConstWidget(self)
