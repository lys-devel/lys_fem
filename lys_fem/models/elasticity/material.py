import itertools
import numpy as np

from lys_fem import FEMParameter


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
            "C": "Elastic constant (Pa)",
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
        from .widgets import ElasticConstWidget
        if name=="C":
            return ElasticConstWidget(self)
        else:
            return super().widget(name)


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
        elif self.type in ["monoclinic", "triclinic", "general"]:
            return self.C

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

