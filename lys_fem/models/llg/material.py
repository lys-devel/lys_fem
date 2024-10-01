import numpy as np
from lys_fem import FEMParameter


class LLGParameters(FEMParameter):
    name = "LLG"

    def __init__(self, Ms=1e6, Aex=1e-11, alpha=0, Ku=None, u_Ku=None):
        self.alpha = alpha
        self.Ms = Ms
        self.Aex = Aex
        self.Ku = Ku
        self.u_Ku = u_Ku

    def getParameters(self, dim):
        res = {}
        if self.Ms is not None:
            res["Ms"] = self.Ms
        if self.Aex is not None:
            res["Aex"] = self.Aex
        if self.alpha is not None:
            res["alpha"] = self.alpha
        if self.Ku is not None:
            res["Ku"] = self.Ku
        if self.u_Ku is not None:
            res["u_Ku"] = self.u_Ku/np.linalg.norm(self.u_Ku)
        return res

    @property
    def description(self):
        return {
            "Ms": "Saturation magnetization (A/m)",
            "Aex": "Exchange constant (J/m)",
            "alpha": "Gilbert damping const.",
            "Ku": "Uniaxial anisotropy (J/m^3)",
            "u_Ku": "Uniaxial anisotropy direction",
        }

    @property
    def default(self):
        return {
            "Ms": 1e6,
            "Aex": 1e-11,
            "alpha": 0,
            "Ku": 1e3,
            "u_Ku": [0,0,1]
        }
