import numpy as np
from lys_fem import FEMParameter


class LLGParameters(FEMParameter):
    name = "LLG"

    def __init__(self, Ms=1e6, Aex=1e-11, alpha_LLG=0, Ku=None, u_Ku=None, Kc=None, beta_st=None, B1=None, B2=None):
        self.alpha_LLG = alpha_LLG
        self.Ms = Ms
        self.Aex = Aex
        self.Ku = Ku
        self.u_Ku = u_Ku
        self.Kc = Kc
        self.beta_st = beta_st
        self.B1 = B1
        self.B2 = B2

    def getParameters(self):
        res = {}
        if self.Ms is not None:
            res["Ms"] = self.Ms
        if self.Aex is not None:
            res["Aex"] = self.Aex
        if self.alpha_LLG is not None:
            res["alpha_LLG"] = self.alpha_LLG
        if self.Ku is not None:
            res["Ku"] = self.Ku
        if self.u_Ku is not None:
            res["u_Ku"] = self.u_Ku
        if self.Kc is not None:
            res["Kc"] = self._construct_Kc(self.Kc)
        if self.beta_st is not None:
            res["beta_st"] = self.beta_st
        if self.B1 is not None and self.B2 is not None:
            res["K_MS"] = self._construct_MS(self.B1, self.B2)
        return res

    @property
    def description(self):
        return {
            "Ms": "Saturation magnetization (A/m)",
            "Aex": "Exchange constant (J/m)",
            "alpha_LLG": "Gilbert damping const.",
            "Ku": "Uniaxial anisotropy (J/m^3)",
            "u_Ku": "Uniaxial anisotropy direction",
            "Kc": "Cubic anisotropy (J/m^3)",
            "beta_st": "Nonadiabasity of spin-transfer torque",
            "B1": "Cubic magnetostriction coefficient (GPa)",
            "B2": "Cubic magnetostriction coefficient (GPa)",
        }

    @property
    def default(self):
        return {
            "Ms": 1e6,
            "Aex": 1e-11,
            "alpha_LLG": 0,
            "Ku": 1e3,
            "u_Ku": [0,0,1],
            "Kc": 100,
            "beta_st": 0.1,
            "B1": 0,
            "B2": 0,
        }

    def _construct_Kc(self, Kc):
        res = np.zeros((3,3,3,3), dtype=object)
        res[0,0,1,1] = Kc/6
        res[0,0,2,2] = Kc/6
        res[0,1,0,1] = Kc/6
        res[0,1,1,0] = Kc/6
        res[0,2,0,2] = Kc/6
        res[0,2,2,0] = Kc/6

        res[1,0,0,1] = Kc/6
        res[1,0,1,0] = Kc/6
        res[1,1,0,0] = Kc/6
        res[1,1,2,2] = Kc/6
        res[1,2,1,2] = Kc/6
        res[1,2,2,1] = Kc/6

        res[2,0,0,2] = Kc/6
        res[2,0,2,0] = Kc/6
        res[2,1,1,2] = Kc/6
        res[2,1,2,1] = Kc/6
        res[2,2,0,0] = Kc/6
        res[2,2,1,1] = Kc/6

        return res.tolist()

    def _construct_MS(self, B1, B2):
        K = np.zeros((3,3,3,3), dtype=object)
        K[0,0,0,0] = B1
        K[0,1,0,1] = B2/4
        K[0,1,1,0] = B2/4
        K[0,2,0,2] = B2/4
        K[0,2,2,0] = B2/4
        K[1,0,0,1] = B2/4
        K[1,0,1,0] = B2/4
        K[1,1,1,1] = B1
        K[1,2,1,2] = B2/4
        K[1,2,2,1] = B2/4
        K[2,0,0,2] = B2/4
        K[2,0,2,0] = B2/4
        K[2,1,1,2] = B2/4
        K[2,1,2,1] = B2/4
        K[2,2,2,2] = B1
        return K