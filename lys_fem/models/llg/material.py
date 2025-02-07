from lys_fem import FEMParameter


class LLGParameters(FEMParameter):
    name = "LLG"

    def __init__(self, Ms=1e6, Aex=1e-11, alpha_LLG=0, Ku=None, u_Ku=None, Kc=None, u_Kc=None, beta_st=None, lam100=None, lam111=None):
        self.alpha_LLG = alpha_LLG
        self.Ms = Ms
        self.Aex = Aex
        self.Ku = Ku
        self.u_Ku = u_Ku
        self.Kc = Kc
        self.u_Kc = u_Kc
        self.beta_st = beta_st
        self.lam100 = lam100
        self.lam111 = lam111

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
            res["Kc"] = self.Kc
        if self.u_Kc is not None:
            res["u_Kc"] = self.u_Kc
        if self.beta_st is not None:
            res["beta_st"] = self.beta_st
        if self.lam100 is not None:
            res["lam100"] = self.lam100
        if self.lam111 is not None:
            res["lam111"] = self.lam111
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
            "u_Kc": "Cubic anisotropy directions",
            "beta_st": "Nonadiabasity of spin-transfer torque",
            "lam100": "Magnetostriction coefficient",
            "lam111": "Magnetostriction coefficient",
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
            "u_Kc": [[1,0,0], [0,1,0], [0,1,0]],
            "beta_st": 0.1,
            "lam100": 1e-6,
            "lam111": 1e-6,
        }
