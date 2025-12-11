import numpy as np
from lys_fem import FEMParameter, Coef


class LLGParameters(FEMParameter):
    name = "LLG"

    def __init__(self, Ms=1e6, Aex=1e-11, alpha_LLG=0, Ku=None, u_Ku=None, Kc=None, beta_st=None, B1=None, B2=None, **kwargs):
        self.alpha_LLG = Coef(alpha_LLG, description="Gilbert damping const.")
        self.Ms = Coef(Ms, description="Saturation magnetization (A/m)")
        self.Aex = Coef(Aex, description="Exchange constant (J/m)")
        self.Ku = Coef(Ku, description="Uniaxial anisotropy constant (J/m^3)", default=1e3)
        self.u_Ku = Coef(u_Ku, shape=(3,), description= "Uniaxial anisotropy direction", default=[0,0,1])
        self.Kc = CubicAnisotropyCoef(Kc)
        self.beta_st = Coef(beta_st, description="Nonadiabasity of spin-transfer torque", default=0.1)
        self.K_MS = CubicMagnetostrictionCoef(B1, B2)

    def saveAsDictionary(self):
        d = super().saveAsDictionary()
        d["Kc"] = self.Kc.Kc

        del d["K_MS"]
        d["B1"] = self.K_MS.B1
        d["B2"] = self.K_MS.B2
        return d

class CubicAnisotropyCoef(Coef):
    def __init__(self, Kc, shape=(3,3,3,3), description="Cubic anisotropy (J/m^3)"):
        super().__init__(None, shape=shape, description=description)
        self.Kc = Kc 

    @property
    def expression(self):
        if self.Kc is None:
            return None
        res = np.zeros((3,3,3,3), dtype=object)
        Kc = float(self.Kc)
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
    
    def setDefault(self):
        self.Kc = 100

    def widget(self):
        from lys_fem.widgets import ScalarFunctionWidget
        return ScalarFunctionWidget(None, self.Kc, valueChanged=lambda x: setattr(self, "Kc", x))


class CubicMagnetostrictionCoef(Coef):
    def __init__(self, B1, B2, shape=(3,3,3,3), description="Cubic magnetostriction (GPa)"):
        super().__init__(None, shape=shape, description=description)
        self.B1 = B1
        self.B2 = B2

    @property
    def expression(self):
        if self.B1 is None or self.B2 is None:
            return None
        B1, B2 = float(self.B1), float(self.B2)
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
        return K.tolist()
    
    def setDefault(self):
        self.B1 = 0.0
        self.B2 = 0.0

    def widget(self):
        from lys.Qt import QtWidgets
        from lys_fem.widgets import ScalarFunctionWidget
        B1 = ScalarFunctionWidget(None, self.B1, valueChanged=lambda x: setattr(self, "B1", x))
        B2 = ScalarFunctionWidget(None, self.B2, valueChanged=lambda x: setattr(self, "B2", x))

        w = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel("B1 (GPa)"), 0, 0)
        layout.addWidget(B1, 0, 1)
        layout.addWidget(QtWidgets.QLabel("B2 (GPa)"), 1, 0)
        layout.addWidget(B2, 1, 1)

        return w