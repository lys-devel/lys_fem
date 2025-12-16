import numpy as np
from lys_fem import FEMParameter, Coef


class HeatConductionParameters(FEMParameter):
    name = "Heat Conduction"
    def __init__(self, C_v=1.0, k=np.eye(3)):
        self["C_v"] = Coef(C_v, description="Heat capacity (J/kg m^3)")
        self["k"] = Coef(k, shape=(3,3), description="Thermal diffusion coef. (W/m K)")
