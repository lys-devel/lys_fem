import numpy as np
from lys_fem import FEMParameter


class HeatConductionParameters(FEMParameter):
    name = "Heat Conduction"
    def __init__(self, C_v=1.0, k=np.eye(3).tolist()):
        self.C_v = C_v # J/K m^3
        self.k = k # W/mK

    def getParameters(self):
        res = {}
        if self.C_v is not None:
            res["C_v"] = self.C_v
        if self.k is not None:
            res["k"] = np.array(self.k).tolist()
        return res

    @property
    def description(self):
        return {
            "C_v": "Heat capacity (J/kg m^3)",
            "k": "Thermal diffusion coef. (W/m K)",
        }

    @property
    def default(self):
        return {
            "C_v": 1,
            "k": np.eye(3).tolist(),
        }
