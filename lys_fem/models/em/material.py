import numpy as np
from lys_fem import FEMParameter


class ElectrostaticParameters(FEMParameter):
    name = "Electrostatics"

    def __init__(self, eps_r=np.eye(3).tolist()):
        self.eps_r = eps_r

    def getParameters(self, dim):
        res = {}
        if self.eps_r is not None:
            res["eps_r"] = (np.array(self.eps_r))[:dim,:dim].tolist()
        return res

    @property
    def description(self):
        return {
            "eps_r": "Relative permittivity",
        }

    @property
    def default(self):
        return {
            "eps_r": np.eye(3).tolist(),
        }
