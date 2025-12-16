import numpy as np
from lys_fem import FEMParameter, Coef


class ElectrostaticParameters(FEMParameter):
    name = "Electrostatics"

    def __init__(self, eps_r=np.eye(3)):
        self["eps_r"] = Coef(eps_r, shape=(3,3), description="Relative permittivity")
