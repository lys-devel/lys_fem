import numpy as np

from lys import Wave
from .FEM import FEMProject


class FEMSolution:
    _keys = ["point", "line", "triangle", "quad", "tetra", "hexa", "prism", "pyramid"]

    def __init__(self, path=".", solver="Solver0"):
        from lys_fem.ngs import NGSSolution
        self._fem = FEMProject.fromFile(path + "/input.dic")
        self._path = path
        self._sol = NGSSolution(self._fem, path+"/Solutions/"+solver)

    def eval(self, varName, data_number=0, coords=None):
        return self._sol.eval(varName, data_number, coords)


