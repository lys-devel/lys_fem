import numpy as np

from lys import Wave
from .FEM import FEMProject


class FEMSolution:
    _keys = ["point", "line", "triangle", "quad", "tetra", "hexa", "prism", "pyramid"]

    def __init__(self, path, fem=None):
        if fem is None:
            self._fem = FEMProject.fromFile(path + "/input.dic")
        else:
            self._fem = fem
        self._path = path

    def variableList(self, data_number=0, solver="Solver0"):
        path = self._path + "/Solutions/" + solver + "/"
        data = self._loadData(path, data_number)
        return data.keys()

    def eval(self, varName, data_number=0, solver="Solver0"):
        path = self._path + "/Solutions/" + solver + "/"
        meshes = np.load(path + "mesh.npz", allow_pickle=True)
        data = self._loadData(path, data_number)
        array = eval(varName, {}, data)

        res = []
        coords = meshes["coords"]
        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 3-coords.shape[1]))])
        for domain in meshes["mesh"]:
            nodes = np.unique([n for n in domain.values()])
            elems = {elem: np.searchsorted(nodes, n) for elem, n in domain.items()}
            res.append(Wave(array[nodes-1], coords[nodes-1], elements=elems))
        return res

    def _loadData(self, path, data_number):
        return np.load(path + "data" + str(data_number) + ".npz")

