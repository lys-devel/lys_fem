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

    def eval(self, varName, model=None, data_number=0, solver="Solver0"):
        model = self._getModel(self._fem, model)

        path = self._path + "/Solutions/" + solver + "/"
        meshes = np.load(path + "mesh.npz", allow_pickle=True)
        data = self._loadData(path, data_number)
        array = model.eval(data, self._fem, varName)

        res = []
        coords = meshes["coords"]
        for domain in meshes["mesh"]:
            nodes = np.unique([n for n in domain.values()])
            elems = {elem: np.searchsorted(nodes, n) for elem, n in domain.items()}
            res.append(Wave(array[nodes-1], coords[nodes-1], elements=elems))
        return res

    def _loadData(self, path, data_number):
        return np.load(path + "data" + str(data_number) + ".npz")

    def _getModel(self, fem, model):
        if model is None:
            return fem.models[0]
        else:
            return model
