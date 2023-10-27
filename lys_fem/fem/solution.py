import os
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
        meshes = self._loadMesh(path)
        data = self._loadData(path, data_number)
        array = model.eval(data, self._fem, varName)
        res = []
        for mesh in meshes:
            elems = {key: mesh[key] for key in self._keys if key in mesh}
            res.append(Wave(array[mesh["nodes"]], mesh["coords"], elements=elems))
        return res

    def _loadMesh(self, path):
        i = 0
        meshes = []
        while os.path.exists(path + "mesh" + str(i) + ".npz"):
            meshes.append(np.load(path + "mesh" + str(i) + ".npz"))
            i += 1
        return meshes

    def _loadData(self, path, data_number):
        return np.load(path + "data" + str(data_number) + ".npz")

    def _getModel(self, fem, model):
        if model is None:
            return fem.models[0]
        else:
            return model
