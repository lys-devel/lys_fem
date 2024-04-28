import numpy as np

from ngsolve import GridFunction
from lys import Wave

from .mesh import generateMesh, exportMesh
from .material import generateMaterial
from .models import generateModel

class NGSSolution:
    def __init__(self, fem, dirname):
        self._fem = fem
        self._dirname = "Solutions/" + dirname
        self._mesh = generateMesh(fem)
        mats = generateMaterial(fem, self._mesh)
        self._model = generateModel(fem, self._mesh, mats)
        self._fes = self._model.finiteElementSpace
        self._grid = GridFunction(self._fes)
        self._meshInfo = exportMesh(self._mesh)

    def eval(self, expression, index, coords=None):
        self._grid.Load(self._dirname+"/ngs"+str(index))

        data = {}
        vars = self._model.variables
        if len(vars) == 1:
            data[vars[0].name] = vars[0].scale * self._grid
        else:
            for v, g in zip(vars, self._grid):
                data[v.name] = v.scale * g

        array = eval(expression, {}, data)
        return self.__getDomainValues(array)

    def __getDomainValues(self, array):
        domains, coords = self._meshInfo
        data=np.array([array(self._mesh(*c)) for c in coords/self._fem.scaling.getScaling("m")])
        res = []
        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 3-coords.shape[1]))])
        for domain in domains:
            nodes = np.unique([n for n in domain.values()])
            elems = {elem: np.searchsorted(nodes, n) for elem, n in domain.items()}
            res.append(Wave(data[nodes-1], coords[nodes-1], elements=elems))
        return res
