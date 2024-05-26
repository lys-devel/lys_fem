import glob
import numpy as np

from lys import Wave
from .mesh import generateMesh, exportMesh, loadMesh
from .material import generateMaterial
from .models import generateModel
from .util import GridFunction

class NGSSolution:
    def __init__(self, fem, dirname):
        self._fem = fem
        self._dirname = dirname
        self._mesh = generateMesh(fem)
        #self._mesh = loadMesh(fem, self._dirname+"/mesh0.vol")
        self._mats = generateMaterial(fem, self._mesh)
        self._model = generateModel(fem, self._mesh, self._mats)

        self._grid = GridFunction(self._model.finiteElementSpace)
        self._meshInfo = exportMesh(self._mesh)

    @property
    def maxIndex(self):
        return len(glob.glob(self._dirname+"/ngs*"))-1

    def coef(self, expression, index=-1):
        if index < 0:
            index = self.maxIndex + index + 1
        self._grid.Load(self._dirname+"/ngs"+str(index), parallel=self._fem.parallel)

        data = {}
        n = 0
        for v in self._model.variables:
            data[v.name] = [v.scale * self._grid.components[i] for i in range(n,n+v.size)]
            if v.size == 1:
                data[v.name] = data[v.name][0]
            n += v.size

        return eval(expression, {}, data)

    def eval(self, expression, index, coords=None):
        f = self.coef(expression, index)
        if coords is None:
            return self.__getDomainValues(f)
        else:
            if not hasattr(coords, "__iter__"):
                mip = self.__coordsToMIP(np.array([coords])/self._fem.scaling.getScaling("m"))
                return np.array([f(mi) for mi in mip])[0]
            else:
                mip = self.__coordsToMIP(np.array(coords)/self._fem.scaling.getScaling("m"))
                return np.array([f(mi) for mi in mip])

    def __getDomainValues(self, f):
        domains, coords = self._meshInfo
        mip = [self._mesh(*c) for c in coords/self._fem.scaling.getScaling("m")]
        data=np.array([f(mi) for mi in mip])
        res = []
        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 3-coords.shape[1]))])
        for domain in domains:
            nodes = np.unique([n for n in domain.values()])
            elems = {elem: np.searchsorted(nodes, n) for elem, n in domain.items()}
            res.append(Wave(data[nodes-1], coords[nodes-1], elements=elems))
        return res
    
    def __coordsToMIP(self, coords):
        dim = 0 if self._fem.dimension == 1 else 1
        if len(coords.shape) > dim:
            return [self.__coordsToMIP(c) for c in coords]
        else:
            if self._fem.dimension == 1:
                return self._mesh(coords)
            else:
                return self._mesh(*coords)