import numpy as np

from ngsolve import GridFunction, H1
from lys import Wave

from .mesh import generateMesh, exportMesh
from .material import generateMaterial
from .models import generateModel


class NGSSolution:
    def __init__(self, fem, dirname):
        self._fem = fem
        self._dirname = dirname
        self._mesh = generateMesh(fem)
        self._mats = generateMaterial(fem, self._mesh)
        self._model = generateModel(fem, self._mesh, self._mats)

        self._grid = GridFunction(self._model.finiteElementSpace)
        self._meshInfo = exportMesh(self._mesh)

    def eval(self, expression, index, coords=None):
        self._grid.Load(self._dirname+"/ngs"+str(index))

        data = {}
        vars = self._model.variables
        if len(vars) == 1:
            data[vars[0].name] = vars[0].scale * self._grid
        else:
            for v, g in zip(vars, self._grid.components):
                data[v.name] = v.scale * g

        f = eval(expression, {}, data)
        if coords is None:
            return self.__getDomainValues(f)
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
    
    @property
    def materialSolution(self):
        result = {}
        def eval(c):
            sp = H1(self._mesh, order=1)
            gf = GridFunction(sp)
            gf.Set(c)
            return gf.vec
        
        for name, mat in self._mats.items():
            if len(mat.shape) == 0:
                result[name] = eval(mat)
            elif len(mat.shape) == 1:
                result[name] = [eval(mat[i]) for i in range(mat.shape[0])]
            elif len(mat.shape) == 2:
                result[name] = [[eval(mat[i,j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        return result