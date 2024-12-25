import glob
import os
import numpy as np
import ngsolve

from . import mpi, util
from .mesh import generateMesh
from .material import generateMaterial
from .models import generateModel
from .solver import _Sol


class NGSSolution:
    def __init__(self, fem, dirname):
        self._fem = fem
        self._dirname = dirname
        self._index = None

    @property
    def maxIndex(self):
        return len(glob.glob(self._dirname+"/ngs*"))-len(glob.glob(self._dirname+"/ngs*_*"))-1

    def coef(self, expression, index=-1):
        self.update(index)
        d = self._sol.replaceDict
        return self._mats[expression].replace(d).eval(self._fes)

    def update(self, index=-1):
        if index < 0:
            index = self.maxIndex + index + 1
        if os.path.exists(self._dirname+"/mesh"+str(index)+".msh"):
            if index!=self._index:
                self._fem.mesher.file = self._dirname+"/mesh"+str(index)+".msh"
                self.__generate()
        else:
            if self._index is None:
                self.__generate()
        self._index = index
        self._meshInfo = None
        sol_new = _Sol.load(self._fes, self._dirname+"/ngs"+str(index), self._fem.parallel)
        self._sol.set(sol_new)

    def __generate(self):
        self._mesh = generateMesh(self._fem)
        self._mats = generateMaterial(self._fem)
        model = generateModel(self._fem, self._mats)
        self._fes = util.FiniteElementSpace(model, self._mesh)
        self._sol = _Sol(self._fes)
        
    def eval(self, expression, index, coords=None):
        f = self.coef(expression, index)
        if coords is None:
            return self.__getDomainValues(f)
        else:
            if not hasattr(coords, "__iter__"):
                mip = self.__coordsToMIP(np.array([coords]))
                return np.array([f(mi) for mi in mip])[0].squeeze()
            else:
                mip = self.__coordsToMIP(np.array(coords))
                return np.array([f(mi) for mi in mip]).squeeze()
            
    def integrate(self, expression, index):
        f = self.coef(expression, index)
        return ngsolve.Integrate(f, self._mesh)

    def __getDomainValues(self, f):
        if mpi.isParallel():
            raise RuntimeError("Evaluation of solution on full mesh is only available for serial mode.")
        from lys import Wave

        if self._meshInfo is None:
            self._meshInfo = self.__exportMesh(self._mesh)
        domains, coords = self._meshInfo
        mip = [self._mesh(*c) for c in coords]
        data=np.array([f(mi) for mi in mip])
        res = []
        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 3-coords.shape[1]))])
        for domain in domains:
            nodes = np.unique([n for n in domain.values()])
            elems = {elem: np.searchsorted(nodes, n) for elem, n in domain.items()}
            res.append(Wave(data[nodes-1].squeeze(), coords[nodes-1], elements=elems))
        return res
    
    def __exportMesh(self, mesh):
        gmesh = mesh.ngmesh

        if gmesh.dim == 1:
            elems, types = gmesh.Elements1D(), {2: "line"}
        if gmesh.dim == 2:
            elems, types = gmesh.Elements2D(), {4: "quad", 3: "triangle"}
        if gmesh.dim == 3:
            elems, types = gmesh.Elements3D(), {4: "tetra", 5: "pyramid", 6: "prism", 8:"hexa"}

        result = []
        for mat in range(1, 1+ len(mesh.GetMaterials())):
            elements = {}
            for e in elems:
                if mat == e.index:
                    t = types[len(e.vertices)]
                    if t not in elements:
                        elements[t] = []
                    elements[t].append(tuple([v.nr for v in e.vertices]))
            result.append(elements)

        return result, np.array(gmesh.Coordinates())

    def __coordsToMIP(self, coords):
        dim = 0 if self._fem.dimension == 1 else 1
        if len(coords.shape) > dim:
            return [self.__coordsToMIP(c) for c in coords]
        else:
            if self._fem.dimension == 1:
                return self._mesh(coords)
            else:
                return self._mesh(*coords)