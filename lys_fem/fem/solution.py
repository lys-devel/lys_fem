import numpy as np
import glob
import os
import numpy as np
import ngsolve

from lys_fem import util
from . import mpi


class SolutionFields(dict):
    def add(self, name, path, expression, index=-1):
        self[name] = SolutionField(path, expression, index)

    def eval(self):
        return {key: util.SolutionFieldFunction(coef.get(), tdep=coef.index is None) for key, coef in self.items()}

    def update(self, step):
        for f in self.values():
            if f.index is None:
                f.update(step)

    def saveAsDictionary(self):
        return {key: value.saveAsDictionary() for key, value in self.items()}

    @classmethod
    def loadFromDictionary(cls, d):
        res = SolutionFields()
        for key, value in d.items():
            res[key] = SolutionField.loadFromDictionary(value)
        return res


class SolutionField:
    def __init__(self, path="", expression="", index=-1):
        self.set(path, expression, index)

    def set(self, path, expression, index):
        self._path = path
        self._expression = expression
        self._index = index
        self._sol = None

    def get(self):
        if self._index is None:
            return self.solution.coef(self.expression, -1, eval=False)
        else:
            return self.solution.coef(self.expression, self.index, eval=False)

    def update(self, step):
        self.solution.update(step)
    
    @property
    def solution(self):
        if self._sol is None:
            self._sol = FEMSolution(self._path)
        return self._sol

    @property
    def path(self):
        return self._path
    
    @property
    def expression(self):
        return self._expression
    
    @property
    def index(self):
        return self._index
    
    def saveAsDictionary(self):
        return {"path": self._path, "expression": self._expression, "index": self._index}
    
    @classmethod
    def loadFromDictionary(cls, d):
        return SolutionField(**d)


class FEMSolution:
    def __init__(self, path=".", solver="Solver0"):
        from .FEM import FEMProject
        self._fem = FEMProject.fromFile(path + "/input.dic")
        self._dirname = path+"/Solutions/"+solver
        self._index = None
        self._mats = self._fem.evaluator(solutions=True)
        self._model = self._fem.compositeModel
        self._last = (None, None, None)

    @property
    def maxIndex(self):
        return len(glob.glob(self._dirname+"/ngs*"))-len(glob.glob(self._dirname+"/ngs*_*"))-1

    def update(self, index=-1):
        if index < 0:
            index = self.maxIndex + index + 1
        mpath = self._dirname+"/ngs"+str(index)+"_mesh.msh"
        if os.path.exists(mpath):
            if index!=self._index:
                m = util.Mesh(mpath, self._fem.geometries.generateGeometry().scale)
                self.__generate(m)
        else:
            if self._index is None:
                self.__generate()
        self._index = index
        self._meshInfo = None
        self._sol.load(self._dirname+"/ngs"+str(index), self._fem.parallel)

    def __generate(self, mesh=None):
        if mesh is None:
            self._mesh = self._fem.mesh
        else:
            self._mesh = mesh
        self._fes = util.FiniteElementSpace(self._model.variables, self._mesh)
        self._sol = Solution(self._fes, nlog=1)

    def coef(self, expression, index=-1, eval=True):
        self.update(index)
        d = self._sol.replaceDict()
        res = self._mats[expression].replace(d)
        if eval:
            res = res.eval(self._fes)
        return res

    def eval(self, expression, data_number, coords=None):
        if self._last[1] != expression or self._last[2] != data_number:
            f = self.coef(expression, data_number)
            self._last = (f, expression, data_number)
        f = self._last[0]
        if coords is None:
            return self.__getDomainValues(f)
        else:
            if not hasattr(coords, "__iter__"):
                mip = self.__coordsToMIP(np.array([coords]))
                return np.vectorize(f)(mip)[0].squeeze()
            else:
                mip = self.__coordsToMIP(np.array(coords))
                return np.array(np.vectorize(f)(mip)).squeeze()
            
    def integrate(self, expression, data_number, **kwargs):
        f = self.coef(expression, data_number)
        return ngsolve.Integrate(f, self._mesh, **kwargs)

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
            

class Solution:
    """
    Solution class stores the solutions and the time derivatives as grid function.
    The NGSFunctions based on the grid function is also provided by this class.
    """
    def __init__(self, fes, value=None, nlog=2, old=None):
        self._fes = fes
        self._sols = [_Sol(fes) for _ in range(nlog)]
        self._nlog = nlog

        if value is not None:
            self.__update(_Sol(value))

        if old is not None:
            for i, s in enumerate(self._sols):
                s.set(old[i].project(self._fes))

    def __getitem__(self, index):
        return self._sols[index]

    def reset(self):
        zero = self._fes.gridFunction()
        for _ in range(len(self._sols)):
            self.__update(_Sol((self._sols[0][0], zero, zero)))

    def copyVector(self):
        g = self._fes.gridFunction()
        for v in self._fes.variables:
            if v.type == "x":
                g.setComponent(v, util.GridField(self._sols[0][0], v))
            if v.type == "v":
                g.setComponent(v, util.GridField(self._sols[0][1], v))
            if v.type == "a":
                g.setComponent(v, util.GridField(self._sols[0][2], v))
        return g
    
    def updateSolution(self, model, x0):
        tdep = model.updater()
        fes = self._fes

        x, v, a = fes.gridFunction(), fes.gridFunction(), fes.gridFunction()
        x.vec.data = x0.vec

        d = self.replaceDict(trial=False, prev=True)
        d.update({util.TrialFunction(v): util.GridField(x0, v) for v in self._fes.variables})

        for var in self._fes.variables:
            trial = util.TrialFunction(var)
            if trial in tdep:
                x.setComponent(var, tdep[trial].replace(d))
        for var in self._fes.variables:
            trial = util.TrialFunction(var)
            if trial.t in tdep:
                v.setComponent(var, tdep[trial.t].replace(d))
        for var in self._fes.variables:
            trial = util.TrialFunction(var)
            if trial.tt in tdep:
                a.setComponent(var, tdep[trial.tt].replace(d))

        self.__update(_Sol((x,v,a)))

    def __update(self, xva):
        for n in range(1, len(self._sols)):
            self._sols[-n].set(self._sols[-n+1])
        self._sols[0].set(xva)

    def replaceDict(self, trial=True, prev=False):
        """
        Returns a dictionary that replace trial functions with corresponding solutions.
        """
        res = {}
        for v in self._fes.variables:
            x = util.TrialFunction(v)
            if trial:
                res[v.trial] = util.GridField(self._sols[0][0], v)
            if prev:
                for n in range(self._nlog):
                    res[util.prev(x,n)] = util.GridField(self._sols[n][0], v)
                    res[util.prev(x.t,n)] = util.GridField(self._sols[n][1], v)
                    res[util.prev(x.tt,n)] = util.GridField(self._sols[n][2], v)
        return res

    def error(self, var):
        g = util.GridField(self._sols[0][0], var).error()
        return lambda x: g(self._fes, x)

    def save(self, path, mesh=False):
        self._sols[0].save(path, mesh=mesh)

    def load(self, path, parallel=None):
        self._sols[0].load(path, parallel=parallel)


class _Sol:
    def __init__(self, value):
        if isinstance(value, util.FiniteElementSpace):
            self._fes = value
            self._sols = (self._fes.gridFunction(), self._fes.gridFunction(), self._fes.gridFunction())
        else:
            self._fes = value[0].finiteElementSpace
            self._sols = value

    def __getitem__(self, index):
        return self._sols[index]
    
    def set(self, xva):
        if isinstance(xva, _Sol):
            xva = xva._sols
        for xi, yi in zip(self._sols, xva):
            if yi is not None:
                xi.vec.data = yi.vec
            else:
                xi.vec.data *= 0

    def project(self, fes):
        x, v, a = fes.gridFunction(), fes.gridFunction(), fes.gridFunction()
        for var in self._fes.variables:
            x.setComponent(var, util.GridField(self._sols[0], var))
            v.setComponent(var, util.GridField(self._sols[1], var))
            a.setComponent(var, util.GridField(self._sols[2], var))
        return x,v,a

    def save(self, path, mesh=False):
        self._sols[0].Save(path, parallel=mpi.isParallel())
        self._sols[1].Save(path+"_v", parallel=mpi.isParallel())
        self._sols[2].Save(path+"_a", parallel=mpi.isParallel())
        if mesh:
            self._fes.mesh.save(path+"_mesh.msh")

    def load(self, path, parallel=None):
        if parallel is None:
            parallel = mpi.isParallel()
        x, v, a = (self._fes.gridFunction(), self._fes.gridFunction(), self._fes.gridFunction())
        if os.path.exists(path):
            x.Load(path, parallel)
        if os.path.exists(path+"_v"):
            v.Load(path+"_v", parallel)
        if os.path.exists(path+"_a"):
            a.Load(path+"_a", parallel)
        self.set((x,v,a))


