from .base import FEMObjectList
from .parameters import Parameters
from .geometry import GeometryGenerator
from .mesh import OccMesher
from .material import Material, Materials
from .model import loadModel
from .solver import FEMSolver


class FEMProject:
    def __init__(self, dim):
        self.reset(dim)

    def reset(self, dim=3):
        self._dim = dim
        self._parallel = False
        self._params = Parameters()
        self._geom = GeometryGenerator()
        self._geom.setParent(self)
        self._mesher = OccMesher(self)
        self._materials = Materials(self, [Material(objName="Material1")])
        self._models = FEMObjectList(self)
        self._solvers = FEMObjectList(self)
        self._submit = {}

    def saveAsDictionary(self, parallel=False):
        d = {"dimension": self._dim, "parallel": parallel}
        d["parameters"] = self._params.saveAsDictionary()
        d["geometries"] = self._geom.saveAsDictionary()
        d["mesh"] = self._mesher.saveAsDictionary()
        d["materials"] = [m.saveAsDictionary() for m in self._materials]
        d["models"] = [m.saveAsDictionary() for m in self._models]
        d["solvers"] = [s.saveAsDictionary() for s in self._solvers]
        d["submit"] = self._submit
        return d

    def loadFromDictionary(self, d):
        self._dim = d.get("dimension", 3)
        self._parallel = d.get("parallel", False)
        if "parameters" in d:
            self._params = Parameters.loadFromDictionary(d["parameters"])
        if "geometries" in d:
            self._geom = GeometryGenerator.loadFromDictionary(d["geometries"])
            self._geom.setParent(self)
        if "mesh" in d:
            self._mesher = OccMesher.loadFromDictionary(d["mesh"])
            self._mesher.setParent(self)
        if "materials" in d:
            self._materials = Materials(self, [Material.loadFromDictionary(dic) for dic in d["materials"]])
        if "models" in d:
            self._models = FEMObjectList(self, [loadModel(dic) for dic in d["models"]])
        if "solvers" in d:
            self._solvers = FEMObjectList(self, [FEMSolver.loadFromDictionary(dic) for dic in d["solvers"]])
        if "submit" in d:
            self._submit = d["submit"]

    @classmethod
    def fromFile(cls, file):
        res = FEMProject(2)
        with open(file) as f:
            d = eval(f.read())
        res.loadFromDictionary(d)
        return res

    @property
    def dimension(self):
        return self._dim

    @property
    def parallel(self):
        return self._parallel
    
    @property
    def parameters(self):
        return self._params

    @property
    def geometries(self):
        return self._geom

    @property
    def mesher(self):
        return self._mesher

    @property
    def materials(self):
        return self._materials

    @property
    def models(self):
        return self._models

    @property
    def solvers(self):
        return self._solvers

    @property
    def submitSetting(self):
        return self._submit

    @property
    def domainAttributes(self):
        return self.geometries.geometryAttributes(self._dim)

    @property
    def boundaryAttributes(self):
        return self.geometries.geometryAttributes(self._dim-1)

    def getMeshWave(self, dim=None, nomesh=False):
        if dim is None:
            dim = self._dim
        if nomesh:
            mesher = OccMesher(self)
        else:
            mesher = self._mesher
        return mesher.getMeshWave(self._geom.generateGeometry(), dim=dim)

