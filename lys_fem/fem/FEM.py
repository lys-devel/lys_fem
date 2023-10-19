from .geometry import GeometryGenerator
from .mesh import OccMesher
from .material import Material
from .model import FEMModel
from .solver import FEMSolver


class FEMProject:
    def __init__(self, dim):
        self._dim = dim
        self._geom = GeometryGenerator()
        self._mesher = OccMesher()
        self._materials = [Material("Material1")]
        self._models = []
        self._solvers = []

    def saveAsDictionary(self):
        d = {"dimension": self._dim}
        d["geometries"] = self._geom.saveAsDictionary()
        d["materials"] = [m.saveAsDictionary() for m in self._materials]
        d["models"] = [m.saveAsDictionary() for m in self._models]
        d["solvers"] = [s.saveAsDictionary(self) for s in self._solvers]
        return d

    def loadFromDictionary(self, d):
        self._dim = d.get("dimension", 3)
        if "geometries" in d:
            self._geom = GeometryGenerator.loadFromDictionary(d["geometries"])
        if "materials" in d:
            self._materials = [Material.loadFromDictionary(dic) for dic in d["materials"]]
        if "models" in d:
            self._models = [FEMModel.loadFromDictionary(dic) for dic in d["models"]]
        if "solvers" in d:
            self._solvers = [FEMSolver.loadFromDictionary(self, dic) for dic in d["solvers"]]

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
    def geometryGenerator(self):
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

    def getMeshWave(self, dim=None, nomesh=False):
        if dim is None:
            dim = self._dim
        if nomesh:
            mesher = OccMesher()
        else:
            mesher = self._mesher
        return mesher.getMeshWave(self._geom.generateGeometry(), dim=dim)
