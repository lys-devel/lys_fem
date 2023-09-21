from .geometry import GeometryGenerator
from .mesh import OccMesher
from .material import Material
from .model import ElasticModelGenerator, FEMModel


class FEMProject:
    def __init__(self, dim):
        self._dim = dim
        self._geom = GeometryGenerator()
        self._mesher = OccMesher()
        self._materials = [Material("Material1", domains="all")]
        self._models = [ElasticModelGenerator()]

    def saveAsDictionary(self):
        d = {"dimension": self._dim}
        d["geometries"] = self._geom.saveAsDictionary()
        d["materials"] = [m.saveAsDictionary() for m in self._materials]
        d["models"] = [m.saveAsDictionary() for m in self._models]
        return d

    def loadFromDictionary(self, d):
        self._dim = d.get("dimension", 3)
        if "geometries" in d:
            self._geom = GeometryGenerator.loadFromDictionary(d["geometries"])
        if "materials" in d:
            self._materials = [Material.loadFromDictionary(dic) for dic in d["materials"]]
        if "models" in d:
            self._models = [FEMModel.loadFromDictionary(dic) for dic in d["models"]]

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

    def getMeshWave(self, dim=None, nomesh=False):
        if dim is None:
            dim = self._dim
        if nomesh:
            mesher = OccMesher()
        else:
            mesher = self._mesher
        return mesher.getMeshWave(self._geom.generateGeometry(), dim=dim)
